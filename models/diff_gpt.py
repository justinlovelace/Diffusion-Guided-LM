import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from tqdm import tqdm
from collections import namedtuple
import math
import os

from ema_pytorch import EMA

from models.modules.diffusion import SinusoidalPosEmb
from models.modules.transformer import TransformerModel

from diffusion.noise_schedule import (
    get_scaled_noise_schedule,
    log_snr_to_alpha2,
    alpha2_to_shifted_log_snr,
)
from diffusion.time_sampler import LossEMASampler
from diffusion.diff_utils import (
    predict_noise_from_v,
    predict_start_from_v,
    predict_v_from_start_and_eps,
    predict_noise_from_start,
    predict_start_from_noise,
)
from diffusion.loss_weighting import (
    Asymmetric_LogNormal_V_Weighting,
    LogNormal_V_Weighting,
)

from text_datasets.CONSTANTS import DATA_STATS_PATH

ModelPrediction = namedtuple("ModelPrediction", ["pred_eps", "pred_x", "pred_v"])


def exists(val):
    return val is not None


@torch.cuda.amp.autocast(enabled=False)
def variance_preserving_map(x, alpha2, eps=None):
    if eps is None:
        eps = torch.randn_like(x)

    return alpha2.sqrt() * x + torch.sqrt(1 - alpha2) * eps


def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)


class ScoreNet(nn.Module):
    def __init__(
        self,
        model_arch="transformer",
        sentence_emb_dim=768,
        transformer_dim=768,
        n_layers=12,
        tok_emb_dim=1280,
        num_tokens=64,
        dropout=0.0,
        output_pooling="mean",
        output_dim_mult=4,
    ):
        super().__init__()
        time_emb_dim = transformer_dim // 2

        input_features = transformer_dim

        self.splicer = nn.Sequential(
            nn.Linear(sentence_emb_dim, sentence_emb_dim * 8),
            Rearrange("b (l d) -> b l d", l=num_tokens),
            nn.Linear(sentence_emb_dim * 8 // num_tokens, transformer_dim // 2),
        )

        self.context_splicer = nn.Sequential(
            nn.Linear(sentence_emb_dim, sentence_emb_dim * 4),
            Rearrange("b (l d) -> b l d", l=num_tokens),
            nn.Linear(sentence_emb_dim * 4 // num_tokens, transformer_dim // 2),
        )

        self.null_context_emb = nn.Parameter(torch.randn(sentence_emb_dim))

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(transformer_dim),
            nn.Linear(transformer_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_linear = nn.Linear(input_features, transformer_dim)
        if model_arch == "transformer":
            self.network = TransformerModel(
                dim=transformer_dim,
                num_layers=n_layers,
                causal=False,
                dim_head=64,
                time_emb_dim=time_emb_dim,
                dense=True,
                pos_emb="absolute",
                ff_dropout=dropout,
            )
        else:
            raise NotImplementedError

        if output_pooling == "inv_splicer":
            self.output_linear = nn.Sequential(
                nn.Linear(
                    transformer_dim, sentence_emb_dim * output_dim_mult // num_tokens
                ),
                Rearrange("b l d -> b (l d)"),
                nn.Linear(sentence_emb_dim * output_dim_mult, sentence_emb_dim),
            )
        else:
            raise ValueError(f"Unknown output pooling {output_pooling}")

    def forward(
        self,
        noised_sentence_emb,
        prompt_sentence_emb,
        alpha2,
        prompt_sentence_emb_mask=None,
    ):
        alpha2 = rearrange(alpha2, "b ()-> b")
        time_emb = self.time_mlp(alpha2 * 1000)
        time_emb = time_emb

        input_emb = self.splicer(noised_sentence_emb)

        if exists(prompt_sentence_emb_mask):
            assert exists(prompt_sentence_emb)
            prompt_sentence_emb[prompt_sentence_emb_mask] = self.null_context_emb
        if not exists(prompt_sentence_emb):
            prompt_sentence_emb = repeat(
                self.null_context_emb, "d -> b d", b=input_emb.shape[0]
            )

        context_emb = self.context_splicer(prompt_sentence_emb)

        input_emb = rearrange([input_emb, context_emb], "n b l d -> b l (n d)", n=2)

        tx_input = self.input_linear(input_emb)

        x = self.network(tx_input, time_emb=time_emb)
        x = self.output_linear(x)
        return x


class SoftPromptGenerator(nn.Module):
    def __init__(
        self,
        sentence_emb_dim=768,
        transformer_dim=768,
        prompt_length=8,
        n_layers=6,
        dropout=0.0,
    ):
        super(SoftPromptGenerator, self).__init__()
        self.splicer = nn.Sequential(
            nn.Linear(sentence_emb_dim, sentence_emb_dim * 4),
            Rearrange("b (l d) -> b l d", l=prompt_length),
            nn.Linear(sentence_emb_dim * 4 // prompt_length, transformer_dim),
        )

        time_emb_dim = sentence_emb_dim // 2
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(sentence_emb_dim),
            nn.Linear(sentence_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.transformer = TransformerModel(
            dim=transformer_dim,
            num_layers=n_layers,
            causal=False,
            pos_emb="absolute",
            time_emb_dim=time_emb_dim,
            ff_dropout=dropout,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(transformer_dim, 1280),
        )

    def forward(self, noised_sentence_emb, alpha2):
        assert alpha2 is not None
        alpha2 = rearrange(alpha2, "b ()-> b")
        time_emb = self.time_mlp(alpha2 * 1000)

        prompt = self.splicer(noised_sentence_emb)
        prompt = self.transformer(prompt, time_emb=time_emb)
        prompt = self.output_proj(prompt)
        return prompt


def get_diffusion_loss_weighting(weighting_name, **kwargs):
    if weighting_name == "asymmetric_lognormal_v":
        return Asymmetric_LogNormal_V_Weighting(**kwargs)
    elif weighting_name == "lognormal_v":
        return LogNormal_V_Weighting(**kwargs)
    else:
        raise ValueError(f"Unknown weighting name {weighting_name}")


class DiffusionAugmentedGPT(nn.Module):
    def __init__(
        self,
        dataset_name="c4",
        gpt2_model_name="gpt2-large",
        sentence_encoder_name="sentence-transformers/sentence-t5-xl",
        prompt_cfg=None,
        diffusion_cfg=None,
        gamma_min=-15,
        gamma_max=15,
        clf_guidance_dropout=0.1,
        scale_by_std=False,
    ):
        super(DiffusionAugmentedGPT, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Freeze gpt2
        if prompt_cfg.arch.freeze_gpt:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        # FP16 precision for sentence encoder
        self.sentence_encoder = SentenceTransformer(sentence_encoder_name).half()
        # Freeze sentence encoder
        for param in self.sentence_encoder.parameters():
            param.requires_grad = False

        # Prompt Generator
        self.soft_prompt_generator = SoftPromptGenerator(
            transformer_dim=prompt_cfg.arch.dim,
            prompt_length=prompt_cfg.arch.prompt_length,
            n_layers=prompt_cfg.arch.depth,
            dropout=prompt_cfg.arch.dropout,
        )
        self.noise_schedule = get_scaled_noise_schedule(
            prompt_cfg.augmentation.noise_schedule_name,
            scale=prompt_cfg.augmentation.noise_schedule_scale,
        )

        self.sample_noise_schedule = get_scaled_noise_schedule(
            diffusion_cfg.sampling.noise_schedule_name,
            scale=diffusion_cfg.sampling.noise_schedule_scale,
        )

        # Diffusion Network
        score_net = ScoreNet(
            model_arch=diffusion_cfg.arch.model_arch,
            transformer_dim=diffusion_cfg.arch.dim,
            n_layers=diffusion_cfg.arch.depth,
            num_tokens=diffusion_cfg.arch.num_tokens,
            dropout=diffusion_cfg.arch.dropout,
            output_pooling=diffusion_cfg.arch.output_pooling,
            output_dim_mult=diffusion_cfg.arch.output_dim_mult,
        )

        self.score_net_ema = EMA(score_net, beta=0.9999, update_every=1, power=3 / 4)

        # Optionally rescale data to have unit variance
        self.scale_by_std = scale_by_std
        self.register_buffer("data_mean", torch.full((768,), fill_value=0.0))
        self.register_buffer("data_std", torch.full((768,), fill_value=0.0))
        if scale_by_std:
            self.init_data_stats(dataset_name)
        self.adaptive_sampler = LossEMASampler(
            n_bins=100, ema_decay=0.9, gamma_min=gamma_min, gamma_max=gamma_max
        )
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.diffusion_loss_weighting = get_diffusion_loss_weighting(
            diffusion_cfg.loss.weighting_name, **diffusion_cfg.loss.weighting_kwargs
        )

        self.clf_guidance_dropout = torch.distributions.Bernoulli(
            probs=clf_guidance_dropout
        )

    def normalize_sentence_emb(self, sentence_emb):
        return (sentence_emb - self.data_mean) / self.data_std

    def unnormalize_sentence_emb(self, sentence_emb):
        return sentence_emb * self.data_std + self.data_mean

    def init_data_stats(self, dataset_name):
        """
        Initialize data mean and std from first batch of sentence embeddings
        sentence_emb: (n_batch, emb_dim)
        """
        self.data_mean = torch.load(
            os.path.join(DATA_STATS_PATH[dataset_name], "mean.pt")
        )
        self.data_std = torch.load(
            os.path.join(DATA_STATS_PATH[dataset_name], "std.pt")
        )

    def get_endpoints(self):
        return self.gamma_min, self.gamma_max

    def get_loss_emas(self):
        return self.adaptive_sampler.get_loss_emas()

    def get_unweighted_loss_emas(self):
        return self.adaptive_sampler.get_unweighted_loss_emas()

    def get_weighted_loss(self):
        return self.adaptive_sampler.weights().mean()

    def get_normalized_loss_emas(self):
        return self.adaptive_sampler.get_normalized_loss_emas()

    def get_cdf(self):
        return self.adaptive_sampler.get_cdf()

    def update_score_net_ema(self):
        self.score_net_ema.update()

    def get_sampling_timesteps(
        self, batch, sampling_timesteps, *, device, start_time=1.0
    ):
        times = torch.linspace(start_time, 0.0, sampling_timesteps + 1, device=device)
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def sample(
        self,
        input_ids,
        prompt_text,
        sampler="ddpm",
        var_lambda=0.2,
        sampling_timesteps=250,
        cls_free_guidance=1.0,
        cls_guidance=0.0,
        classifier=None,
        cls_target=1.0,
    ):
        if cls_guidance != 0.0:
            assert exists(classifier)
        batch = input_ids.shape[0]
        device = input_ids.device
        assert sampler in {"ddim", "ddpm"}
        assert var_lambda >= 0 and var_lambda <= 1.0

        time_pairs = self.get_sampling_timesteps(
            batch, sampling_timesteps=sampling_timesteps, device=device
        )

        z_t = torch.randn((batch, 768), device=device)

        x_start = None
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prompt_sentence_emb = self.sentence_encoder.encode(
                prompt_text,
                batch_size=batch,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            if self.scale_by_std:
                prompt_sentence_emb = self.normalize_sentence_emb(prompt_sentence_emb)
            else:
                prompt_sentence_emb = prompt_sentence_emb * math.sqrt(
                    prompt_sentence_emb.shape[-1]
                )

        for time, time_next in tqdm(
            time_pairs, desc="sampling loop time step", total=sampling_timesteps
        ):
            # get alpha sigma of time and next time
            alpha2 = self.sample_noise_schedule(time).unsqueeze(-1)
            alpha2_next = self.sample_noise_schedule(time_next).unsqueeze(-1)
            # get predicted x0

            context = torch.enable_grad() if cls_guidance != 0.0 else torch.no_grad()
            with context:
                model_output = self.diffusion_model_predictions(
                    z_t,
                    alpha2,
                    prompt_sentence_emb,
                    cls_free_guidance=cls_free_guidance,
                    rescale_x=(not self.scale_by_std),
                    ema=True,
                    cls_guidance=cls_guidance,
                    classifier=classifier,
                    cls_target=cls_target,
                )

            # calculate x0 and noise
            x_start = model_output.pred_x

            eps = model_output.pred_eps

            if time_next[0] <= 0:
                z_t = x_start
                continue

            # get noise
            if sampler == "ddim":
                z_t = x_start * alpha2_next.sqrt() + eps * (1 - alpha2_next).sqrt()
            elif sampler == "ddpm":
                # get noise
                noise = torch.randn_like(z_t)
                alpha2_now = alpha2 / alpha2_next

                min_var = torch.exp(
                    torch.log1p(-alpha2_next) - torch.log1p(-alpha2)
                ) * (1.0 - alpha2_now)
                max_var = 1.0 - alpha2_now
                sigma = torch.exp(
                    var_lambda * torch.log(max_var)
                    + (1 - var_lambda) * torch.log(min_var)
                )
                z_t = (
                    1
                    / alpha2_now.sqrt()
                    * (z_t - (1 - alpha2_now) / (1 - alpha2).sqrt() * eps)
                    + torch.sqrt(sigma) * noise
                )

        if self.scale_by_std:
            z_t = self.unnormalize_sentence_emb(z_t)
            # Scale pred_x to unit ball
            z_t = F.normalize(z_t, p=2, dim=-1)
        return z_t

    def prompt_forward(
        self,
        input_ids,
        labels,
        continuation_text,
        diffusion_token_mask,
        continuation_emb=None,
        return_prompt_emb=False,
        alpha2=None,
    ):
        n_batch = input_ids.shape[0]

        with torch.no_grad():
            assert not (exists(continuation_emb) and exists(continuation_text))
            if exists(continuation_emb):
                sentence_emb = continuation_emb
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    sentence_emb = self.sentence_encoder.encode(
                        continuation_text,
                        batch_size=n_batch,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    )
            if self.scale_by_std:
                sentence_emb = self.normalize_sentence_emb(sentence_emb)
            else:
                sentence_emb = sentence_emb * math.sqrt(sentence_emb.shape[-1])

            input_embed = self.gpt2.transformer.wte(input_ids)
            if alpha2 is not None:
                noised_sentence_emb = variance_preserving_map(sentence_emb, alpha2)
            else:
                time = torch.rand(size=(n_batch, 1), device=input_ids.device)
                alpha2 = self.noise_schedule(time)
                noised_sentence_emb = variance_preserving_map(sentence_emb, alpha2)

        soft_prompt = self.soft_prompt_generator(noised_sentence_emb, alpha2).float()

        input_embed[diffusion_token_mask] = rearrange(soft_prompt, "b l d -> (b l) d")

        outputs = self.gpt2(inputs_embeds=input_embed, labels=labels)

        if return_prompt_emb:
            return outputs.loss, input_embed

        return outputs.loss

    def diffusion_model_predictions(
        self,
        z_t,
        alpha2,
        prompt_sentence_emb,
        prompt_sentence_emb_mask=None,
        cls_free_guidance=1.0,
        rescale_x=False,
        ema=False,
        cls_guidance=0.0,
        classifier=None,
        cls_target=1.0,
        cls_guidance_n_samples=32,
    ):
        # https://proceedings.mlr.press/v202/song23k/song23k.pdf
        score_net = (
            self.score_net_ema.ema_model if ema else self.score_net_ema.online_model
        )
        if cls_guidance != 0.0:
            assert exists(classifier)
            z_t.requires_grad = True
        model_output = score_net(
            z_t, prompt_sentence_emb, alpha2, prompt_sentence_emb_mask
        )
        if cls_free_guidance != 1.0:
            unc_model_output = score_net(
                z_t,
                prompt_sentence_emb=None,
                alpha2=alpha2,
                prompt_sentence_emb_mask=None,
            )
            model_output = model_output * cls_free_guidance + unc_model_output * (
                1 - cls_free_guidance
            )

        pred_v = model_output
        pred_x = predict_start_from_v(z_t, pred_v, alpha2)
        pred_eps = predict_noise_from_v(z_t, pred_v, alpha2)
        if cls_guidance != 0.0:
            assert exists(classifier)
            # Check if classifier is a tuple
            if isinstance(classifier, tuple):
                pass
            else:
                classifier = (classifier,)
                cls_target = (cls_target,)
            loss_estimates = []
            for cls, targ in zip(classifier, cls_target):
                assert cls is not None
                sigma2 = 1 - alpha2
                pred_sent_emb = repeat(
                    pred_x, "b d -> (b n) d", n=cls_guidance_n_samples
                )
                sample_var = torch.exp((torch.log(sigma2) - torch.log(alpha2)))
                sample_var = repeat(
                    sample_var, "b 1-> (b n) 1", n=cls_guidance_n_samples
                )
                pred_sent_emb = pred_sent_emb + (sample_var.sqrt()) * torch.randn_like(
                    pred_sent_emb
                )
                pred_sent_emb = self.unnormalize_sentence_emb(pred_sent_emb)
                # Scale pred_x to unit ball
                pred_sent_emb = F.normalize(pred_sent_emb, p=2, dim=-1)
                if targ == 0.0:
                    target = torch.zeros((pred_x.shape[0],), device=pred_x.device)
                elif targ == 1.0:
                    target = torch.ones((pred_x.shape[0],), device=pred_x.device)
                else:
                    raise ValueError(f"Invalid cls_target {targ}")
                # Repeat pred_sent_emb cls_guidance_n_samples times

                target = repeat(target, "b -> b n", n=cls_guidance_n_samples)
                pred_sent_emb = rearrange(
                    pred_sent_emb, "(b n) d -> b n d", n=cls_guidance_n_samples
                )
                cls_loss = cls.get_loss(pred_sent_emb, target, disable_reduction=True)
                mc_loss_estimate = (
                    torch.logsumexp(cls_loss, dim=-1) - math.log(cls_guidance_n_samples)
                ).sum()
                loss_estimates.append(mc_loss_estimate)
                # mc_loss_estimate = reduce(cls_loss, 'b n -> b', 'mean').sum()
            loss_estimates = torch.stack(loss_estimates)
            mc_loss_estimate = loss_estimates.sum()
            grad = torch.autograd.grad(mc_loss_estimate, z_t)[0]
            pred_eps = pred_eps + cls_guidance * sigma2.sqrt() * grad

            pred_x = predict_start_from_noise(z_t, pred_eps, alpha2)

            return ModelPrediction(pred_eps, pred_x, None)

        if rescale_x:
            assert not self.scale_by_std
            # Scale pred_x to sqrt(d) ball
            pred_x = F.normalize(pred_x, p=2, dim=-1) * math.sqrt(pred_x.shape[-1])
            pred_eps = predict_noise_from_start(z_t, pred_x, alpha2)
            pred_v = predict_v_from_start_and_eps(pred_x, pred_eps, alpha2)
        else:
            pred_eps = predict_noise_from_v(z_t, pred_v, alpha2)

        return ModelPrediction(pred_eps, pred_x, pred_v)

    def diffusion_forward(
        self, input_ids, prompt_text, continuation_text, alpha2=None, ema=False
    ):
        n_batch = input_ids.shape[0]
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                sentence_emb = self.sentence_encoder.encode(
                    prompt_text + continuation_text,
                    batch_size=n_batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            if self.scale_by_std:
                sentence_emb = self.normalize_sentence_emb(sentence_emb)
            else:
                sentence_emb = sentence_emb * math.sqrt(sentence_emb.shape[-1])
            assert sentence_emb.shape[0] == n_batch * 2
            prompt_sentence_emb = sentence_emb[:n_batch]
            continuation_sentence_emb = sentence_emb[n_batch:]

            if alpha2 is None:
                gamma, density = self.adaptive_sampler.sample(
                    batch_size=n_batch, device=input_ids.device
                )
                alpha2 = log_snr_to_alpha2(gamma)
                alpha2 = rearrange(alpha2, "b -> b ()")
            else:
                density = None
                gamma = alpha2_to_shifted_log_snr(alpha2)
                gamma = gamma.squeeze()

            eps = torch.randn_like(continuation_sentence_emb)
            noised_sentence_emb = variance_preserving_map(
                continuation_sentence_emb, alpha2, eps=eps
            )
        # Need to clone prompt_sentence_emb to avoid gradient issues
        prompt_sentence_emb = prompt_sentence_emb.clone()
        prompt_sentence_emb_mask = (
            self.clf_guidance_dropout.sample((n_batch, 1)).bool().squeeze()
        )

        model_preds = self.diffusion_model_predictions(
            noised_sentence_emb,
            alpha2,
            prompt_sentence_emb,
            prompt_sentence_emb_mask,
            ema=ema,
        )
        v_pred = model_preds.pred_v
        v_target = predict_v_from_start_and_eps(continuation_sentence_emb, eps, alpha2)

        unweighted_loss = F.mse_loss(v_pred, v_target, reduction="none")
        unweighted_loss = reduce(unweighted_loss, "b d -> b", "mean")

        diffusion_loss_weighting = self.diffusion_loss_weighting.v_loss_weighting(
            gamma=gamma
        ).squeeze()
        weighted_loss = diffusion_loss_weighting * unweighted_loss
        # Update loss ema
        # Disable autocast for this step
        if self.training:
            with torch.cuda.amp.autocast(enabled=False):
                self.adaptive_sampler.update_with_all_losses(
                    gamma.squeeze(), weighted_loss
                )
                self.adaptive_sampler.update_with_all_unweighted_losses(
                    gamma.squeeze(), unweighted_loss
                )
        if exists(density):
            # Monte-carlo training loss
            monte_carlo_weighted_loss = (
                torch.exp(torch.log(diffusion_loss_weighting) - torch.log(density))
                * unweighted_loss
            )
            return (monte_carlo_weighted_loss).mean()

        return weighted_loss.mean()

    def forward(
        self,
        input_ids,
        labels,
        prompt_text,
        continuation_text,
        diffusion_token_mask,
        return_prompt_emb=False,
        continuation_emb=None,
        alpha2=None,
        mode="prompt",
    ):
        if mode == "prompt":
            return self.prompt_forward(
                input_ids,
                labels,
                continuation_text,
                diffusion_token_mask,
                return_prompt_emb=return_prompt_emb,
                alpha2=alpha2,
                continuation_emb=continuation_emb,
            )
        elif mode == "diffusion":
            return self.diffusion_forward(
                input_ids, prompt_text, continuation_text, alpha2=alpha2
            )
        else:
            raise ValueError(f"Unknown mode {mode}")
