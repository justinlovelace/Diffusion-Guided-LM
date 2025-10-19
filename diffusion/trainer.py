import math
from pathlib import Path
import random
from collections import defaultdict, OrderedDict
import os
import numpy as np
import json
from omegaconf import OmegaConf, open_dict
import itertools


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from einops import rearrange, repeat

from tqdm.auto import tqdm

from transformers import get_scheduler, GPT2LMHeadModel

from accelerate import Accelerator
import wandb


import models.optimization.optimizer as optimizer
from utils.torch_utils import compute_grad_norm
import utils.file_utils as file_utils
from text_datasets.dataset_utils import get_dataset, DataCollatorWithDiffusionTokens
from evaluation.wrappers import (
    compute_olmo_perplexity,
)
from diffusion.noise_schedule import log_snr_to_alpha2

# helpers functions


def exists(x):
    return x is not None


def cycle(dl):
    while True:
        for data in dl:
            yield data


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class Trainer(object):
    def __init__(
        self,
        args,
        model,
        dataset_name,
        eval_dataset_name,
        *,
        optimizer_name="adamw",
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        lr_schedule="cosine",
        num_warmup_steps=500,
        clip_grad_norm=0.0,
        adam_betas=(0.9, 0.99),
        independent_weight_decay=0.01,
        eval_every=50,
        save_every=5000,
        mixed_precision="no",
    ):
        super().__init__()
        self.args = args
        self.clip_grad_norm = clip_grad_norm

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            log_with="wandb",
        )
        self.num_devices = self.accelerator.num_processes
        with open_dict(args):
            args.num_devices = self.num_devices

        if self.accelerator.is_main_process:
            if args.get("output_dir", None) is None:
                with open_dict(args):
                    args.output_dir = file_utils.get_output_dir(args)
                with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
                    OmegaConf.save(config=args, f=f)
            results_folder = args.output_dir
            run = "diffusion-guided-lm"
            if args.eval_mode != "test_pipeline":
                self.accelerator.init_trackers(
                    run,
                    config=OmegaConf.to_container(args),
                    init_kwargs={
                        "wandb": {"dir": results_folder, "name": args.wandb_name}
                    },
                )

        self.model = model

        self.eval_every = eval_every
        self.save_every = save_every

        self.train_batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        self.num_diffusion_tokens = args.prompt.arch.prompt_length
        dataset = get_dataset(
            dataset_name,
            self.model.tokenizer,
            num_diffusion_tokens=self.num_diffusion_tokens,
        )
        if exists(eval_dataset_name):
            eval_dataset = get_dataset(
                eval_dataset_name,
                self.model.tokenizer,
                num_diffusion_tokens=self.num_diffusion_tokens,
            )
            dataset["test"] = eval_dataset["test"]

        # Shuffle dataset
        dataset["test"].shuffle(seed=42)

        diffusion_data_collator = DataCollatorWithDiffusionTokens(
            self.model.tokenizer, num_diffusion_tokens=self.num_diffusion_tokens
        )
        val_diffusion_data_collator = DataCollatorWithDiffusionTokens(
            self.model.tokenizer,
            num_diffusion_tokens=self.num_diffusion_tokens,
            validation=True,
        )
        self.dataloader = DataLoader(
            dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=diffusion_data_collator,
        )
        self.val_dataloader = DataLoader(
            dataset["test"],
            batch_size=self.train_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=val_diffusion_data_collator,
        )

        # optimizer
        if args.train_mode == "prompt":
            trainable_params = itertools.chain(
                self.model.soft_prompt_generator.parameters(),
                self.model.gpt2.parameters(),
            )
        elif args.train_mode == "diffusion":
            trainable_params = self.model.score_net_ema.parameters()
        else:
            # For evaluation mode, use dummy parameters
            trainable_params = self.model.parameters()
        self.opt = optimizer.get_adamw_optimizer(
            trainable_params,
            lr=train_lr,
            betas=adam_betas,
            independent_weight_decay=independent_weight_decay,
        )

        # scheduler
        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps * self.num_devices,
            num_training_steps=train_num_steps * self.num_devices,
        )

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        (
            self.model,
            self.opt,
            self.dataloader,
            self.val_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.opt, self.dataloader, self.val_dataloader, lr_scheduler
        )
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)
        self.reference_dict = {}

    def save(self, best=False):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
            "scheduler": self.lr_scheduler.state_dict(),
        }
        if best:
            torch.save(data, str(self.results_folder / "best_model.pt"))
        else:
            torch.save(data, str(self.results_folder / "model.pt"))

    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def _log_diffusion_training_metrics(self, total_loss, grad_norm):
        """Helper function to log basic diffusion training metrics"""
        return {
            "diffusion/train/loss": total_loss,
            "diffusion/train/weighted_loss_ema": self.model.get_weighted_loss().item(),
            "diffusion/learning_rate": self.lr_scheduler.get_last_lr()[0],
            "diffusion/grad_norm": grad_norm,
            "diffusion/step": self.step,
            "diffusion/epoch": (self.step * self.gradient_accumulate_every)
            / len(self.dataloader),
            "diffusion/samples": self.step
            * self.train_batch_size
            * self.gradient_accumulate_every
            * self.num_devices,
        }

    def _log_diffusion_validation_metrics(self, logs, loss, uniform_loss):
        """Helper function to log diffusion validation metrics and plots"""
        logs["diffusion/val/loss"] = loss.item()
        logs["diffusion/val/uniform_loss"] = uniform_loss.item()

        # Log loss EMA plots
        loss_emas_dict = self.model.get_loss_emas()
        data = [[(x[0] + x[1]) / 2, y] for (x, y) in loss_emas_dict.items()]
        table = wandb.Table(data=data, columns=["LogSNR", "Loss"])
        logs["diffusion/train/plots/loss_ema"] = wandb.plot.line(
            table, "LogSNR", "Loss", title="Loss EMA Across LogSNR"
        )

        # Log normalized loss EMA plots
        loss_emas_dict = self.model.get_normalized_loss_emas()
        data = [[(x[0] + x[1]) / 2, y] for (x, y) in loss_emas_dict.items()]
        table = wandb.Table(data=data, columns=["LogSNR", "Density"])
        logs["diffusion/train/plots/normalized_loss_ema"] = wandb.plot.line(
            table, "LogSNR", "Density", title="Normalized Loss EMA Across LogSNR"
        )

        # Log unweighted loss EMA
        loss_emas_dict = self.model.get_unweighted_loss_emas()
        for key, value in loss_emas_dict.items():
            logs[
                f"diffusion/train/ema/unweighted_ema_loss/{key[0]:.2f}-{key[0]:.2f}"
            ] = value
        data = [[(x[0] + x[1]) / 2, y] for (x, y) in loss_emas_dict.items()]
        table = wandb.Table(data=data, columns=["LogSNR", "Loss"])
        logs["diffusion/train/plots/unweighted_loss_ema"] = wandb.plot.line(
            table, "LogSNR", "Loss", title="Unweighted Loss EMA Across LogSNR"
        )

        # Log CDF plot
        cdf_dict = self.model.get_cdf()
        data = [[(x[0] + x[1]) / 2, y] for (x, y) in cdf_dict.items()]
        table = wandb.Table(data=data, columns=["LogSNR", "CDF"])
        logs["diffusion/train/plots/cdf"] = wandb.plot.line(
            table, "LogSNR", "CDF", title="CDF Across LogSNR"
        )

    def _log_prompt_training_metrics(self, total_loss, grad_norm):
        """Helper function to log basic prompt training metrics"""
        return {
            "train/loss": total_loss,
            "train/perplexity": math.exp(total_loss),
            "learning_rate": self.lr_scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
            "step": self.step,
            "epoch": (self.step * self.gradient_accumulate_every)
            / len(self.dataloader),
            "samples": self.step
            * self.train_batch_size
            * self.gradient_accumulate_every
            * self.num_devices,
        }

    def _log_prompt_validation_metrics(self, logs, loss, clean_loss, shuffled_loss):
        """Helper function to log prompt validation metrics"""
        logs["val/loss"] = loss.item()
        logs["val/perplexity"] = math.exp(loss.item())
        logs["val/clean_loss"] = clean_loss.item()
        logs["val/clean_perplexity"] = math.exp(clean_loss.item())
        logs["val/shuffled_loss"] = shuffled_loss.item()
        logs["val/shuffled_perplexity"] = math.exp(shuffled_loss.item())

    def load_prompt_model(self, prompt_model_path):
        # Init diffusion.encoder and diffusion.decoder with model at ar_model_path
        prev_state_dict = torch.load(
            os.path.join(prompt_model_path, f"model.pt"),
            map_location=self.accelerator.device,
        )["model"]
        model = self.unwrapped_model()
        current_state_dict = model.state_dict()
        for k in current_state_dict.keys():
            if "soft_prompt_generator." in k or "gpt2." in k:
                current_state_dict[k] = prev_state_dict[k]

        model.load_state_dict(current_state_dict)

    def load_diffusion_model(self, diffusion_model_path):
        # Init diffusion.encoder and diffusion.decoder with model at ar_model_path
        prev_state_dict = torch.load(
            os.path.join(diffusion_model_path, f"model.pt"),
            map_location=self.accelerator.device,
        )["model"]
        model = self.unwrapped_model()
        current_state_dict = model.state_dict()
        for k in current_state_dict.keys():
            if "score_net" in k:
                current_state_dict[k] = prev_state_dict[k]

        model.load_state_dict(current_state_dict)

    def load(self, file_path=None, best=False, init_only=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        if best:
            data = torch.load(str(file_path / f"best_model.pt"), map_location=device)
        else:
            data = torch.load(str(file_path / f"model.pt"), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        # For backwards compatibility with earlier models
        model.load_state_dict(data["model"], strict=False)
        self.opt.load_state_dict(data["opt"])

        if init_only:
            return
        self.step = data["step"]

        self.lr_scheduler.load_state_dict(data["scheduler"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    @torch.no_grad()
    def validate_diffusion(
        self, cls_free_guidance=1.0, num_samples=2500, generation_eval=False, seed=None, sampling_timesteps=None
    ):
        if exists(seed):
            set_seeds(seed)
        self.model.eval()
        unwrapped_model = self.unwrapped_model()
        sampled_embeddings = []
        ref_embeddings = []
        num_sampled = 0
        for val_data in self.val_dataloader:
            sample_kwargs = {
                "cls_free_guidance": cls_free_guidance,
            }
            if sampling_timesteps is not None:
                sample_kwargs["sampling_timesteps"] = sampling_timesteps
            sampled_continuation = unwrapped_model.sample(
                val_data["input_ids"],
                prompt_text=val_data["prompt"],
                **sample_kwargs,
            )
            sampled_embeddings.append(sampled_continuation.detach().cpu())
            # Store pretrained sentence encoder in fp16
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                ref_continuation_embeddings = self.model.sentence_encoder.encode(
                    val_data["continuation"],
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            ref_embeddings.append(ref_continuation_embeddings.detach().cpu())
            num_sampled += sampled_continuation.shape[0]
            if num_sampled >= num_samples:
                break
        sampled_embeddings = torch.cat(sampled_embeddings, dim=0)[:num_samples]
        ref_embeddings = torch.cat(ref_embeddings, dim=0)[:num_samples]

        # compute cosine-similarities
        val_logs = {
            f"diffusion/val_gen/guidance_{cls_free_guidance:.1f}/gen_similarity": F.cosine_similarity(
                ref_embeddings, sampled_embeddings
            )
            .mean()
            .item(),
        }
        val_logs[
            f"diffusion/val_gen/guidance_{cls_free_guidance:.1f}/gen_shuffled_similarity"
        ] = (
            F.cosine_similarity(
                ref_embeddings,
                sampled_embeddings[torch.randperm(sampled_embeddings.shape[0])],
            )
            .mean()
            .item()
        )

        # Log values
        if generation_eval:
            sampled_continuation2 = unwrapped_model.sample(
                val_data["input_ids"],
                prompt_text=val_data["prompt"],
                **sample_kwargs,
            )
            # Decode last val batch
            # Set alpha2 to 0.8
            alpha2 = torch.full(
                (self.train_batch_size, 1), 0.8, device=self.accelerator.device
            )
            text_gen_continuations = []
            text_gen2_continuations = []
            text_gt_continuations = []
            text_unprompted_continuations = []
            gt_loss, gt_emb_batch = self.model(
                val_data["input_ids"],
                labels=val_data["labels"],
                prompt_text=val_data["prompt"],
                continuation_text=val_data["continuation"],
                diffusion_token_mask=val_data["diffusion_token_mask"],
                return_prompt_emb=True,
                alpha2=alpha2,
            )

            gen_loss, gen_emb_batch = self.model(
                val_data["input_ids"],
                labels=val_data["labels"],
                prompt_text=val_data["prompt"],
                continuation_text=None,
                continuation_emb=sampled_continuation,
                diffusion_token_mask=val_data["diffusion_token_mask"],
                return_prompt_emb=True,
                alpha2=alpha2,
            )

            gen2_loss, gen2_emb_batch = self.model(
                val_data["input_ids"],
                labels=val_data["labels"],
                prompt_text=val_data["prompt"],
                continuation_text=None,
                continuation_emb=sampled_continuation2,
                diffusion_token_mask=val_data["diffusion_token_mask"],
                return_prompt_emb=True,
                alpha2=alpha2,
            )

            clean_loss = self.model.gpt2(
                val_data["clean_input_ids"], labels=val_data["clean_labels"]
            ).loss

            for idx in range(gt_emb_batch.shape[0]):
                batched_embs = torch.stack(
                    [
                        gt_emb_batch[idx, : val_data["continuation_start"][idx] + 8],
                        gen_emb_batch[idx, : val_data["continuation_start"][idx] + 8],
                        gen2_emb_batch[idx, : val_data["continuation_start"][idx] + 8],
                    ]
                )
                gen_ids = self.model.gpt2.generate(
                    inputs_embeds=batched_embs,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    max_length=64,
                    repetition_penalty=1.2,
                )
                text_list = self.model.tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True
                )
                if val_data["continuation_start"][idx] == 0:
                    unprompted_text_list = ["No prompt."]
                else:
                    unprompted_gen_ids = self.model.gpt2.generate(
                        inputs_embeds=gt_emb_batch[
                            idx : idx + 1, : val_data["continuation_start"][idx]
                        ],
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self.model.tokenizer.eos_token_id,
                        max_length=96,
                    )
                    unprompted_text_list = self.model.tokenizer.batch_decode(
                        unprompted_gen_ids, skip_special_tokens=True
                    )
                text_gt_continuations.append(text_list[0])
                text_gen_continuations.append(text_list[1])
                text_gen2_continuations.append(text_list[2])
                text_unprompted_continuations.append(unprompted_text_list[0])

            # Compute perplexity of prompt with decoded continuation
            gen_perplexity = compute_olmo_perplexity(
                val_data["prompt"][:len(text_gen_continuations)],
                [[cont] for cont in text_gen_continuations]
            )['olmo_perplexity']
            gt_perplexity = compute_olmo_perplexity(
                val_data["prompt"][:len(text_gt_continuations)],
                [[cont] for cont in text_gt_continuations]
            )['olmo_perplexity']
            ref_perplexity = compute_olmo_perplexity(
                val_data["prompt"][:len(val_data["continuation"])],
                [[cont] for cont in val_data["continuation"]]
            )['olmo_perplexity']
            unprompted_perplexity = compute_olmo_perplexity(
                val_data["prompt"][:len(text_unprompted_continuations)],
                [[cont] for cont in text_unprompted_continuations]
            )['olmo_perplexity']
            # Add to val logs
            val_logs[
                f"diffusion/val_gen/guidance_{cls_free_guidance:.1f}/gen_diff_perplexity"
            ] = gen_perplexity
            val_logs[
                f"diffusion/val_gen/guidance_{cls_free_guidance:.1f}/gen_gt_perplexity"
            ] = gt_perplexity
            val_logs[
                f"diffusion/val_gen/guidance_{cls_free_guidance:.1f}/gen_ref_perplexity"
            ] = ref_perplexity
            val_logs[
                f"diffusion/val_gen/guidance_{cls_free_guidance:.1f}/gen_unprompted_perplexity"
            ] = unprompted_perplexity

            # Log prompt, continuation, decoded_continuations
            if self.accelerator.is_main_process:
                # Log results
                columns = [
                    "prompts",
                    "continuations",
                    "gen_continuations",
                    "gen2_continuations",
                    "gt_continuations",
                    "unprompted_continuations",
                ]
                data = []
                for i in range(len(text_gt_continuations)):
                    row = [
                        val_data["prompt"][i],
                        val_data["continuation"][i],
                        text_gen_continuations[i],
                        text_gen2_continuations[i],
                        text_gt_continuations[i],
                        text_unprompted_continuations[i],
                    ]
                    data.append(row)
                table = wandb.Table(columns=columns, data=data)
                self.accelerator.log(
                    {f"Samples_guidance_{cls_free_guidance:.1f}": table}, self.step
                )

        # Log values
        if self.accelerator.is_main_process:
            # Print logs
            self.accelerator.print(
                f"Validation results for guidance {cls_free_guidance:.1f}:"
            )
            for key, value in val_logs.items():
                self.accelerator.print(f"{key}: {value:.3f}")
            self.accelerator.log(val_logs, self.step)

    @torch.no_grad()
    def validate_pipeline(
        self,
        cls_free_guidance=1.0,
        sampling_timesteps=250,
        num_samples=2500,
        seed=42,
        alpha2=0.8,
    ):
        if exists(seed):
            set_seeds(seed)
        self.model.eval()
        unwrapped_model = self.unwrapped_model()
        sampled_embeddings = []
        sampled_embeddings2 = []
        ref_embeddings = []
        num_sampled = 0
        alpha2_tensor = torch.full(
            (self.train_batch_size, 1), alpha2, device=self.accelerator.device
        )
        text_gen_continuations = []
        text_gen2_continuations = []
        text_gt_continuations = []
        text_unprompted_continuations = []

        config_prefix = (
            f"diffusion/val_gen/guidance_{cls_free_guidance:.1f}/alpha_{alpha2:.2f}"
        )

        for val_data in self.val_dataloader:
            sampled_continuation = unwrapped_model.sample(
                val_data["input_ids"],
                prompt_text=val_data["prompt"],
                cls_free_guidance=cls_free_guidance,
                sampling_timesteps=sampling_timesteps,
            )
            sampled_embeddings.append(sampled_continuation.detach().cpu())
            sampled_continuation2 = unwrapped_model.sample(
                val_data["input_ids"],
                prompt_text=val_data["prompt"],
                cls_free_guidance=cls_free_guidance,
                sampling_timesteps=sampling_timesteps,
            )
            sampled_embeddings2.append(sampled_continuation2.detach().cpu())

            # Store pretrained sentence encoder in fp16
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                ref_continuation_embeddings = self.model.sentence_encoder.encode(
                    val_data["continuation"],
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            ref_embeddings.append(ref_continuation_embeddings.detach().cpu())
            num_sampled += sampled_continuation.shape[0]

            gt_loss, gt_emb_batch = self.model(
                val_data["input_ids"],
                labels=val_data["labels"],
                prompt_text=val_data["prompt"],
                continuation_text=val_data["continuation"],
                diffusion_token_mask=val_data["diffusion_token_mask"],
                return_prompt_emb=True,
                alpha2=alpha2_tensor,
            )

            gen_loss, gen_emb_batch = self.model(
                val_data["input_ids"],
                labels=val_data["labels"],
                prompt_text=val_data["prompt"],
                continuation_text=None,
                continuation_emb=sampled_continuation,
                diffusion_token_mask=val_data["diffusion_token_mask"],
                return_prompt_emb=True,
                alpha2=alpha2_tensor,
            )

            gen2_loss, gen2_emb_batch = self.model(
                val_data["input_ids"],
                labels=val_data["labels"],
                prompt_text=val_data["prompt"],
                continuation_text=None,
                continuation_emb=sampled_continuation2,
                diffusion_token_mask=val_data["diffusion_token_mask"],
                return_prompt_emb=True,
                alpha2=alpha2_tensor,
            )

            for idx in range(gt_emb_batch.shape[0]):
                batched_embs = torch.stack(
                    [
                        gt_emb_batch[idx, : val_data["continuation_start"][idx] + 8],
                        gen_emb_batch[idx, : val_data["continuation_start"][idx] + 8],
                        gen2_emb_batch[idx, : val_data["continuation_start"][idx] + 8],
                    ]
                )
                gen_ids = self.model.gpt2.generate(
                    inputs_embeds=batched_embs,
                    do_sample=True,
                    num_beams=1,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    max_new_tokens=32,
                    top_p=0.9,
                    repetition_penalty=1.2,
                )
                text_list = self.model.tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True
                )
                if val_data["continuation_start"][idx] == 0:
                    unprompted_text_list = ["No prompt."]
                else:
                    unprompted_gen_ids = self.model.gpt2.generate(
                        inputs_embeds=gt_emb_batch[
                            idx : idx + 1, : val_data["continuation_start"][idx]
                        ],
                        do_sample=True,
                        num_beams=1,
                        pad_token_id=self.model.tokenizer.eos_token_id,
                        max_new_tokens=32,
                        top_p=0.9,
                        repetition_penalty=1.2,
                    )
                    unprompted_text_list = self.model.tokenizer.batch_decode(
                        unprompted_gen_ids, skip_special_tokens=True
                    )
                text_gt_continuations.append(text_list[0])
                text_gen_continuations.append(text_list[1])
                text_gen2_continuations.append(text_list[2])

                text_unprompted_continuations.append(unprompted_text_list[0])
                num_sampled += 1
            if num_sampled >= num_samples:
                break

        sampled_embeddings = torch.cat(sampled_embeddings, dim=0)[:num_samples]
        ref_continuation_embeddings = torch.cat(ref_embeddings, dim=0)[:num_samples]
        text_gt_continuations = text_gt_continuations[:num_samples]
        text_gen_continuations = text_gen_continuations[:num_samples]
        text_gen2_continuations = text_gen2_continuations[:num_samples]
        text_unprompted_continuations = text_unprompted_continuations[:num_samples]

        # Store pretrained sentence encoder in fp16
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            diffusion_continuation_embeddings = (
                self.model.sentence_encoder.encode(
                    text_gen_continuations,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .detach()
                .cpu()
            )
            unprompted_continuation_embeddings = (
                self.model.sentence_encoder.encode(
                    text_unprompted_continuations,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .detach()
                .cpu()
            )
            gt_continuation_embeddings = (
                self.model.sentence_encoder.encode(
                    text_gt_continuations,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                .detach()
                .cpu()
            )

        val_logs = {}

        # Compute perplexity of prompt with decoded continuation
        gen_perplexity = compute_olmo_perplexity(
            val_data["prompt"][:len(text_gen_continuations)],
            [[cont] for cont in text_gen_continuations]
        )['olmo_perplexity']
        gt_perplexity = compute_olmo_perplexity(
            val_data["prompt"][:len(text_gt_continuations)],
            [[cont] for cont in text_gt_continuations]
        )['olmo_perplexity']
        ref_perplexity = compute_olmo_perplexity(
            val_data["prompt"][:len(val_data["continuation"])],
            [[cont] for cont in val_data["continuation"]]
        )['olmo_perplexity']
        unprompted_perplexity = compute_olmo_perplexity(
            val_data["prompt"][:len(text_unprompted_continuations)],
            [[cont] for cont in text_unprompted_continuations]
        )['olmo_perplexity']

        # Add to val logs
        val_logs[f"{config_prefix}/perplexity/diff_perplexity"] = gen_perplexity
        val_logs[f"{config_prefix}/perplexity/gt_perplexity"] = gt_perplexity
        val_logs[f"{config_prefix}/perplexity/ref_perplexity"] = ref_perplexity
        val_logs[f"{config_prefix}/perplexity/unprompted_perplexity"] = (
            unprompted_perplexity
        )

        # Log prompt, continuation, decoded_continuations
        if self.accelerator.is_main_process:
            # Log results
            columns = [
                "prompts",
                "continuations",
                "gen_continuations",
                "gen2_continuations",
                "gt_continuations",
                "unprompted_continuations",
            ]
            data = []
            for i in range(len(text_gt_continuations)):
                row = [
                    val_data["prompt"][i],
                    val_data["continuation"][i],
                    text_gen_continuations[i],
                    text_gen2_continuations[i],
                    text_gt_continuations[i],
                    text_unprompted_continuations[i],
                ]
                data.append(row)
            table = wandb.Table(columns=columns, data=data)
            self.accelerator.log(
                {f"{config_prefix}/samples/Generations": table}, self.step
            )

        # Log values
        if self.accelerator.is_main_process:
            # Print logs
            self.accelerator.print(
                f"Validation results for guidance {cls_free_guidance:.1f}:"
            )
            for key, value in val_logs.items():
                # Skip wandb plots
                if "classifier_preds" in key:
                    continue
                self.accelerator.print(f"{key}: {value:.3f}")
            self.accelerator.log(val_logs, self.step)
            # Log values
        if self.accelerator.is_main_process:
            gen_dict = defaultdict(list)
            gen_dict["prompt"] = val_data["prompt"]
            gen_dict["gen_continuation"] = text_gen_continuations
            # Save gen_dict to json file at self.results_dir
            file_path = os.path.join(
                str(self.results_folder), config_prefix, f"gen_dict_{self.step}.json"
            )
            # create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(gen_dict, f)

    @torch.no_grad()
    def test_pipeline(
        self,
        sampling_kwargs,
        cls_free_guidance=1.0,
        seed=42,
        alpha2=0.8,
        cls_guidance=0.0,
        classifier=None,
        cls_target=0.0,
        samples_per_prompt=25,
        cache_only=False,
        cache_dir=None,
        max_new_tokens=20,
        prefix_str="",
    ):
        if exists(seed):
            set_seeds(seed)
        self.model.eval()
        unwrapped_model = self.unwrapped_model()
        sampled_embeddings = []
        ref_embeddings = []
        num_sampled = 0
        alpha2_tensor = torch.full(
            (samples_per_prompt, 1), alpha2, device=self.accelerator.device
        )
        gen_dict = defaultdict(list)

        pretrained_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2-large")
        pretrained_gpt2.to(self.accelerator.device)
        if exists(cache_dir):
            # Load cached embeddings
            file_path = os.path.join(cache_dir, f"sampled_embeddings_{self.step}.pt")
            sampled_embeddings = torch.load(file_path)

        config_prefix = (
            f"diffusion/val_gen/guidance_{cls_free_guidance:.1f}/alpha_{alpha2:.2f}/cls-guidance_{cls_guidance:.2f}_cls-target_{cls_target}/seed_{seed}"
            + prefix_str
        )
        print(f"Number of validation batches: {len(self.val_dataloader)}")
        for val_data in tqdm(self.val_dataloader, total=len(self.val_dataloader)):
            if exists(cache_dir):
                sampled_continuation = (
                    sampled_embeddings[
                        num_sampled : num_sampled + val_data["input_ids"].shape[0]
                    ]
                    .clone()
                    .to(self.accelerator.device)
                )
                num_sampled += sampled_continuation.shape[0]
            else:
                continuations = []
                for idx in range(samples_per_prompt):
                    continuations.append(
                        unwrapped_model.sample(
                            val_data["input_ids"],
                            prompt_text=val_data["prompt"],
                            cls_free_guidance=cls_free_guidance,
                            cls_guidance=cls_guidance,
                            classifier=classifier,
                            cls_target=cls_target,
                            **sampling_kwargs,
                        )
                    )
                sampled_continuation = rearrange(continuations, "n b l -> b n l")
                sampled_embeddings.append(sampled_continuation.detach().cpu())

                num_sampled += sampled_continuation.shape[0]

            if cache_only:
                continue
            for idx in range(sampled_continuation.shape[0]):
                gen_sentence_embs = sampled_continuation[idx]
                input_ids = repeat(
                    val_data["input_ids"][idx], "l -> n l", n=gen_sentence_embs.shape[0]
                )
                labels = repeat(
                    val_data["labels"][idx], "l -> n l", n=gen_sentence_embs.shape[0]
                )
                diffusion_token_mask = repeat(
                    val_data["diffusion_token_mask"][idx],
                    "l -> n l",
                    n=gen_sentence_embs.shape[0],
                )
                gen_loss, gen_emb_batch = self.model(
                    input_ids,
                    labels=labels,
                    prompt_text=None,
                    continuation_text=None,
                    continuation_emb=gen_sentence_embs,
                    diffusion_token_mask=diffusion_token_mask,
                    return_prompt_emb=True,
                    alpha2=alpha2_tensor,
                )

                batched_embs = gen_emb_batch[
                    :, : val_data["continuation_start"][idx] + 8
                ]
                gen_ids = self.model.gpt2.generate(
                    inputs_embeds=batched_embs,
                    do_sample=True,
                    num_beams=1,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    top_p=0.9,
                    repetition_penalty=1.2,
                )
                text_list = self.model.tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True
                )
                gen_dict["prompt"].append(val_data["prompt"][idx])
                gen_dict["continuation"].append(val_data["continuation"][idx])
                gen_dict["gen_continuation"].append(text_list)
            print(f"Number of samples: {num_sampled}")
            if num_sampled >= 1000:
                break

        # Log values
        if self.accelerator.is_main_process:
            if cache_dir is not None:
                # Save gen_dict to json file at self.results_dir
                file_path = os.path.join(
                    cache_dir,
                    f"gen_dict_{self.step}_alpha_{alpha2:.2f}_{num_sampled}samples.json",
                )
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # create directory if it doesn't exist
                with open(file_path, "w") as f:
                    json.dump(gen_dict, f)
                return
            # Save sampled_embeddings to file
            sampled_embeddings = torch.cat(sampled_embeddings, dim=0)
            file_path = os.path.join(
                str(self.results_folder),
                config_prefix,
                f"sampled_embeddings_{self.step}.pt",
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            torch.save(sampled_embeddings, file_path)
            if cache_only:
                return
            # Save gen_dict to json file at self.results_dir
            file_path = os.path.join(
                str(self.results_folder), config_prefix, f"gen_dict_{self.step}.json"
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # create directory if it doesn't exist
            with open(file_path, "w") as f:
                json.dump(gen_dict, f)

    @torch.no_grad()
    def validate_prompt(self, alpha2=None, seed=42):
        set_seeds(seed)

        decoded_continuations = []
        decoded_unprompted_continuations = []
        decoded_positive_continuations = []
        decoded_negative_continuations = []
        self.model.eval()
        alpha2_str = f"_alpha2_{alpha2:.1f}" if exists(alpha2) else ""
        if exists(alpha2):
            alpha2 = torch.full(
                (self.train_batch_size, 1), alpha2, device=self.accelerator.device
            )
        total_loss = 0.0
        total_clean_loss = 0.0
        for val_data in self.val_dataloader:
            loss, input_emb_batch = self.model(
                val_data["input_ids"],
                labels=val_data["labels"],
                prompt_text=val_data["prompt"],
                continuation_text=val_data["continuation"],
                diffusion_token_mask=val_data["diffusion_token_mask"],
                return_prompt_emb=True,
                alpha2=alpha2,
            )
            total_loss += loss.item()

            clean_loss = self.model.gpt2(
                val_data["clean_input_ids"], labels=val_data["clean_labels"]
            ).loss
            total_clean_loss += clean_loss.item()

            for idx, input_emb in enumerate(input_emb_batch):
                gen_ids = self.model.gpt2.generate(
                    inputs_embeds=input_emb_batch[
                        idx : idx + 1, : val_data["continuation_start"][idx] + 8
                    ],
                    do_sample=True,
                    num_beams=1,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    max_new_tokens=20,
                    top_p=0.9,
                )
                text_list = self.model.tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True
                )
                decoded_continuations.extend(text_list)

                if val_data["continuation_start"][idx] == 0:
                    decoded_unprompted_continuations.append("No prompt.")
                    continue

                un_prompted_gen_ids = self.model.gpt2.generate(
                    inputs_embeds=input_emb_batch[
                        idx : idx + 1, : val_data["continuation_start"][idx]
                    ],
                    do_sample=True,
                    num_beams=1,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    max_new_tokens=20,
                    top_p=0.9,
                )
                unprompted_text_list = self.model.tokenizer.batch_decode(
                    un_prompted_gen_ids, skip_special_tokens=True
                )
                decoded_unprompted_continuations.extend(unprompted_text_list)
            break

        val_logs = {
            f"prompt/val_gen/loss{alpha2_str}": total_loss,
            f"prompt/val_gen/ref_perplexity{alpha2_str}": math.exp(total_loss),
            f"prompt/val_gen/clean_loss{alpha2_str}": total_clean_loss,
            f"prompt/val_gen/ref_perplexity_clean{alpha2_str}": math.exp(
                total_clean_loss
            ),
        }

        # Store pretrained sentence encoder in fp16
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            ref_continuation_embeddings = self.model.sentence_encoder.encode(
                val_data["continuation"],
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            decoded_continuation_embeddings = self.model.sentence_encoder.encode(
                decoded_continuations, convert_to_tensor=True, show_progress_bar=False
            )
            decoded_unprompted_embeddings = self.model.sentence_encoder.encode(
                decoded_unprompted_continuations,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        # compute cosine-similarities
        val_logs[f"prompt/val_gen/gen_similarity{alpha2_str}"] = (
            F.cosine_similarity(
                ref_continuation_embeddings, decoded_continuation_embeddings
            )
            .mean()
            .item()
        )
        val_logs[f"prompt/val_gen/gen_similarity_unprompted{alpha2_str}"] = (
            F.cosine_similarity(
                ref_continuation_embeddings, decoded_unprompted_embeddings
            )
            .mean()
            .item()
        )

        # Compute perplexity of prompt with decoded continuation

        perplexity = compute_olmo_perplexity(
            val_data["prompt"][:len(decoded_continuations)],
            [[cont] for cont in decoded_continuations]
        )['olmo_perplexity']
        perplexity_unprompted = compute_olmo_perplexity(
            val_data["prompt"][:len(decoded_unprompted_continuations)],
            [[cont] for cont in decoded_unprompted_continuations]
        )['olmo_perplexity']
        val_logs[f"prompt/val_gen/gen_perplexity{alpha2_str}"] = perplexity
        val_logs[f"prompt/val_gen/gen_perplexity_unprompted{alpha2_str}"] = (
            perplexity_unprompted
        )

        # Log prompt, continuation, decoded_continuations
        if self.accelerator.is_main_process:
            # Log results
            columns = [
                "prompts",
                "continuations",
                "decoded_continuations",
                "unprompted_continuations",
            ]
            data = []
            for i in range(len(decoded_continuations)):
                row = [
                    val_data["prompt"][i],
                    val_data["continuation"][i],
                    decoded_continuations[i],
                    decoded_unprompted_continuations[i],
                ]
                data.append(row)
            table = wandb.Table(columns=columns, data=data)
            self.accelerator.log({f"Samples{alpha2_str}": table}, self.step)

            self.accelerator.log(val_logs, self.step)

        # Save results to json
        if self.accelerator.is_main_process:
            results = {
                "prompt": val_data["prompt"],
                "continuation": val_data["continuation"],
                "decoded_continuations": decoded_continuations,
                "decoded_unprompted_continuations": decoded_unprompted_continuations,
            }
            with open(
                os.path.join(self.args.output_dir, f"results{alpha2_str}.json"), "w"
            ) as f:
                json.dump(results, f, indent=2)

    def train_diffusion(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                for grad_accum_step in range(self.gradient_accumulate_every):
                    data = next(self.data_iter)
                    loss = self.model(
                        data["input_ids"],
                        labels=data["labels"],
                        prompt_text=data["prompt"],
                        continuation_text=data["continuation"],
                        diffusion_token_mask=data["diffusion_token_mask"],
                        mode="diffusion",
                    )
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()
                    self.accelerator.backward(loss)

                grad_norm = compute_grad_norm(self.model.parameters())
                accelerator.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )
                accelerator.wait_for_everyone()
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad(set_to_none=True)
                self.model.update_score_net_ema()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    logs = self._log_diffusion_training_metrics(total_loss, grad_norm)

                if self.step % self.eval_every == 0:
                    self.model.eval()
                    val_data = next(self.val_iter)
                    with torch.no_grad():
                        loss = self.model(
                            val_data["input_ids"],
                            labels=val_data["labels"],
                            prompt_text=val_data["prompt"],
                            continuation_text=val_data["continuation"],
                            diffusion_token_mask=val_data["diffusion_token_mask"],
                            mode="diffusion",
                        )

                        gamma_min, gamma_max = self.unwrapped_model().get_endpoints()
                        gammas = (
                            torch.rand(
                                (val_data["input_ids"].shape[0], 1), device=device
                            )
                            * (gamma_max - gamma_min)
                            + gamma_min
                        )
                        alpha2 = log_snr_to_alpha2(gammas)
                        uniform_loss = self.model(
                            val_data["input_ids"],
                            labels=val_data["labels"],
                            prompt_text=val_data["prompt"],
                            continuation_text=val_data["continuation"],
                            diffusion_token_mask=val_data["diffusion_token_mask"],
                            alpha2=alpha2,
                            mode="diffusion",
                        )
                    self.model.train()
                    if accelerator.is_main_process:
                        self._log_diffusion_validation_metrics(logs, loss, uniform_loss)
                        pbar.set_postfix(
                            OrderedDict(
                                [
                                    ("name", self.args.wandb_name),
                                    (
                                        "weighted_loss",
                                        self.model.get_weighted_loss().item(),
                                    ),
                                ]
                            )
                        )
                        accelerator.log(logs, step=self.step)
                if self.step % self.save_every == 0:
                    guidances = [1.0, 3.0]
                    for guidance in guidances:
                        self.validate_diffusion(cls_free_guidance=guidance)
                    self.save()
                pbar.update(1)

                if accelerator.is_main_process:
                    accelerator.log(logs, step=self.step)
            accelerator.wait_for_everyone()
        self.save()
        accelerator.print("training complete")

    def train_prompt(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                for grad_accum_step in range(self.gradient_accumulate_every):
                    data = next(self.data_iter)
                    loss = self.model(
                        data["input_ids"],
                        labels=data["labels"],
                        prompt_text=data["prompt"],
                        continuation_text=data["continuation"],
                        diffusion_token_mask=data["diffusion_token_mask"],
                    )
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()
                    self.accelerator.backward(loss)

                grad_norm = compute_grad_norm(self.model.parameters())
                accelerator.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )
                accelerator.wait_for_everyone()
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad(set_to_none=True)
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    logs = self._log_prompt_training_metrics(total_loss, grad_norm)

                if self.step % self.eval_every == 0:
                    self.model.eval()
                    val_data = next(self.val_iter)
                    with torch.no_grad():
                        loss = self.model(
                            val_data["input_ids"],
                            labels=val_data["labels"],
                            prompt_text=val_data["prompt"],
                            continuation_text=val_data["continuation"],
                            diffusion_token_mask=val_data["diffusion_token_mask"],
                        )
                        # Shuffle continuation text
                        shuffled_continuation = val_data["continuation"].copy()
                        random.shuffle(shuffled_continuation)
                        shuffled_loss = self.model(
                            val_data["input_ids"],
                            labels=val_data["labels"],
                            prompt_text=val_data["prompt"],
                            continuation_text=shuffled_continuation,
                            diffusion_token_mask=val_data["diffusion_token_mask"],
                        )

                        clean_loss = self.model.gpt2(
                            val_data["clean_input_ids"], labels=val_data["clean_labels"]
                        ).loss
                    self.model.train()
                    if accelerator.is_main_process:
                        self._log_prompt_validation_metrics(
                            logs, loss, clean_loss, shuffled_loss
                        )
                    pbar.set_postfix(
                        OrderedDict(
                            [("name", self.args.wandb_name), ("loss", total_loss)]
                        )
                    )
                if self.step % self.save_every == 0:
                    self.save()
                pbar.update(1)

                if accelerator.is_main_process:
                    accelerator.log(logs, step=self.step)
            accelerator.wait_for_everyone()
        self.save()
        accelerator.print("training complete")
