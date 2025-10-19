import argparse
from utils import file_utils
import json
import os
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra

from diffusion.trainer import Trainer

from models.diff_gpt import DiffusionAugmentedGPT


def main(args):
    model = DiffusionAugmentedGPT(dataset_name=args.dataset_name, prompt_cfg=args.prompt, diffusion_cfg=args.diffusion, scale_by_std=True,)

    with open_dict(args):
        args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        args.score_net_params = sum(p.numel() for p in model.score_net_ema.parameters() if p.requires_grad)
        args.soft_prompt_generator_params = sum(p.numel() for p in model.soft_prompt_generator.parameters() if p.requires_grad)

    if args.train_mode == 'prompt':
        train_cfg = args.prompt.train
    elif args.train_mode == 'diffusion':
        train_cfg = args.diffusion.train
    else:
        raise ValueError(f'Invalid train_mode: {args.train_mode}')

    trainer = Trainer(
        args=args,
        model=model,
        dataset_name=args.dataset_name,
        eval_dataset_name=args.eval_dataset_name,
        optimizer_name= train_cfg.optimizer,
        train_batch_size = train_cfg.train_batch_size,
        train_lr = train_cfg.learning_rate,
        train_num_steps = train_cfg.num_train_steps,
        lr_schedule = train_cfg.lr_schedule,
        num_warmup_steps = train_cfg.lr_warmup_steps,
        clip_grad_norm= train_cfg.clip_grad_norm,
        adam_betas = train_cfg.betas,
        independent_weight_decay= train_cfg.independent_weight_decay,
        gradient_accumulate_every = train_cfg.gradient_accumulation_steps,
        eval_every = train_cfg.eval_every,
        mixed_precision = train_cfg.mixed_precision,
    )

    if args.load_prompt_model is not None:
        trainer.load_prompt_model(args.load_prompt_model)

    if args.load_diffusion_model is not None:
        trainer.load_diffusion_model(args.load_diffusion_model)

    if args.resume_training:
        assert args.resume_dir is not None, "Must specify resume dir"
        trainer.load(args.resume_dir)
        
    if args.train_mode == "prompt":
        trainer.train_prompt()
    elif args.train_mode == "diffusion":
        trainer.train_diffusion()


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def launch(cfg: DictConfig) -> None:
    assert cfg.wandb_name is not None, "Must specify wandb project name"

    if cfg.load_prompt_model is not None:
        prompt_cfg = OmegaConf.load(os.path.join(cfg.load_prompt_model, 'args.yaml'))
        cfg.prompt = prompt_cfg.prompt
    
    if cfg.load_diffusion_model is not None:
        diffusion_cfg = OmegaConf.load(os.path.join(cfg.load_diffusion_model, 'args.yaml'))
        heldout_keys = {'sampling'}
        for k,v in diffusion_cfg.diffusion.arch.items():
            if k in heldout_keys:
                continue
            if k in cfg.diffusion.arch:
                cfg.diffusion.arch[k] = v
            else:
                print(f'Warning: {k} not found in config, skipping')

    main(cfg)

if __name__ == "__main__":        
    launch()
