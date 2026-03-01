#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Train SAR2RGB via ControlNet on Text2Earth.

Fine-tune a ControlNet branch conditioned on SAR imagery, using the pre-trained
Text2Earth text2image diffusion model as the base. The ControlNet is initialized
from the UNet and trained while freezing VAE, text encoder, and UNet.

Based on: libs/diffusers/examples/controlnet/train_controlnet.py
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from examples.text2earth_sar2rgb.dataset_utils import (
    MavicTSAR2RGBDataset,
    load_sar2rgb_train_records,
)

if is_wandb_available():
    import wandb

logger = get_logger(__name__)

DEFAULT_TEXT2EARTH_PATH = "models/lcybuaa/Text2Earth"


def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet for SAR2RGB on Text2Earth.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=DEFAULT_TEXT2EARTH_PATH,
        help="Path to Text2Earth or compatible SD model.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained ControlNet (optional). If not set, init from UNet.",
    )
    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default="models/lcybuaa/Text2Earth/vae",
        help="VAE model to use. Default: models/lcybuaa/Text2Earth/vae.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ckpt/text2earth_sar2rgb_controlnet",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--refined_root",
        type=str,
        default=None,
        help="MAVIC-T refined dataset root.",
    )
    parser.add_argument(
        "--sar2rgb_sup_manifest",
        type=str,
        default="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_sar2rgb_sup.txt",
        help="Path to sar2rgb supervised manifest.",
    )
    parser.add_argument(
        "--paired_val_manifest",
        type=str,
        default="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2rgb.txt",
        help="Path to paired val manifest (excluded from train).",
    )
    parser.add_argument(
        "--exclude_file",
        type=str,
        default="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt",
        help="Path to bad samples exclude file.",
    )
    parser.add_argument(
        "--use_augmented",
        action="store_true",
        default=True,
        help="Include sar2rgb_crop_aug in training.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="18_GOOGLE_LEVEL_ a satellite optical image",
        help="Caption for conditioning. Use N_GOOGLE_LEVEL_ prefix for resolution: N in [10,18] maps to 2^(17-N)m (e.g. 17→1m, 18→0.5m).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training resolution.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Override num_train_epochs if set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=0,
        help="Text2Earth resolution class label (0 for 512).",
    )
    return parser.parse_args()


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str = None):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    conditioning_pixel_values = torch.stack([e["conditioning_pixel_values"] for e in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([e["input_ids"] for e in examples])
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir) / "logs"
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # Load models
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing ControlNet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing ControlNet from UNet")
        controlnet = ControlNetModel.from_unet(unet)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if accelerate.__version__ >= "0.16.0":
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                while len(weights) > 0:
                    weights.pop()
                    model = models[-1]
                    model.save_pretrained(os.path.join(output_dir, "controlnet"))

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention and is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate *= (
            args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("Install bitsandbytes for 8-bit Adam: pip install bitsandbytes")
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset
    records = load_sar2rgb_train_records(
        refined_root=args.refined_root,
        sar2rgb_sup_manifest=args.sar2rgb_sup_manifest,
        exclude_file=args.exclude_file,
        paired_val_manifest=args.paired_val_manifest,
        use_augmented=args.use_augmented,
    )
    train_dataset = MavicTSAR2RGBDataset(
        records=records,
        resolution=args.resolution,
        caption=args.caption,
        use_random_crop=True,
        use_horizontal_flip=True,
        use_vertical_flip=False,
        tokenizer=tokenizer,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_training_steps = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps,
    )

    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    has_class_embedding = getattr(unwrap_model(unet), "class_embedding", None) is not None
    class_label = args.class_label

    if accelerator.is_main_process:
        accelerator.init_trackers("text2earth_sar2rgb_controlnet", config=vars(args))

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path:
            accelerator.print(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, math.ceil(args.max_train_steps / num_update_steps_per_epoch)):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(weight_dtype)

                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(accelerator.device), return_dict=False
                )[0]
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                unet_kwargs = dict(
                    down_block_additional_residuals=[s.to(weight_dtype) for s in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(weight_dtype),
                    return_dict=False,
                )
                if has_class_embedding:
                    class_labels = torch.full(
                        (bsz,), class_label, dtype=torch.long, device=latents.device
                    )
                    unet_kwargs["class_labels"] = class_labels

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    **unet_kwargs,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]),
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for d in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, d))
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)
        logger.info(f"Saved final ControlNet to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
