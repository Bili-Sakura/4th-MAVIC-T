#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The DDBM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Training script for Denoising Diffusion Bridge Models (DDBM) using
# the Hugging Face accelerate library for distributed training.
#
# This script follows the diffusers training script conventions for
# easy integration with the Hugging Face ecosystem.

"""
Training script for DDBM models using accelerate and diffusers patterns.

Example usage:
    # Single GPU training
    python scripts/train_ddbm_diffusers.py \
        --dataset_name edges2handbags \
        --data_dir /path/to/data \
        --output_dir ./outputs/ddbm-e2h \
        --resolution 64 \
        --train_batch_size 32 \
        --num_epochs 100 \
        --learning_rate 1e-4 \
        --pred_mode vp

    # Multi-GPU training with accelerate
    accelerate launch scripts/train_ddbm_diffusers.py \
        --dataset_name edges2handbags \
        --data_dir /path/to/data \
        --output_dir ./outputs/ddbm-e2h \
        --mixed_precision fp16
"""

import argparse
import logging
import math
import os
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm

from ddbm import DDBMScheduler, DDBMPipeline
from ddbm.utils.script_util import create_model, model_and_diffusion_defaults


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a DDBM model for image-to-image translation."
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="edges2handbags",
        choices=["edges2handbags", "diode"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Image resolution for training.",
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="adm",
        choices=["adm", "edm"],
        help="UNet architecture type: 'adm' or 'edm'.",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=128,
        help="Base number of channels in UNet.",
    )
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="Number of residual blocks per resolution.",
    )
    parser.add_argument(
        "--attention_resolutions",
        type=str,
        default="32,16,8",
        help="Comma-separated list of resolutions for attention layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate in the model.",
    )
    parser.add_argument(
        "--condition_mode",
        type=str,
        default="concat",
        help="How to condition on source image: 'concat' or None.",
    )

    # Scheduler arguments
    parser.add_argument(
        "--pred_mode",
        type=str,
        default="vp",
        choices=["ve", "vp", "ve_simple", "vp_simple"],
        help="Prediction mode for diffusion.",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80.0,
        help="Maximum sigma value. Use 1.0 for VP mode.",
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Minimum sigma value.",
    )
    parser.add_argument(
        "--beta_d",
        type=float,
        default=2.0,
        help="Beta_d parameter for VP schedule.",
    )
    parser.add_argument(
        "--beta_min",
        type=float,
        default=0.1,
        help="Beta_min parameter for VP schedule.",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/ddbm",
        help="Directory to save model checkpoints and logs.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size per device for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. Overrides num_epochs if set.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer.",
    )

    # EMA arguments
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for model weights.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate.",
    )

    # Logging and saving arguments
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="Logging backend to use.",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=10,
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--save_images_epochs",
        type=int,
        default=10,
        help="Generate and save sample images every N epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save training state every N steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )

    # Hardware arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of data loading workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank.",
    )

    # Sampling arguments (for evaluation)
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of inference steps for sampling.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="Guidance weight for sampling.",
    )
    parser.add_argument(
        "--churn_step_ratio",
        type=float,
        default=0.33,
        help="Churn step ratio for stochastic sampling.",
    )

    args = parser.parse_args()

    # Handle environment variable for local_rank
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Adjust sigma_max for VP mode
    if args.pred_mode.startswith("vp") and args.sigma_max == 80.0:
        args.sigma_max = 1.0
        logger.info("Adjusted sigma_max to 1.0 for VP mode")

    return args


def get_dataset(args):
    """Load the dataset based on configuration."""
    if args.dataset_name == "edges2handbags":
        from datasets.aligned_dataset import EdgesDataset
        train_dataset = EdgesDataset(
            dataroot=args.data_dir,
            train=True,
            img_size=args.resolution,
            random_crop=True,
            random_flip=True,
        )
        val_dataset = EdgesDataset(
            dataroot=args.data_dir,
            train=True,
            img_size=args.resolution,
            random_crop=False,
            random_flip=False,
        )
    elif args.dataset_name == "diode":
        from datasets.aligned_dataset import DIODE
        train_dataset = DIODE(
            dataroot=args.data_dir,
            train=True,
            img_size=args.resolution,
            random_crop=True,
            random_flip=True,
            disable_cache=True,
        )
        val_dataset = DIODE(
            dataroot=args.data_dir,
            train=True,
            img_size=args.resolution,
            random_crop=False,
            random_flip=False,
            disable_cache=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    return train_dataset, val_dataset


def preprocess_batch(batch, device):
    """Preprocess a batch of data."""
    x0 = batch[0].to(device) * 2 - 1  # Scale to [-1, 1]
    x_T = batch[1].to(device) * 2 - 1  # Scale to [-1, 1]
    return x0, x_T


def compute_training_loss(model, scheduler, x0, x_T, sigma_data=0.5, pred_mode="vp"):
    """
    Compute the training loss for DDBM.

    Args:
        model: The UNet model.
        scheduler: The DDBM scheduler.
        x0: Clean target samples (batch_size, channels, height, width).
        x_T: Source/condition samples (batch_size, channels, height, width).
        sigma_data: Data standard deviation.
        pred_mode: Prediction mode ('ve', 'vp', etc.).

    Returns:
        The training loss tensor.
    """
    batch_size = x0.shape[0]
    device = x0.device
    dtype = x0.dtype

    # Sample random timesteps (sigmas)
    # For DDBM, we sample sigmas from the bridge distribution
    sigma_min = scheduler.config.sigma_min
    sigma_max = scheduler.config.sigma_max
    rho = scheduler.config.rho

    # Sample sigmas using Karras schedule distribution
    u = torch.rand(batch_size, device=device, dtype=dtype)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = (sigma_max - 1e-4) ** (1 / rho)
    sigmas = (max_inv_rho + u * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = torch.clamp(sigmas, max=sigma_max)

    # Sample noise
    noise = torch.randn_like(x0)

    # Add noise using bridge process
    noisy_samples = scheduler.add_noise(x0, noise, sigmas, x_T)

    # Get bridge scalings
    c_skip, c_out, c_in = get_bridge_scalings(
        sigmas, sigma_data, sigma_max,
        scheduler.config.beta_d, scheduler.config.beta_min,
        pred_mode
    )

    # Expand dimensions for broadcasting
    dims = x0.ndim
    c_skip = append_dims(c_skip, dims)
    c_out = append_dims(c_out, dims)
    c_in = append_dims(c_in, dims)

    # Rescale timesteps for model input
    rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)

    # Model forward pass
    model_output = model(c_in * noisy_samples, rescaled_t, xT=x_T)

    # Compute denoised prediction
    denoised = c_out * model_output + c_skip * noisy_samples

    # Compute loss weights
    weights = get_loss_weights(sigmas, sigma_data, sigma_max,
                               scheduler.config.beta_d, scheduler.config.beta_min,
                               pred_mode)
    weights = append_dims(weights, dims)

    # MSE loss
    loss = F.mse_loss(denoised, x0, reduction="none")
    loss = (loss * weights).mean()

    return loss


def get_bridge_scalings(sigma, sigma_data, sigma_max, beta_d, beta_min, pred_mode):
    """Get the bridge scalings (c_skip, c_out, c_in) for the denoiser."""
    sigma_data_end = sigma_data
    cov_xy = 0.0
    c = 1

    if pred_mode == 've':
        A = (sigma**4 / sigma_max**4 * sigma_data_end**2 +
             (1 - sigma**2 / sigma_max**2)**2 * sigma_data**2 +
             2 * sigma**2 / sigma_max**2 * (1 - sigma**2 / sigma_max**2) * cov_xy +
             c**2 * sigma**2 * (1 - sigma**2 / sigma_max**2))
        c_in = 1 / A ** 0.5
        c_skip = ((1 - sigma**2 / sigma_max**2) * sigma_data**2 +
                  sigma**2 / sigma_max**2 * cov_xy) / A
        c_out = ((sigma / sigma_max)**4 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) +
                 sigma_data**2 * c**2 * sigma**2 * (1 - sigma**2 / sigma_max**2))**0.5 * c_in
        return c_skip, c_out, c_in

    elif pred_mode == 'vp':
        logsnr_t = vp_logsnr(sigma, beta_d, beta_min)
        logsnr_T = vp_logsnr(torch.tensor(1.0), beta_d, beta_min)
        logs_t = vp_logs(sigma, beta_d, beta_min)
        logs_T = vp_logs(torch.tensor(1.0), beta_d, beta_min)

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()

        A = a_t**2 * sigma_data_end**2 + b_t**2 * sigma_data**2 + 2 * a_t * b_t * cov_xy + c**2 * c_t

        c_in = 1 / A ** 0.5
        c_skip = (b_t * sigma_data**2 + a_t * cov_xy) / A
        c_out = (a_t**2 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) +
                 sigma_data**2 * c**2 * c_t)**0.5 * c_in
        return c_skip, c_out, c_in

    elif pred_mode in ['ve_simple', 'vp_simple']:
        c_in = torch.ones_like(sigma)
        c_out = torch.ones_like(sigma)
        c_skip = torch.zeros_like(sigma)
        return c_skip, c_out, c_in

    raise ValueError(f"Unknown pred_mode: {pred_mode}")


def get_loss_weights(sigma, sigma_data, sigma_max, beta_d, beta_min, pred_mode):
    """Get loss weights based on bridge Karras weighting."""
    sigma_data_end = sigma_data
    cov_xy = 0.0
    c = 1

    if pred_mode == 've':
        A = (sigma**4 / sigma_max**4 * sigma_data_end**2 +
             (1 - sigma**2 / sigma_max**2)**2 * sigma_data**2 +
             2 * sigma**2 / sigma_max**2 * (1 - sigma**2 / sigma_max**2) * cov_xy +
             c**2 * sigma**2 * (1 - sigma**2 / sigma_max**2))
        weights = A / ((sigma / sigma_max)**4 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) +
                       sigma_data**2 * c**2 * sigma**2 * (1 - sigma**2 / sigma_max**2))

    elif pred_mode == 'vp':
        logsnr_t = vp_logsnr(sigma, beta_d, beta_min)
        logsnr_T = vp_logsnr(torch.tensor(1.0), beta_d, beta_min)
        logs_t = vp_logs(sigma, beta_d, beta_min)
        logs_T = vp_logs(torch.tensor(1.0), beta_d, beta_min)

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()

        A = a_t**2 * sigma_data_end**2 + b_t**2 * sigma_data**2 + 2 * a_t * b_t * cov_xy + c**2 * c_t
        weights = A / (a_t**2 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) +
                       sigma_data**2 * c**2 * c_t)

    elif pred_mode in ['vp_simple', 've_simple']:
        weights = torch.ones_like(sigma)

    else:
        raise ValueError(f"Unknown pred_mode: {pred_mode}")

    return weights


def vp_logsnr(t, beta_d, beta_min):
    """Compute log SNR for VP schedule."""
    t = torch.as_tensor(t)
    return -torch.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)


def vp_logs(t, beta_d, beta_min):
    """Compute log scale for VP schedule."""
    t = torch.as_tensor(t)
    return -0.25 * t ** 2 * beta_d - 0.5 * t * beta_min


def append_dims(x, target_dims):
    """Append dimensions to tensor."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    return x[(...,) + (None,) * dims_to_append]


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging directory
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir
    )

    # Initialize accelerator
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    logger.info("Creating model...")
    model = create_model(
        image_size=args.resolution,
        in_channels=3,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        unet_type=args.model_type,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
        condition_mode=args.condition_mode,
    )

    # Initialize scheduler
    logger.info("Creating scheduler...")
    scheduler = DDBMScheduler(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        beta_d=args.beta_d,
        beta_min=args.beta_min,
        pred_mode=args.pred_mode,
        num_train_timesteps=args.num_inference_steps,
    )

    # Initialize EMA if requested
    ema_model = None
    if args.use_ema:
        from diffusers.training_utils import EMAModel
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
            use_ema_warmup=True,
            model_cls=type(model),
        )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Load dataset
    logger.info("Loading dataset...")
    train_dataset, val_dataset = get_dataset(args)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    # Setup learning rate scheduler
    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(
            args.max_train_steps if args.max_train_steps
            else len(train_dataloader) * args.num_epochs
        ) * args.gradient_accumulation_steps,
    )

    # Prepare for distributed training
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers("ddbm-training", config=tracker_config)

    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.warning(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting from scratch.")
        else:
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resuming from checkpoint: {path}")

    # Training epochs
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    for epoch in range(first_epoch, args.num_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Preprocess batch
                x0, x_T = preprocess_batch(batch, accelerator.device)

                # Compute loss
                loss = compute_training_loss(
                    model=model,
                    scheduler=scheduler,
                    x0=x0,
                    x_T=x_T,
                    sigma_data=scheduler.config.sigma_data,
                    pred_mode=args.pred_mode,
                )

                # Backward pass
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update EMA
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                # Logging
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # Save checkpoint
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # Cleanup old checkpoints
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) > args.checkpoints_total_limit:
                                for checkpoint in checkpoints[:-args.checkpoints_total_limit]:
                                    checkpoint_path = os.path.join(args.output_dir, checkpoint)
                                    logger.info(f"Removing checkpoint: {checkpoint_path}")
                                    import shutil
                                    shutil.rmtree(checkpoint_path)

            if global_step >= args.max_train_steps:
                break

        # Save model at epoch end
        if accelerator.is_main_process and (epoch + 1) % args.save_model_epochs == 0:
            # Save model weights
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(unwrapped_model.state_dict(), save_path)
            logger.info(f"Saved model to {save_path}")

            # Save EMA model
            if args.use_ema:
                ema_save_path = os.path.join(args.output_dir, f"ema_model_epoch_{epoch+1}.pt")
                torch.save(ema_model.state_dict(), ema_save_path)
                logger.info(f"Saved EMA model to {ema_save_path}")

            # Save scheduler config
            scheduler.save_config(args.output_dir)

    # End training
    accelerator.end_training()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
