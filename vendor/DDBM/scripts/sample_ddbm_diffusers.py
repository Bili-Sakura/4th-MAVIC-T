#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The DDBM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Sampling script for DDBM models using the diffusers-compatible pipeline.

This script generates images from a trained DDBM model using the DDBMPipeline.

Example usage:
    # Basic sampling
    python scripts/sample_ddbm_diffusers.py \
        --model_path ./outputs/ddbm-e2h/model.pt \
        --data_dir /path/to/data \
        --output_dir ./samples \
        --num_inference_steps 40

    # With custom parameters
    python scripts/sample_ddbm_diffusers.py \
        --model_path ./outputs/ddbm-e2h/model.pt \
        --data_dir /path/to/data \
        --output_dir ./samples \
        --guidance 1.0 \
        --churn_step_ratio 0.33 \
        --batch_size 16
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from ddbm import DDBMScheduler, DDBMPipeline
from ddbm.utils.script_util import create_model, model_defaults


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample from a trained DDBM model."
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--scheduler_config",
        type=str,
        default=None,
        help="Path to scheduler config directory. If not provided, uses defaults.",
    )

    # Model architecture (should match training)
    parser.add_argument(
        "--model_type",
        type=str,
        default="adm",
        choices=["adm", "edm"],
        help="UNet architecture type.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Image resolution.",
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
        "--condition_mode",
        type=str,
        default="concat",
        help="How to condition on source image.",
    )

    # Dataset arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="edges2handbags",
        choices=["edges2handbags", "diode"],
        help="Dataset to sample from.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use.",
    )

    # Sampling arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./samples",
        help="Directory to save generated samples.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate. If not set, uses entire dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for sampling.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of diffusion steps for sampling.",
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
        default=1.0,
        help="Maximum sigma value (use 1.0 for VP, 80.0 for VE).",
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

    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run sampling on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Save samples as NPZ file for FID evaluation.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=True,
        help="Save individual sample images.",
    )

    return parser.parse_args()


def get_dataset(args):
    """Load the dataset based on configuration."""
    if args.dataset_name == "edges2handbags":
        from datasets.aligned_dataset import EdgesDataset
        dataset = EdgesDataset(
            dataroot=args.data_dir,
            train=(args.split == "train"),
            img_size=args.image_size,
            random_crop=False,
            random_flip=False,
        )
    elif args.dataset_name == "diode":
        from datasets.aligned_dataset import DIODE
        dataset = DIODE(
            dataroot=args.data_dir,
            train=(args.split == "train"),
            img_size=args.image_size,
            random_crop=False,
            random_flip=False,
            disable_cache=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    return dataset


def main():
    """Main sampling function."""
    args = parse_args()

    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {args.device}")
    logger.info(f"Output directory: {output_dir}")

    # Load model
    logger.info("Loading model...")
    model = create_model(
        image_size=args.image_size,
        in_channels=3,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        unet_type=args.model_type,
        attention_resolutions=args.attention_resolutions,
        condition_mode=args.condition_mode,
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model = model.to(args.device)
    model.eval()

    # Create scheduler
    if args.scheduler_config:
        logger.info(f"Loading scheduler from {args.scheduler_config}")
        scheduler = DDBMScheduler.from_config(args.scheduler_config)
    else:
        logger.info("Creating scheduler with default config")
        scheduler = DDBMScheduler(
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            beta_d=args.beta_d,
            beta_min=args.beta_min,
            pred_mode=args.pred_mode,
        )

    # Create pipeline
    pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = get_dataset(args)
    
    num_samples = args.num_samples if args.num_samples else len(dataset)
    num_samples = min(num_samples, len(dataset))
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    logger.info(f"Generating {num_samples} samples...")
    logger.info(f"Sampling parameters:")
    logger.info(f"  - Inference steps: {args.num_inference_steps}")
    logger.info(f"  - Guidance: {args.guidance}")
    logger.info(f"  - Churn step ratio: {args.churn_step_ratio}")

    all_samples = []
    total_nfe = 0
    sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Sampling")):
            if sample_idx >= num_samples:
                break

            # Get source images (x_T in bridge terminology)
            # batch[0] is target, batch[1] is source/condition
            source_images = batch[1].to(args.device) * 2 - 1  # Scale to [-1, 1]
            current_batch_size = source_images.shape[0]

            # Run pipeline
            result = pipeline(
                source_image=source_images,
                num_inference_steps=args.num_inference_steps,
                guidance=args.guidance,
                churn_step_ratio=args.churn_step_ratio,
                output_type="pt",
            )

            samples = result.images  # Shape: (B, C, H, W), range [-1, 1]
            total_nfe += result.nfe

            # Convert to uint8 for saving
            samples_uint8 = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            samples_uint8 = samples_uint8.permute(0, 2, 3, 1).cpu().numpy()

            # Save individual images
            if args.save_images:
                for i, sample in enumerate(samples_uint8):
                    if sample_idx + i >= num_samples:
                        break
                    img = Image.fromarray(sample)
                    img.save(output_dir / f"sample_{sample_idx + i:05d}.png")

            all_samples.append(samples_uint8)
            sample_idx += current_batch_size

    # Concatenate all samples
    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]

    # Save as NPZ for FID evaluation
    if args.save_npz:
        npz_path = output_dir / f"samples_{num_samples}_nfe{total_nfe}.npz"
        np.savez(npz_path, arr_0=all_samples)
        logger.info(f"Saved NPZ to {npz_path}")

    # Save a grid of samples
    if len(all_samples) >= 16:
        from torchvision.utils import save_image
        grid_samples = torch.from_numpy(all_samples[:16]).permute(0, 3, 1, 2).float() / 255.0
        save_image(grid_samples, output_dir / "sample_grid.png", nrow=4, normalize=False)
        logger.info(f"Saved sample grid to {output_dir / 'sample_grid.png'}")

    logger.info(f"Sampling complete!")
    logger.info(f"  - Total samples: {len(all_samples)}")
    logger.info(f"  - Total NFE: {total_nfe}")
    logger.info(f"  - Average NFE per sample: {total_nfe / len(all_samples):.1f}")


if __name__ == "__main__":
    main()
