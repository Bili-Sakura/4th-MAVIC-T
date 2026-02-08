#!/usr/bin/env python
"""Sample (inference) script for trained DDIB models on any MAVIC-T task.

DDIB translates images by:
  1. Encoding source images to a shared latent via DDIM reverse sampling
     with the source-domain diffusion model.
  2. Decoding the latent to the target domain via DDIM forward sampling
     with the target-domain diffusion model.

Usage::

    python -m src.ddib_baseline.sample \
        --task sar2ir \
        --source_model_path ./ckpt/ddib_source/sar2ir/checkpoint-epoch-100/unet/diffusion_pytorch_model.safetensors \
        --target_model_path ./ckpt/ddib_target/sar2ir/checkpoint-epoch-100/unet/diffusion_pytorch_model.safetensors \
        --split test \
        --output_dir ./samples/ddib_sar2ir
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ddib_baseline.schedulers import DDIBScheduler  # noqa: E402
from src.ddib_baseline.pipelines import DDIBPipeline  # noqa: E402

from src.ddib_baseline.config import (  # noqa: E402
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from src.ddib_baseline.models import create_model  # noqa: E402

# Reuse the DDBM dataset wrapper for loading paired data (source side only)
from src.ddbm_baseline.dataset_wrapper import MavicTDDBMDataset  # noqa: E402

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Translate images using trained DDIB models.")
    parser.add_argument("--task", type=str, required=True, choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument("--source_model_path", type=str, required=True, help="Path to source-domain model .pt/.safetensors checkpoint.")
    parser.add_argument("--target_model_path", type=str, required=True, help="Path to target-domain model .pt/.safetensors checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=250)
    parser.add_argument("--clip_denoised", type=bool, default=True)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_npz", action="store_true")
    return parser.parse_args()


def _load_state_dict(path: str):
    """Load a state dict from a .pt or .safetensors file."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    cfg: TaskConfig = _TASK_CONFIG_MAP[args.task]()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build source model
    logger.info("Loading source model …")
    source_model = create_model(
        image_size=cfg.resolution,
        in_channels=cfg.source_channels,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=0.0,
        learn_sigma=cfg.learn_sigma,
        channel_mult=cfg.channel_mult,
    )
    source_model.load_state_dict(_load_state_dict(args.source_model_path))
    source_model = source_model.to(args.device).eval()

    # Build target model
    logger.info("Loading target model …")
    target_model = create_model(
        image_size=cfg.resolution,
        in_channels=cfg.target_channels,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=0.0,
        learn_sigma=cfg.learn_sigma,
        channel_mult=cfg.channel_mult,
    )
    target_model.load_state_dict(_load_state_dict(args.target_model_path))
    target_model = target_model.to(args.device).eval()

    # Build scheduler + pipeline
    scheduler = DDIBScheduler(
        num_train_timesteps=cfg.diffusion_steps,
        noise_schedule=cfg.noise_schedule,
        learn_sigma=cfg.learn_sigma,
        predict_xstart=cfg.predict_xstart,
        rescale_timesteps=cfg.rescale_timesteps,
    )
    pipeline = DDIBPipeline(
        source_unet=source_model,
        target_unet=target_model,
        scheduler=scheduler,
    )

    # Load evaluation data (source side only)
    dataset = MavicTDDBMDataset(
        task=args.task,
        split=args.split,
        resolution=cfg.resolution,
        model_channels=cfg.source_channels,
        with_target=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info(f"Generating samples for {args.task} ({args.split}), {len(dataset)} inputs …")
    all_samples = []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="DDIB Translation"):
            # batch = (zeros_target, source)
            source = batch[1].to(args.device) * 2 - 1  # [0,1] → [-1,1]

            result = pipeline(
                source_image=source,
                num_inference_steps=args.num_inference_steps,
                clip_denoised=args.clip_denoised,
                eta=args.eta,
                output_type="pt",
            )
            images = result.images  # (B, C, H, W) in [-1, 1]
            images_uint8 = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            images_uint8 = images_uint8.permute(0, 2, 3, 1).cpu().numpy()

            for img_arr in images_uint8:
                if img_arr.shape[2] == 1:
                    img_arr = img_arr.squeeze(2)
                img = Image.fromarray(img_arr)
                img.save(output_dir / f"sample_{sample_idx:05d}.png")
                sample_idx += 1

            all_samples.append(images_uint8)

    all_samples = np.concatenate(all_samples, axis=0)

    if args.save_npz:
        np.savez(output_dir / f"samples_{len(all_samples)}.npz", arr_0=all_samples)
        logger.info(f"Saved NPZ with {len(all_samples)} samples.")

    logger.info(f"Sampling complete – {sample_idx} images saved to {output_dir}")


if __name__ == "__main__":
    main()
