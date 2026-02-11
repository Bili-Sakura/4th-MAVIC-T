#!/usr/bin/env python
"""Sample (inference) script for a trained I2SB model on any MAVIC-T task.

Usage (pretrained directory — recommended)::

    python -m examples.i2sb.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./ckpt/i2sb/sar2ir/checkpoint-epoch-100 \
        --split test \
        --output_dir ./samples/sar2ir \
        --batch_size 32

Use ``--batch_size`` to control inference batch size (default 32). EMA UNet is
loaded by default when ``ema_unet/`` exists under the checkpoint.

Usage (legacy ``.pt`` file)::

    python -m examples.i2sb.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./outputs/i2sb_sar2ir/model_epoch_100.pt \
        --split test \
        --output_dir ./samples/sar2ir

When a directory is provided the script loads the UNet and scheduler via
``from_pretrained`` following the HuggingFace *diffusers* convention.
Legacy ``.pt`` / ``.safetensors`` single-file checkpoints are still
supported for backward compatibility.
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

from src.schedulers import I2SBScheduler  # noqa: E402
from src.pipelines.i2sb import I2SBPipeline  # noqa: E402

from .config import (  # noqa: E402
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from .dataset_wrapper import MavicTI2SBDataset  # noqa: E402
from src.models.unet_i2sb import I2SBUNet, create_model  # noqa: E402

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained I2SB model.")
    parser.add_argument("--task", type=str, required=True, choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to a diffusers-style checkpoint directory or a legacy .pt/.safetensors file.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--ot_ode", action="store_true")
    parser.add_argument("--clip_denoise", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_npz", action="store_true")
    return parser.parse_args()


def _load_pipeline(pretrained_path: str, cfg: TaskConfig, device: str) -> I2SBPipeline:
    """Load the I2SB pipeline from a pretrained directory or legacy file."""
    path = Path(pretrained_path)

    if path.is_dir():
        # ---- diffusers from_pretrained path ----
        logger.info("Loading pipeline from pretrained directory: %s", path)
        pipeline = I2SBPipeline.from_pretrained(pretrained_path)
        ema_unet_dir = path / "ema_unet"
        if ema_unet_dir.is_dir():
            logger.info("Loading EMA UNet from %s", ema_unet_dir)
            pipeline.unet = I2SBUNet.from_pretrained(pretrained_path, subfolder="ema_unet")
    else:
        # ---- legacy single-file checkpoint ----
        logger.info("Loading model from legacy checkpoint: %s", path)
        model = create_model(
            image_size=cfg.resolution,
            in_channels=cfg.model_channels,
            num_channels=cfg.num_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=cfg.attention_resolutions,
            dropout=0.0,
            condition_mode=cfg.condition_mode,
            channel_mult=cfg.channel_mult,
            unet_type=getattr(cfg, "unet_type", "adm"),
        )
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(str(path))
        else:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)

        scheduler = I2SBScheduler(
            interval=cfg.interval,
            beta_max=cfg.beta_max,
            t0=cfg.t0,
            T=cfg.T,
        )
        pipeline = I2SBPipeline(unet=model, scheduler=scheduler)

    pipeline = pipeline.to(device)
    return pipeline


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    cfg: TaskConfig = _TASK_CONFIG_MAP[args.task]()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _load_pipeline(args.pretrained_model_name_or_path, cfg, args.device)

    # Load evaluation data
    dataset = MavicTI2SBDataset(
        task=args.task,
        split=args.split,
        resolution=cfg.resolution,
        model_channels=cfg.model_channels,
        with_target=False,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info(f"Generating samples for {args.task} ({args.split}), {len(dataset)} inputs …")
    all_samples = []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Sampling"):
            # batch = (zeros_target, source)
            source = batch[1].to(args.device) * 2 - 1  # [0,1] → [-1,1]

            result = pipeline(
                source_image=source,
                nfe=args.nfe,
                ot_ode=args.ot_ode,
                clip_denoise=args.clip_denoise,
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
