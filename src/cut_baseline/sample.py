#!/usr/bin/env python
"""Sample (inference) script for a trained CUT model on any MAVIC-T task.

Usage (pretrained directory — recommended)::

    python -m src.cut_baseline.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./ckpt/cut/sar2ir/checkpoint-epoch-400 \
        --split test \
        --output_dir ./samples/cut_sar2ir

Usage (legacy ``.pt`` file)::

    python -m src.cut_baseline.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./outputs/cut_sar2ir/netG_epoch_400.pt \
        --split test \
        --output_dir ./samples/cut_sar2ir

When a directory is provided the script loads the generator via
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

from src.cut_baseline.config import (  # noqa: E402
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from src.cut_baseline.dataset_wrapper import MavicTCUTDataset  # noqa: E402
from src.cut_baseline.models import CUTGenerator, create_generator  # noqa: E402
from src.cut_baseline.pipelines import CUTPipeline  # noqa: E402

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained CUT generator.")
    parser.add_argument("--task", type=str, required=True, choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to a diffusers-style checkpoint directory or a legacy .pt/.safetensors file.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_npz", action="store_true")
    return parser.parse_args()


def _load_pipeline(pretrained_path: str, cfg: TaskConfig, device: str) -> CUTPipeline:
    """Load the CUT pipeline from a pretrained directory or legacy file."""
    path = Path(pretrained_path)

    if path.is_dir():
        # ---- diffusers from_pretrained path ----
        logger.info("Loading generator from pretrained directory: %s", path)
        netG = CUTGenerator.from_pretrained(pretrained_path, subfolder="unet")
        pipeline = CUTPipeline(generator=netG)
    else:
        # ---- legacy single-file checkpoint ----
        logger.info("Loading generator from legacy checkpoint: %s", path)
        netG = create_generator(
            input_nc=cfg.model_channels,
            output_nc=cfg.model_channels,
            ngf=cfg.ngf,
            netG=cfg.netG,
            norm_type=cfg.normG,
            use_dropout=not cfg.no_dropout,
            no_antialias=cfg.no_antialias,
            no_antialias_up=cfg.no_antialias_up,
            init_type=cfg.init_type,
            init_gain=cfg.init_gain,
        )
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(str(path))
        else:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        netG.load_state_dict(ckpt)
        pipeline = CUTPipeline(generator=netG)

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
    dataset = MavicTCUTDataset(
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
            source = batch[0].to(args.device) * 2 - 1  # [0,1] → [-1,1]

            result = pipeline(source_image=source, output_type="pt")
            images = result.images

            # Convert from [-1, 1] to [0, 255] uint8
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
