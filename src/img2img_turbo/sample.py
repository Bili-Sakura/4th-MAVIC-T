#!/usr/bin/env python
"""Sample (inference) script for a trained Pix2Pix-Turbo model on any MAVIC-T task.

Usage::

    python -m src.img2img_turbo.sample \
        --task sar2ir \
        --model_path ./outputs/turbo_sar2ir/checkpoints/model_final.pkl \
        --split test \
        --output_dir ./samples/turbo_sar2ir

The script loads the model checkpoint, reads the evaluation inputs via
:class:`src.mavic_t_dataset.MavicTImageToImageDataset`, runs the Pix2Pix-Turbo
pipeline, and saves the generated images.
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

from src.img2img_turbo.config import (  # noqa: E402
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from src.img2img_turbo.dataset_wrapper import MavicTTurboDataset  # noqa: E402
from src.img2img_turbo.model import Pix2PixTurbo  # noqa: E402

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained Pix2Pix-Turbo model.")
    parser.add_argument("--task", type=str, required=True, choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pkl checkpoint.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 for faster inference.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    cfg: TaskConfig = _TASK_CONFIG_MAP[args.task]()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    logger.info("Loading model …")
    model = Pix2PixTurbo(
        pretrained_path=args.model_path,
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
    )
    model = model.to(args.device)
    model.set_eval()
    if args.use_fp16:
        model.half()

    # Encode prompt
    prompt_embeds = model.encode_prompt(cfg.prompt, torch.device(args.device))
    if args.use_fp16:
        prompt_embeds = prompt_embeds.half()

    # Load evaluation data
    dataset = MavicTTurboDataset(
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
            source = batch["conditioning_pixel_values"].to(args.device)
            if args.use_fp16:
                source = source.half()

            # Scale source from [0, 1] to [-1, 1]
            source_norm = source * 2 - 1

            # Expand prompt embeds for batch
            batch_embeds = prompt_embeds.expand(source.shape[0], -1, -1)

            output = model(source_norm, batch_embeds)
            # output is in [-1, 1]
            images_uint8 = ((output + 1) * 127.5).clamp(0, 255).to(torch.uint8)
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
