#!/usr/bin/env python
"""Ablation: inference steps 10-100 (interval 10) and 100-1000 (interval 100) on a single sample.

Runs DBIM rgb2ir inference for sample id 2 with varying num_inference_steps,
saves a grid of all results using diffusers make_image_grid to temp folder.

Usage:
    python scripts/DBIM_Pixel_Medium-0216/ablation_inference_steps.py
    SAMPLE_ID=5 python scripts/DBIM_Pixel_Medium-0216/ablation_inference_steps.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from diffusers.utils import make_image_grid
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.unet_dbim import DBIMUNet
from src.pipelines.dbim import DBIMPipeline
from src.schedulers import DBIMScheduler

from examples.dbim.config import rgb2ir_config
from examples.dbim.dataset_wrapper import MavicTDBIMDataset

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CKPT_PATH = os.environ.get(
    "CKPT_PATH",
    "/data/projects/models/hf_models/BiliSakura/4th-MAVIC-T-ckpt-0216/dbim/rgb2ir/checkpoint-100000",
)
OUTPUT_DIR = Path("/data/projects/4th-MAVIC-T/temp")
SAMPLE_ID = int(os.environ.get("SAMPLE_ID", "2"))
# Two ranges: 10-100 step 10, then 100-1000 step 100 (100 deduplicated)
STEP_RANGES = [(10, 100, 10), (100, 1000, 100)]
SEED = 42


def _load_pipeline(pretrained_path: str, device: str) -> DBIMPipeline:
    path = Path(pretrained_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")

    logger.info("Loading pipeline from %s", path)
    unet_subfolder = "ema_unet" if (path / "ema_unet").is_dir() else "unet"
    unet = DBIMUNet.from_pretrained(pretrained_path, subfolder=unet_subfolder)

    scheduler_config = path / "scheduler" / "scheduler_config.json"
    if scheduler_config.exists():
        scheduler = DBIMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
    else:
        config_yaml = path / "config.yaml"
        if config_yaml.exists():
            with config_yaml.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            scheduler = DBIMScheduler(
                sigma_min=cfg.get("sigma_min", 0.002),
                sigma_max=cfg.get("sigma_max", 1.0),
                sigma_data=cfg.get("sigma_data", 0.5),
                beta_d=cfg.get("beta_d", 2.0),
                beta_min=cfg.get("beta_min", 0.1),
                pred_mode=cfg.get("pred_mode", "vp"),
                sampler=cfg.get("sampler", "dbim"),
                eta=cfg.get("eta", 1.0),
                order=cfg.get("order", 2),
                lower_order_final=cfg.get("lower_order_final", True),
            )
        else:
            scheduler = DBIMScheduler()

    return DBIMPipeline(unet=unet, scheduler=scheduler).to(device)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = rgb2ir_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = _load_pipeline(CKPT_PATH, device)

    dataset = MavicTDBIMDataset(
        task="rgb2ir",
        split="test",
        resolution=cfg.resolution,
        source_channels=cfg.source_channels,
        target_channels=cfg.target_channels,
        with_target=False,
    )

    if SAMPLE_ID < 0 or SAMPLE_ID >= len(dataset):
        raise ValueError(f"Sample id {SAMPLE_ID} out of range [0, {len(dataset) - 1}]")

    target, source = dataset[SAMPLE_ID]
    source_batch = source.unsqueeze(0).to(device) * 2 - 1  # [0,1] -> [-1,1]
    out_name = dataset.get_output_name(SAMPLE_ID)

    step_values = sorted(
        {s for start, end, step in STEP_RANGES for s in range(start, end + 1, step)}
    )
    logger.info(
        "Ablation: sample_id=%d (%s), steps=%s -> %d runs",
        SAMPLE_ID,
        out_name,
        "10-100 step 10, 100-1000 step 100",
        len(step_values),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pil_images = []
    GRID_BATCH = 10

    with torch.no_grad():
        for i, n_steps in enumerate(tqdm(step_values, desc="Inference steps", unit="run")):
            result = pipeline(
                source_image=source_batch,
                num_inference_steps=n_steps,
                sampler="dbim",
                guidance=1.0,
                churn_step_ratio=0.33,
                eta=1.0,
                order=2,
                lower_order_final=True,
                clip_denoised=False,
                output_type="pt",
            )
            img = result.images[0]
            img_uint8 = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            img_uint8 = img_uint8.permute(1, 2, 0).cpu().numpy()
            if img_uint8.shape[2] == 1:
                img_uint8 = img_uint8.squeeze(2)
            pil_images.append(Image.fromarray(img_uint8))

            # Save grid every 10 results
            if (i + 1) % GRID_BATCH == 0:
                batch_images = pil_images[-GRID_BATCH:]
                step_lo = step_values[i - GRID_BATCH + 1]
                step_hi = step_values[i]
                batch_grid = make_image_grid(batch_images, rows=2, cols=5)
                batch_path = OUTPUT_DIR / f"ablation_steps_sample{SAMPLE_ID}_{step_lo}-{step_hi}.png"
                batch_grid.save(batch_path)
                logger.info("Saved batch grid %d-%d to %s", step_lo, step_hi, batch_path)

    n = len(pil_images)
    rows = int(n**0.5) if n > 0 else 1
    cols = (n + rows - 1) // rows if rows > 0 else n

    grid = make_image_grid(pil_images, rows=rows, cols=cols)
    out_path = OUTPUT_DIR / f"ablation_steps_sample{SAMPLE_ID}_10-1000_full.png"
    grid.save(out_path)

    logger.info("Saved full grid (%d images, %dx%d) to %s", n, rows, cols, out_path)


if __name__ == "__main__":
    main()
