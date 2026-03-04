#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Run all four checkpoints on paired_val manifests and save input, generation, gt.

Output structure: /data/projects/4th-MAVIC-T/temp/{task}/input/, generation/, gt/

Checkpoints used:
  - rgb2ir: DBIM 4th-MAVIC-T-ckpt-0216/dbim/rgb2ir/checkpoint-100000 (5 steps)
  - sar2eo: DBIM 4th-MAVIC-T-ckpt-0216/dbim/sar2eo/checkpoint-100000 (500 steps)
  - sar2ir: CUT Huge 4th-MAVIC-T-ckpt-0226/cut/sar2ir/huge/checkpoint-20000
  - sar2rgb: DBIM 4th-MAVIC-T-ckpt-0216/dbim/sar2rgb/checkpoint-100000 (1000 steps)

Usage:
  conda activate rsgen
  cd /data/projects/4th-MAVIC-T
  python scripts/run_paired_val_inference.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.ddbm.dataset_wrapper import PairedValDataset, resolve_paired_val_manifest
from examples.dbim.config import rgb2ir_config, sar2eo_config, sar2ir_config, sar2rgb_config

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_ROOT = Path("/data/projects/4th-MAVIC-T/temp")
CKPT_BASE = Path("/data/projects/4th-MAVIC-T/models/BiliSakura")

# Task config: (model_type, ckpt_path, resolution, num_steps for DBIM, manifest_key)
TASK_SPECS = {
    "rgb2ir": {
        "model_type": "dbim",
        "ckpt": CKPT_BASE / "4th-MAVIC-T-ckpt-0216/dbim/rgb2ir/checkpoint-100000",
        "resolution": 1024,
        "num_steps": 5,
        "source_channels": 3,
        "target_channels": 1,
        "model_channels": 3,
    },
    "sar2eo": {
        "model_type": "dbim",
        "ckpt": CKPT_BASE / "4th-MAVIC-T-ckpt-0216/dbim/sar2eo/checkpoint-100000",
        "resolution": 256,
        "num_steps": 500,
        "source_channels": 1,
        "target_channels": 1,
        "model_channels": 1,
    },
    "sar2ir": {
        "model_type": "cut",
        "ckpt": CKPT_BASE / "4th-MAVIC-T-ckpt-0226/cut/sar2ir/huge/checkpoint-20000",
        "resolution": 1024,
        "num_steps": 1,  # CUT is single-step
        "source_channels": 1,
        "target_channels": 1,
        "model_channels": 1,
    },
    "sar2rgb": {
        "model_type": "dbim",
        "ckpt": CKPT_BASE / "4th-MAVIC-T-ckpt-0216/dbim/sar2rgb/checkpoint-100000",
        "resolution": 1024,
        "num_steps": 1000,
        "source_channels": 3,  # use model_channels so 1-ch SAR is expanded to 3 (matches DBIM training)
        "target_channels": 3,
        "model_channels": 3,
    },
}


def _tensor_to_pil(tensor: torch.Tensor, channels: int) -> Image.Image:
    """Convert [0,1] tensor (C,H,W) to PIL Image."""
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if channels == 1:
        arr = arr.squeeze(2)
    return Image.fromarray(arr)


def run_dbim_inference(
    manifest_path: Path,
    ckpt_path: Path,
    resolution: int,
    num_steps: int,
    source_channels: int,
    target_channels: int,
    model_channels: int,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 2,
) -> None:
    """Run DBIM pipeline on paired val manifest."""
    from src.models import DBIMUNet
    from src.pipelines.dbim import DBIMPipeline
    from src.schedulers import DBIMScheduler
    import yaml

    unet_subfolder = "ema_unet" if (ckpt_path / "ema_unet").is_dir() else "unet"
    unet = DBIMUNet.from_pretrained(str(ckpt_path), subfolder=unet_subfolder, torch_dtype=torch.bfloat16)
    scheduler_dir = ckpt_path / "scheduler"
    if (scheduler_dir / "scheduler_config.json").exists():
        scheduler = DBIMScheduler.from_pretrained(str(ckpt_path), subfolder="scheduler")
    else:
        config_yaml = ckpt_path / "config.yaml"
        cfg = {}
        if config_yaml.exists():
            with config_yaml.open() as f:
                cfg = yaml.safe_load(f) or {}
        scheduler = DBIMScheduler(
            sigma_min=cfg.get("sigma_min", 0.002),
            sigma_max=cfg.get("sigma_max", 1.0),
            sigma_data=cfg.get("sigma_data", 0.5),
            beta_d=cfg.get("beta_d", 2.0),
            beta_min=cfg.get("beta_min", 0.1),
            pred_mode=cfg.get("pred_mode", "vp"),
        )
    pipeline = DBIMPipeline(unet=unet, scheduler=scheduler).to(device, dtype=torch.bfloat16)

    dataset = PairedValDataset(
        manifest_path=manifest_path,
        resolution=resolution,
        source_channels=source_channels,
        target_channels=target_channels,
        return_order="source_target",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    (output_dir / "input").mkdir(parents=True, exist_ok=True)
    (output_dir / "generation").mkdir(parents=True, exist_ok=True)
    (output_dir / "gt").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"DBIM {manifest_path.stem}", leave=False)):
            source, target = batch
            batch_start = batch_idx * loader.batch_size
            source_norm = source.to(device) * 2 - 1  # [0,1] -> [-1,1]
            out = pipeline(
                source_image=source_norm,
                num_inference_steps=num_steps,
                sampler="dbim",
                guidance=1.0,
                cfg_scale=1.0,
                churn_step_ratio=0.33,
                eta=1.0,
                output_type="pt",
            )
            images = out.images  # (N,C,H,W) in [-1,1], possibly bfloat16
            images_01 = (images.float() + 1) / 2  # convert to float32 for numpy

            for i in range(images.shape[0]):
                global_idx = batch_start + i
                if global_idx >= len(dataset._pairs):
                    break
                stem = Path(dataset._pairs[global_idx][0]).stem
                # Save input
                _tensor_to_pil(source[i], source_channels).save(output_dir / "input" / f"{stem}.png")
                # Save generation
                ch = target_channels
                arr = (images_01[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                if ch == 1:
                    arr = arr[:, :, 0] if arr.ndim == 3 else arr
                Image.fromarray(arr).save(output_dir / "generation" / f"{stem}.png")
                # Save gt
                _tensor_to_pil(target[i], target_channels).save(output_dir / "gt" / f"{stem}.png")


def run_cut_inference(
    manifest_path: Path,
    ckpt_path: Path,
    resolution: int,
    source_channels: int,
    target_channels: int,
    model_channels: int,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 4,
) -> None:
    """Run CUT pipeline on paired val manifest."""
    from src.pipelines.cut import CUTPipeline

    pipeline = CUTPipeline.from_pretrained(str(ckpt_path), torch_dtype=torch.bfloat16).to(device)

    dataset = PairedValDataset(
        manifest_path=manifest_path,
        resolution=resolution,
        source_channels=source_channels,
        target_channels=target_channels,
        return_order="source_target",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    (output_dir / "input").mkdir(parents=True, exist_ok=True)
    (output_dir / "generation").mkdir(parents=True, exist_ok=True)
    (output_dir / "gt").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"CUT {manifest_path.stem}", leave=False)):
            source, target = batch
            batch_start = batch_idx * loader.batch_size
            source_norm = source.to(device) * 2 - 1
            out = pipeline(source_image=source_norm, output_type="pt")
            images = out.images  # (N,C,H,W) in [-1,1], possibly bfloat16
            images_01 = (images.float() + 1) / 2  # convert to float32 for numpy

            for i in range(images.shape[0]):
                global_idx = batch_start + i
                if global_idx >= len(dataset._pairs):
                    break
                stem = Path(dataset._pairs[global_idx][0]).stem
                _tensor_to_pil(source[i], source_channels).save(output_dir / "input" / f"{stem}.png")
                ch = target_channels
                arr = (images_01[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                if ch == 1:
                    arr = arr[:, :, 0] if arr.ndim == 3 else arr
                Image.fromarray(arr).save(output_dir / "generation" / f"{stem}.png")
                _tensor_to_pil(target[i], target_channels).save(output_dir / "gt" / f"{stem}.png")


def main():
    parser = argparse.ArgumentParser(description="Run checkpoints on paired_val manifests.")
    parser.add_argument("--tasks", type=str, nargs="+", default=["rgb2ir", "sar2eo", "sar2ir", "sar2rgb"])
    parser.add_argument("--output_root", type=str, default=str(TEMP_ROOT))
    parser.add_argument("--manifest_dir", type=str, default="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=2, help="DBIM batch size (CUT uses 4)")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifest_dir = Path(args.manifest_dir)
    if not manifest_dir.is_absolute():
        manifest_dir = _PROJECT_ROOT / manifest_dir

    for task in args.tasks:
        if task not in TASK_SPECS:
            logger.warning("Unknown task %s, skipping", task)
            continue
        spec = TASK_SPECS[task]
        manifest_path = manifest_dir / f"paired_val_{task}.txt"
        manifest_resolved = resolve_paired_val_manifest(str(manifest_path))
        if manifest_resolved is None:
            manifest_resolved = manifest_path
        if not manifest_resolved.exists():
            logger.warning("Manifest not found: %s, skipping %s", manifest_resolved, task)
            continue
        if not spec["ckpt"].is_dir():
            logger.warning("Checkpoint not found: %s, skipping %s", spec["ckpt"], task)
            continue

        out_dir = output_root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Running %s -> %s (ckpt=%s)", task, out_dir, spec["ckpt"])

        if spec["model_type"] == "dbim":
            run_dbim_inference(
                manifest_path=manifest_resolved,
                ckpt_path=spec["ckpt"],
                resolution=spec["resolution"],
                num_steps=spec["num_steps"],
                source_channels=spec["source_channels"],
                target_channels=spec["target_channels"],
                model_channels=spec["model_channels"],
                output_dir=out_dir,
                device=args.device,
                batch_size=args.batch_size,
            )
        else:
            run_cut_inference(
                manifest_path=manifest_resolved,
                ckpt_path=spec["ckpt"],
                resolution=spec["resolution"],
                source_channels=spec["source_channels"],
                target_channels=spec["target_channels"],
                model_channels=spec["model_channels"],
                output_dir=out_dir,
                device=args.device,
                batch_size=4,
            )
        logger.info("Done %s: %s/input, generation, gt", task, out_dir)

    logger.info("All tasks completed. Output: %s", output_root)


if __name__ == "__main__":
    main()
