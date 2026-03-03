#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Manual metric evaluation for a trained StegoGAN checkpoint on a paired validation set.

Usage::

    python -m examples.stegogan.evaluate_metrics \
        --checkpoint_dir ckpt/stegogan/sar2eo/checkpoint-epoch-100 \
        --manifest datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2eo.txt \
        --task sar2eo \
        --batch_size 16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.ddbm.dataset_wrapper import PairedValDataset, resolve_paired_val_manifest
from examples.stegogan.config import TaskConfig, sar2eo_config, rgb2ir_config, sar2ir_config, sar2rgb_config
from src.pipelines.stegogan.pipeline_stegogan import StegoGANPipeline
from src.utils.metrics import MetricCalculator, task_score

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAVIC-T metrics on a paired val set.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--task", type=str, default="sar2eo", choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_fid", action="store_true")
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def load_config_from_checkpoint(checkpoint_dir: Path, task: str) -> TaskConfig:
    config_yaml = checkpoint_dir / "config.yaml"
    cfg = _TASK_CONFIG_MAP[task]()
    if config_yaml.is_file():
        import yaml
        with open(config_yaml) as f:
            saved = yaml.safe_load(f)
        if saved:
            for key in ("resolution", "source_channels", "target_channels", "validation_resolution"):
                if key in saved and saved[key] is not None:
                    setattr(cfg, key, saved[key])
        logger.info("Loaded config overrides from %s", config_yaml)
    return cfg


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    resolved = resolve_paired_val_manifest(args.manifest)
    if resolved is not None:
        manifest_path = resolved
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            candidate = _PROJECT_ROOT / args.manifest
            if candidate.is_file():
                manifest_path = candidate
            else:
                raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    cfg: TaskConfig = load_config_from_checkpoint(checkpoint_dir, args.task)
    resolution = args.resolution if args.resolution is not None else getattr(
        cfg, "validation_resolution", None
    ) or cfg.resolution

    val_ds = PairedValDataset(
        manifest_path=manifest_path,
        resolution=resolution,
        source_channels=cfg.source_channels,
        target_channels=cfg.target_channels,
        return_order="source_target",
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logger.info("Paired val set: %s (%d pairs), resolution=%d", manifest_path, len(val_ds), resolution)

    pipeline = StegoGANPipeline.from_pretrained(str(checkpoint_dir))
    pipeline = pipeline.to(args.device)
    pipeline.generator.eval()

    metric_calc = MetricCalculator(device=args.device, compute_fid=not args.no_fid, net_type="vgg")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            source, target = batch
            source = source.to(args.device)
            target = target.to(args.device)

            source_inp = source * 2 - 1
            result = pipeline(source_image=source_inp, output_type="pt")
            pred = (result.images + 1) * 0.5

            if cfg.target_channels == 1 and pred.shape[1] == 3:
                pred = pred.mean(dim=1, keepdim=True)
            elif pred.shape[1] != target.shape[1]:
                if pred.shape[1] == 3 and target.shape[1] == 1:
                    pred = pred.mean(dim=1, keepdim=True)
                elif pred.shape[1] == 1 and target.shape[1] == 3:
                    pred = pred.repeat(1, 3, 1, 1)

            pred = pred.clamp(0, 1)
            metric_calc.update(pred, target)

    results = metric_calc.compute()
    logger.info("Metrics: %s", results)
    if results.score is not None:
        logger.info("Task score = %.4f", results.score)

    print(
        f"lpips={results.lpips:.4f} l1={results.l1:.4f}"
        + (f" fid={results.fid:.2f}" if results.fid is not None else " fid=N/A")
        + (f" task_score={results.score:.4f}" if results.score is not None else "")
    )


if __name__ == "__main__":
    main()
