#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Manual MAVIC-T metric evaluation for a trained DDIB pipeline on a paired validation set.

Same interface as examples/cut/evaluate_metrics.py. Expects a combined pipeline
directory (source_unet/, target_unet/, scheduler/). Normalization follows official-docs/evaluation.md.

Usage::

    conda activate rsgen
    python -m examples.ddib.evaluate_metrics \
        --checkpoint_dir ./ckpt/ddib/sar2eo/pipeline \
        --manifest datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2eo.txt \
        --task sar2eo \
        --batch_size 8 \
        --num_inference_steps 250
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from examples.eval_common import resolve_manifest, run_metric_evaluation
from examples.ddib.config import TaskConfig, sar2eo_config, rgb2ir_config, sar2ir_config, sar2rgb_config

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAVIC-T metrics (DDIB) on a paired val set.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to combined DDIB pipeline dir (source_unet/, target_unet/, scheduler/).")
    parser.add_argument("--manifest", type=str, required=True, help="Path to paired val manifest (source\\ttarget per line).")
    parser.add_argument("--task", type=str, default="sar2eo", choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=250)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_fid", action="store_true", help="Disable FID (faster).")
    args = parser.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return args


def load_config(checkpoint_dir: Path | None, task: str) -> TaskConfig:
    cfg = _TASK_CONFIG_MAP[task]()
    if checkpoint_dir and (checkpoint_dir / "config.yaml").is_file():
        import yaml
        with open(checkpoint_dir / "config.yaml") as f:
            saved = yaml.safe_load(f)
        if saved:
            for key in ("resolution", "source_channels", "target_channels", "validation_resolution"):
                if key in saved and saved[key] is not None:
                    setattr(cfg, key, saved[key])
    return cfg


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    manifest_path = resolve_manifest(args.manifest)
    cfg = load_config(checkpoint_dir, args.task)

    from src.pipelines.ddib import DDIBPipeline
    from src.models.unet.unet_ddib import DDIBUNet

    logger.info("Loading DDIB pipeline from %s", checkpoint_dir)
    pipeline = DDIBPipeline.from_pretrained(str(checkpoint_dir))
    pp = Path(args.checkpoint_dir)
    if (pp / "source_ema_unet").is_dir():
        pipeline.source_unet = DDIBUNet.from_pretrained(str(checkpoint_dir), subfolder="source_ema_unet")
        pipeline.target_unet = DDIBUNet.from_pretrained(str(checkpoint_dir), subfolder="target_ema_unet")

    pipeline = pipeline.to(args.device)
    pipeline.source_unet.eval()
    pipeline.target_unet.eval()

    num_steps = getattr(cfg, "num_inference_steps", None) or args.num_inference_steps

    def inference_fn(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_inp = source * 2 - 1
        result = pipeline(
            source_image=source_inp,
            num_inference_steps=num_steps,
            clip_denoised=getattr(cfg, "clip_denoised", False),
            eta=getattr(cfg, "eta", 0.0),
            output_type="pt",
        )
        return (result.images + 1) * 0.5

    resolution = args.resolution or getattr(cfg, "validation_resolution", None) or cfg.resolution

    run_metric_evaluation(
        manifest_path=manifest_path,
        resolution=resolution,
        source_channels=cfg.source_channels,
        target_channels=cfg.target_channels,
        device=args.device,
        batch_size=args.batch_size,
        no_fid=args.no_fid,
        inference_fn=inference_fn,
    )


if __name__ == "__main__":
    main()
