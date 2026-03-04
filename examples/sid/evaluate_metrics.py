#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Manual MAVIC-T metric evaluation for a trained SID checkpoint on a paired validation set.

Same interface as examples/cut/evaluate_metrics.py. Normalization follows
official-docs/evaluation.md.
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
from examples.sid.config import TaskConfig, rgb2ir_config, sar2eo_config, sar2ir_config, sar2rgb_config

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAVIC-T metrics (SID) on a paired val set.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to diffusers-style checkpoint.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to paired val manifest (source\\ttarget per line).")
    parser.add_argument("--task", type=str, default="sar2eo", choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help="Classifier-Free Guidance scale. Defaults to checkpoint config or 1.0.",
    )
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_fid", action="store_true", help="Disable FID (faster).")
    args = parser.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return args


def load_config(checkpoint_dir: Path, task: str) -> TaskConfig:
    cfg = _TASK_CONFIG_MAP[task]()
    config_yaml = checkpoint_dir / "config.yaml"
    if config_yaml.is_file():
        import yaml

        with open(config_yaml) as f:
            saved = yaml.safe_load(f)
        if saved:
            for key in (
                "resolution",
                "source_channels",
                "target_channels",
                "validation_resolution",
                "num_inference_steps",
                "cfg_scale",
            ):
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
    resolution = args.resolution or getattr(cfg, "validation_resolution", None) or cfg.resolution

    from src.models import SiDUNet
    from src.pipelines.sid import SIDPipeline
    from src.schedulers import SiDScheduler

    logger.info("Loading SID pipeline from %s", checkpoint_dir)
    unet_subfolder = "ema_unet" if (checkpoint_dir / "ema_unet").is_dir() else "unet"
    unet = SiDUNet.from_pretrained(str(checkpoint_dir), subfolder=unet_subfolder)
    scheduler = SiDScheduler.from_pretrained(str(checkpoint_dir), subfolder="scheduler")
    pipeline = SIDPipeline(unet=unet, scheduler=scheduler).to(args.device)
    pipeline.unet.eval()

    num_steps = getattr(cfg, "num_inference_steps", None) or args.num_inference_steps
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else getattr(cfg, "cfg_scale", 1.0)

    def inference_fn(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_inp = source * 2 - 1
        result = pipeline(
            source_image=source_inp,
            num_inference_steps=num_steps,
            cfg_scale=cfg_scale,
            output_type="pt",
        )
        return (result.images + 1) * 0.5

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
