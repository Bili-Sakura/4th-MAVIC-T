#!/usr/bin/env python
"""Manual MAVIC-T metric evaluation for a trained Pix2Pix-Turbo checkpoint on a paired validation set.

Same interface as examples/cut/evaluate_metrics.py. Uses --model_path to a .pkl checkpoint.
Normalization follows official-docs/evaluation.md.

Usage::

    conda activate rsgen
    python -m examples.img2img_turbo.evaluate_metrics \
        --model_path ./outputs/turbo_sar2ir/checkpoints/model_final.pkl \
        --manifest datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2ir.txt \
        --task sar2ir \
        --batch_size 8
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
from examples.img2img_turbo.config import TaskConfig, sar2eo_config, rgb2ir_config, sar2ir_config, sar2rgb_config

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MAVIC-T metrics (Pix2Pix-Turbo) on a paired val set.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pkl checkpoint.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to paired val manifest (source\\ttarget per line).")
    parser.add_argument("--task", type=str, default="sar2eo", choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_fid", action="store_true", help="Disable FID (faster).")
    args = parser.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return args


def load_config(model_path: Path, task: str) -> TaskConfig:
    cfg = _TASK_CONFIG_MAP[task]()
    return cfg


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    manifest_path = resolve_manifest(args.manifest)
    cfg = load_config(model_path, args.task)
    resolution = args.resolution or getattr(cfg, "validation_resolution", None) or cfg.resolution

    from src.models.pix2pix_turbo import Pix2PixTurbo

    logger.info("Loading Pix2Pix-Turbo from %s", model_path)
    model = Pix2PixTurbo(
        pretrained_path=str(model_path),
        pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
    )
    model = model.to(args.device)
    model.set_eval()
    prompt_embeds = model.encode_prompt(cfg.prompt, torch.device(args.device))

    def inference_fn(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_norm = source * 2 - 1
        batch_embeds = prompt_embeds.expand(source.shape[0], -1, -1)
        output = model(source_norm, batch_embeds)
        return (output + 1) * 0.5

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
