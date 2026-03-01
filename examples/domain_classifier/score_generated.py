#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Score generated images with a pre-trained domain real/fake classifier.

This script computes the model's soft probability that each image is "real"
for the target domain, without requiring any ground-truth references.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import ResNetForImageClassification

# Ensure project root is importable when executed as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import (  # noqa: E402
    default_num_channels_for_domain,
    default_resolution_for_domain,
    normalization_stats_for_channels,
    normalize_domain_name,
    target_domain_for_task,
)
from .dataset_wrapper import InferenceImageDataset  # noqa: E402


ALLOWED_IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".webp", ".bmp")


def _bool_arg(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _collect_image_paths(input_dir: Path, recursive: bool) -> list[str]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if recursive:
        files = [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        ]
    else:
        files = [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        ]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No image files found in {input_dir} (allowed: {ALLOWED_IMAGE_SUFFIXES})"
        )
    return [str(path) for path in files]


def _resolve_domain(task: Optional[str], domain: Optional[str]) -> Optional[str]:
    if domain:
        return normalize_domain_name(domain)
    if task:
        return target_domain_for_task(task)
    return None


def _resolve_model_resolution(
    model: ResNetForImageClassification,
    *,
    override_resolution: Optional[int],
    domain: Optional[str],
) -> int:
    if override_resolution is not None:
        return int(override_resolution)

    image_size = getattr(model.config, "image_size", None)
    if image_size is not None:
        if isinstance(image_size, (tuple, list)) and image_size:
            return int(image_size[-1])
        if isinstance(image_size, int):
            return image_size

    if domain is not None:
        return default_resolution_for_domain(domain)
    return 224


def _resolve_model_num_channels(
    model: ResNetForImageClassification,
    *,
    override_num_channels: Optional[int],
    domain: Optional[str],
) -> int:
    if override_num_channels is not None:
        return int(override_num_channels)

    num_channels = getattr(model.config, "num_channels", None)
    if num_channels is not None:
        return int(num_channels)

    if domain is not None:
        return default_num_channels_for_domain(domain)
    return 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score generated images with a domain real/fake classifier"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained classifier checkpoint directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing generated images")
    parser.add_argument("--task", type=str, default=None, help="Task name (e.g. sar2ir) used to infer target domain")
    parser.add_argument("--domain", type=str, default=None, help="Explicit target domain override (sar/eo/rgb/ir)")
    parser.add_argument("--resolution", type=int, default=None, help="Override inference resolution")
    parser.add_argument("--num_channels", type=int, default=None, help="Override model input channels")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--recursive", type=_bool_arg, default=True, help="Recursively scan input_dir")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--output_csv", type=str, default=None, help="CSV path for per-image scores")
    parser.add_argument("--output_json", type=str, default=None, help="JSON path for aggregate summary")
    parser.add_argument(
        "--dataset_stats_path",
        type=str,
        default=None,
        help="Path to pre-computed dataset stats JSON (overrides training_setup.json / ImageNet fallback)",
    )
    return parser.parse_args()


def _resolve_normalization_stats(
    model_path: str,
    num_channels: int,
    dataset_stats_path: Optional[str],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Resolve mean/std: dataset_stats_path > training_setup.json > ImageNet fallback."""
    if dataset_stats_path:
        from .config import load_dataset_stats  # noqa: E402
        return load_dataset_stats(dataset_stats_path)

    # Check for training_setup.json in run directory (parent of model_path when using best/final)
    setup_path = Path(model_path).resolve().parent / "training_setup.json"
    if setup_path.is_file():
        with setup_path.open() as f:
            data = json.load(f)
        resolved = data.get("resolved", {})
        mean = resolved.get("normalize_mean")
        std = resolved.get("normalize_std")
        if mean is not None and std is not None:
            return tuple(mean), tuple(std)

    return normalization_stats_for_channels(num_channels)


def main() -> None:
    args = parse_args()
    domain = _resolve_domain(task=args.task, domain=args.domain)

    model = ResNetForImageClassification.from_pretrained(args.model_path)
    resolution = _resolve_model_resolution(
        model,
        override_resolution=args.resolution,
        domain=domain,
    )
    num_channels = _resolve_model_num_channels(
        model,
        override_num_channels=args.num_channels,
        domain=domain,
    )
    normalize_mean, normalize_std = _resolve_normalization_stats(
        args.model_path,
        num_channels,
        args.dataset_stats_path,
    )

    image_paths = _collect_image_paths(Path(args.input_dir), recursive=args.recursive)
    dataset = InferenceImageDataset(
        image_paths,
        resolution=resolution,
        num_channels=num_channels,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)
    model.eval()

    rows: list[dict[str, object]] = []
    p_real_all: list[float] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scoring", unit="batch"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            logits = model(pixel_values=pixel_values).logits
            probs = torch.softmax(logits, dim=-1)
            p_fake = probs[:, 0].detach().cpu().numpy()
            p_real = probs[:, 1].detach().cpu().numpy()
            pred = torch.argmax(probs, dim=-1).detach().cpu().numpy()

            for path, fake_prob, real_prob, pred_label in zip(
                batch["path"], p_fake, p_real, pred
            ):
                rows.append(
                    {
                        "path": path,
                        "p_fake": float(fake_prob),
                        "p_real": float(real_prob),
                        "predicted_label": "real" if int(pred_label) == 1 else "fake",
                    }
                )
                p_real_all.append(float(real_prob))

    p_real_np = np.asarray(p_real_all, dtype=np.float64)
    summary = {
        "model_path": args.model_path,
        "input_dir": str(Path(args.input_dir).resolve()),
        "task": args.task,
        "domain": domain,
        "resolution": resolution,
        "num_channels": num_channels,
        "num_images": int(p_real_np.size),
        "realness_score_mean_p_real": float(np.mean(p_real_np)),
        "median_p_real": float(np.median(p_real_np)),
        "std_p_real": float(np.std(p_real_np)),
        "min_p_real": float(np.min(p_real_np)),
        "max_p_real": float(np.max(p_real_np)),
        "p_real_q05": float(np.quantile(p_real_np, 0.05)),
        "p_real_q95": float(np.quantile(p_real_np, 0.95)),
        "num_predicted_real": int(sum(row["predicted_label"] == "real" for row in rows)),
        "num_predicted_fake": int(sum(row["predicted_label"] == "fake" for row in rows)),
    }

    input_dir = Path(args.input_dir)
    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else input_dir / "realness_scores.csv"
    )
    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else input_dir / "realness_summary.json"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "p_fake", "p_real", "predicted_label"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with output_json.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

