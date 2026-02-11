#!/usr/bin/env python
"""Pre-compute mean and std for the domain dataset (positive/real images only).

Use these statistics instead of ImageNet for normalization when training
on domain-specific datasets.

Usage::

    # Compute for IR domain (default), save to JSON
    python -m examples.domain_classifier.compute_dataset_stats

    # Compute for EO domain
    python -m examples.domain_classifier.compute_dataset_stats --target_domain eo

    # Output to specific path
    python -m examples.domain_classifier.compute_dataset_stats --output_path ./stats/ir_domain_stats.json

    # Use with training (pass --dataset_stats_path to train_domain_classifier)
    python -m examples.domain_classifier.train_domain_classifier --dataset_stats_path ./stats/ir_domain_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# Ensure project root is importable when executed as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import DomainClassifierConfig  # noqa: E402
from .dataset_wrapper import (  # noqa: E402
    _adapt_channels,
    _to_float01,
    build_binary_records,
)
from .dataset_wrapper import BinaryImageRecord  # noqa: E402


def _load_image_array(
    image_path: str,
    *,
    resolution: int,
    num_channels: int,
) -> np.ndarray:
    """Load image as float [0,1] array, HWC."""
    with Image.open(image_path) as image:
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.BILINEAR)
        arr = np.asarray(image)

    arr = _to_float01(arr)
    arr = _adapt_channels(arr, num_channels=num_channels)
    return arr  # HWC


def compute_dataset_stats(
    records: list[BinaryImageRecord],
    *,
    resolution: int,
    num_channels: int,
    sample_limit: Optional[int] = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute per-channel mean and std over dataset images.

    Uses Welford's online algorithm for numerical stability.
    Images are loaded in [0,1] range with same preprocessing as training.
    """
    n = 0
    mean = np.zeros(num_channels, dtype=np.float64)
    M2 = np.zeros(num_channels, dtype=np.float64)

    iterator = records
    if sample_limit is not None:
        step = max(1, len(records) // sample_limit)
        iterator = records[::step][:sample_limit]

    for record in tqdm(iterator, desc="Computing stats", unit="img"):
        arr = _load_image_array(
            record.image_path,
            resolution=resolution,
            num_channels=num_channels,
        )
        # arr: HWC -> reshape to (H*W, C)
        pixels = arr.reshape(-1, num_channels)

        for p in pixels:
            n += 1
            delta = p - mean
            mean += delta / n
            delta2 = p - mean
            M2 += delta * delta2

    if n < 2:
        raise RuntimeError(f"Too few pixels for std (n={n})")

    variance = M2 / (n - 1)
    std = np.sqrt(variance)
    # Avoid division by zero in normalization
    std = np.maximum(std, 1e-6)

    return (
        tuple(float(x) for x in mean),
        tuple(float(x) for x in std),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute mean and std for domain dataset (positive images)"
    )
    parser.add_argument(
        "--target_domain",
        type=str,
        default="ir",
        help="Target domain (sar/eo/rgb/ir)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSON path (default: ./stats/<domain>_dataset_stats.json)",
    )
    parser.add_argument(
        "--refined_root",
        type=str,
        default=None,
        help="Override refined dataset root",
    )
    parser.add_argument(
        "--positive_tasks_csv",
        type=str,
        default=None,
        help="Override positive tasks (comma-separated)",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Limit number of images for faster stats (default: use all)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override resolution (default: domain native)",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=None,
        help="Override num_channels (default: domain native)",
    )
    args = parser.parse_args()

    cfg = DomainClassifierConfig(
        target_domain=args.target_domain,
        refined_root=args.refined_root,
        positive_tasks_csv=args.positive_tasks_csv,
        use_source_as_fake=True,  # Need at least one fake source for build_binary_records
    )
    resolution = cfg.resolved_resolution() if args.resolution is None else args.resolution
    num_channels = cfg.resolved_num_channels() if args.num_channels is None else args.num_channels

    records = build_binary_records(cfg)
    positive_records = [r for r in records if r.label == 1]
    if not positive_records:
        raise RuntimeError("No positive records found.")

    mean, std = compute_dataset_stats(
        positive_records,
        resolution=resolution,
        num_channels=num_channels,
        sample_limit=args.sample_limit,
    )

    output_path = args.output_path
    if output_path is None:
        output_path = f"./stats/{args.target_domain}_dataset_stats.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "mean": list(mean),
        "std": list(std),
        "num_channels": num_channels,
        "resolution": resolution,
        "target_domain": args.target_domain,
        "num_images": len(positive_records),
        "sample_limit_used": args.sample_limit,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Computed stats for {len(positive_records)} images:")
    print(f"  mean = {mean}")
    print(f"  std  = {std}")
    print(f"  saved to {output_path}")
    print(f"\nUse with training:")
    print(f"  --dataset_stats_path {output_path}")


if __name__ == "__main__":
    main()
