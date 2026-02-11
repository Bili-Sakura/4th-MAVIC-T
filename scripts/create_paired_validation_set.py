#!/usr/bin/env python3
"""Create a paired validation set by randomly sampling 500 pairs per task from the train set.

Writes manifest files under dataset_root/manifests/paired_val_<task>.txt.
Each line: input_path<TAB>target_path (absolute paths for portability).

Usage:
    python scripts/create_paired_validation_set.py
    python scripts/create_paired_validation_set.py --dataset_root ./datasets/BiliSakura/MACIV-T-2025-Structure-Refined
    python scripts/create_paired_validation_set.py --n_pairs 500 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# Standalone constants to avoid heavy imports (torch, datasets, etc.)
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFINED_ROOT = _REPO_ROOT / "datasets/BiliSakura/MACIV-T-2025-Structure-Refined"
REFINED_MANIFEST_NAMES = ("refined_manifest.csv", "refined_manifest_crop_aug.csv")
TASKS = ("sar2eo", "rgb2ir", "sar2ir", "sar2rgb", "rgb2ir_crop_aug", "sar2ir_crop_aug", "sar2rgb_crop_aug")


def _resolve_dataset_root(root: Path) -> Path:
    if root.exists():
        return root
    root_str = str(root)
    if root_str.startswith("/mnt/data/"):
        alt = Path("/data") / Path(root_str).relative_to("/mnt/data")
        if alt.exists():
            return alt
    if root_str.startswith("/data/"):
        alt = Path("/mnt/data") / Path(root_str).relative_to("/data")
        if alt.exists():
            return alt
    return root


# Base tasks only (exclude _crop_aug for deduplication; we sample from base + crop_aug manifests)
BASE_TASKS = ("sar2eo", "rgb2ir", "sar2ir", "sar2rgb")


def load_train_pairs(
    refined_root: Path,
    task: str,
    exclude_paths: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Load all train (input, target) path pairs for a task from refined manifests."""
    pairs: list[tuple[str, str]] = []
    exclude = exclude_paths or set()

    # Include both base task and crop_aug variant to match full train set
    task_variants = [task]
    if f"{task}_crop_aug" in TASKS:
        task_variants.append(f"{task}_crop_aug")

    manifest_paths = [
        refined_root / "manifests" / name
        for name in REFINED_MANIFEST_NAMES
        if (refined_root / "manifests" / name).is_file()
    ]
    if not manifest_paths:
        raise FileNotFoundError(
            f"No manifest files found under {refined_root}/manifests "
            f"(expected one of: {', '.join(REFINED_MANIFEST_NAMES)})"
        )

    for manifest_path in manifest_paths:
        with manifest_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if (row.get("split") or "").lower() != "train":
                    continue
                row_task = (row.get("task") or "").lower()
                if row_task not in task_variants:
                    continue

                input_path = str((refined_root / row["input"]).resolve())
                target_path = str((refined_root / row["target"]).resolve())

                if input_path in exclude or target_path in exclude:
                    continue
                pairs.append((input_path, target_path))

    return pairs


def load_exclude_set(exclude_file: Path | None) -> set[str]:
    """Load absolute paths to exclude (e.g. bad_samples.txt)."""
    if not exclude_file or not exclude_file.is_file():
        return set()
    out: set[str] = set()
    with exclude_file.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.add(str(Path(line).resolve()))
    return out


def create_paired_validation_set(
    dataset_root: Path,
    n_pairs: int = 500,
    seed: int = 42,
    exclude_file: Path | None = None,
) -> dict[str, Path]:
    """Sample n_pairs per task and save to manifests. Returns paths to created files."""
    refined_root = _resolve_dataset_root(dataset_root)
    manifests_dir = refined_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    exclude = load_exclude_set(exclude_file) if exclude_file else set()

    rng = random.Random(seed)
    created: dict[str, Path] = {}

    for task in BASE_TASKS:
        pairs = load_train_pairs(refined_root, task, exclude)

        if len(pairs) < n_pairs:
            chosen = pairs
            print(f"  Warning: {task} has only {len(pairs)} pairs, using all")
        else:
            chosen = rng.sample(pairs, n_pairs)

        out_path = manifests_dir / f"paired_val_{task}.txt"
        with out_path.open("w") as fh:
            for inp, tgt in chosen:
                fh.write(f"{inp}\t{tgt}\n")

        created[task] = out_path
        print(f"  {task}: wrote {len(chosen)} pairs to {out_path}")

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Create paired validation set by sampling from train manifests"
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_REFINED_ROOT,
        help="Path to refined dataset root (contains manifests/)",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=500,
        help="Number of pairs to sample per task (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--exclude_file",
        type=Path,
        default=None,
        help="Optional path to bad_samples.txt to exclude from sampling",
    )
    args = parser.parse_args()

    exclude = args.exclude_file
    if exclude is None:
        default_exclude = args.dataset_root / "manifests" / "bad_samples.txt"
        if default_exclude.exists():
            exclude = default_exclude

    print(f"Creating paired validation set ({args.n_pairs} pairs per task)...")
    create_paired_validation_set(
        dataset_root=args.dataset_root,
        n_pairs=args.n_pairs,
        seed=args.seed,
        exclude_file=exclude,
    )
    print("Done.")


if __name__ == "__main__":
    main()
