"""Filter training images that are entirely black or contain only N/A (NoData) pixels.

Satellite imagery often contains tiles where the sensor returned no useful data.
This script scans the refined training dataset and writes the absolute paths of
such "bad" images to a text file so they can be excluded from training.

Usage::

    # Scan all tasks, write bad paths to bad_samples.txt
    python scripts/filter_bad_samples.py

    # Scan a specific task
    python scripts/filter_bad_samples.py --tasks sar2ir sar2rgb

    # Custom output and black-pixel threshold
    python scripts/filter_bad_samples.py --output filtered.txt --black_thresh 1e-6

The generated text file can then be passed to any baseline trainer via the
``--exclude_file`` flag to skip these samples during training.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_REFINED_ROOT = (
    _PROJECT_ROOT / "datasets" / "BiliSakura" / "MACIV-T-2025-Structure-Refined"
)
DEFAULT_OUTPUT = _PROJECT_ROOT / "bad_samples.txt"
ALL_TASKS = ("sar2eo", "rgb2ir", "sar2ir", "sar2rgb",
             "rgb2ir_crop_aug", "sar2ir_crop_aug", "sar2rgb_crop_aug")

IMAGE_EXTS = {".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def is_bad_image(path: str, black_thresh: float = 0.0) -> bool:
    """Return ``True`` if the image at *path* is entirely black or all-NaN.

    Parameters
    ----------
    path : str
        Absolute path to the image file.
    black_thresh : float
        Pixel values at or below this threshold are treated as "black".
        Defaults to ``0.0`` (strict all-zero check).
    """
    try:
        img = Image.open(path)
        arr = np.array(img, dtype=np.float64)
    except Exception:
        # Unreadable / corrupt file counts as bad
        return True

    # Check for all-NaN (possible in float TIFFs)
    if np.isnan(arr).all():
        return True

    # Replace NaN with 0 for the max check
    finite = np.nan_to_num(arr, nan=0.0)
    if finite.max() <= black_thresh:
        return True

    return False


def _check_one(args: Tuple[str, float]) -> Tuple[str, bool]:
    """Worker function for multiprocessing."""
    path, thresh = args
    return path, is_bad_image(path, black_thresh=thresh)


# ---------------------------------------------------------------------------
# Manifest-based scanning
# ---------------------------------------------------------------------------

def collect_paths_from_manifests(
    refined_root: Path,
    tasks: Optional[List[str]] = None,
) -> List[str]:
    """Return all training image paths (input + target) from the manifest CSVs."""
    manifest_dir = refined_root / "manifests"
    csv_files = sorted(manifest_dir.glob("refined_manifest*.csv")) if manifest_dir.is_dir() else []

    paths: List[str] = []
    task_set = set(tasks) if tasks else None

    for csv_path in csv_files:
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row_task = (row.get("task") or "").lower()
                if task_set and row_task not in task_set:
                    continue
                for col in ("input", "target"):
                    rel = row.get(col, "")
                    if rel:
                        abs_path = str((refined_root / rel).resolve())
                        paths.append(abs_path)

    return paths


def collect_paths_from_dirs(
    refined_root: Path,
    tasks: Optional[List[str]] = None,
) -> List[str]:
    """Fallback: walk task directories to find images when no manifest exists."""
    paths: List[str] = []
    task_list = tasks if tasks else list(ALL_TASKS)

    for task in task_list:
        for sub in ("input", "target"):
            d = refined_root / task / "train" / sub
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    paths.append(str(f.resolve()))

    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter bad (all-black / all-NA) training images from MAVIC-T."
    )
    parser.add_argument(
        "--refined_root",
        type=str,
        default=str(DEFAULT_REFINED_ROOT),
        help="Root of the refined dataset (default: datasets/BiliSakura/MACIV-T-2025-Structure-Refined).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=f"Tasks to scan (default: all). Choices: {', '.join(ALL_TASKS)}",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output text file listing bad image paths (default: bad_samples.txt).",
    )
    parser.add_argument(
        "--black_thresh",
        type=float,
        default=0.0,
        help="Pixel-value threshold for black detection (default: 0.0, strict all-zero).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="Number of parallel workers (default: min(32, cpu_count)).",
    )
    args = parser.parse_args()

    refined_root = Path(args.refined_root)
    tasks = [t.lower() for t in args.tasks] if args.tasks else None

    if tasks:
        for t in tasks:
            if t not in ALL_TASKS:
                parser.error(f"Unknown task '{t}'. Choose from: {', '.join(ALL_TASKS)}")

    # Collect image paths
    print(f"Scanning refined root: {refined_root}")
    paths = collect_paths_from_manifests(refined_root, tasks)
    if not paths:
        print("  No manifest entries found; falling back to directory scan.")
        paths = collect_paths_from_dirs(refined_root, tasks)

    # Deduplicate (input and target may share paths across manifests)
    paths = sorted(set(paths))
    print(f"  Found {len(paths)} unique image files to check.")

    if not paths:
        print("Nothing to check. Exiting.")
        return

    # Check images in parallel
    t0 = time.time()
    bad_paths: List[str] = []
    work = [(p, args.black_thresh) for p in paths]

    if args.workers <= 1:
        for w in work:
            path, is_bad = _check_one(w)
            if is_bad:
                bad_paths.append(path)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_check_one, w): w[0] for w in work}
            for fut in as_completed(futures):
                path, is_bad = fut.result()
                if is_bad:
                    bad_paths.append(path)

    bad_paths.sort()
    elapsed = time.time() - t0

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for p in bad_paths:
            fh.write(p + "\n")

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  Total images checked : {len(paths)}")
    print(f"  Bad images found     : {len(bad_paths)}")
    print(f"  Output written to    : {out_path}")


if __name__ == "__main__":
    main()
