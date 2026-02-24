#!/usr/bin/env python3
"""Compute IR quality metrics for sar2ir dataset pruning.

Evaluates IR (target) images for:
  - Thermal contrast (variance): high = good contrast, low = washed out
  - Structural sharpness (Laplacian variance): high = sharp, low = blurry
  - Saturation/clipping ratio: high = many pixels maxed out or dead zero

Usage::

    # From sar2ir manifest (refined_manifest*.csv) - default
    python scripts/ir_quality_metrics.py

    # From a directory of IR images
    python scripts/ir_quality_metrics.py --ir_dir /path/to/IR_images

    # Custom output
    python scripts/ir_quality_metrics.py -o datasets/.../manifests/ir_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_REFINED_ROOT = (
    _PROJECT_ROOT / "datasets" / "BiliSakura" / "MACIV-T-2025-Structure-Refined"
)
DEFAULT_MANIFESTS = DEFAULT_REFINED_ROOT / "manifests"
SAR2IR_TASKS = ("sar2ir", "sar2ir_crop_aug")
IMAGE_EXTS = (".png", ".tif", ".tiff")


def _resolve_path(root: Path, rel_path: str) -> Path:
    """Resolve path, trying /mnt/data <-> /data alternates if needed."""
    p = (root / rel_path).resolve()
    if p.exists():
        return p
    root_str = str(root)
    if root_str.startswith("/data/"):
        alt_root = Path("/mnt/data") / Path(root_str).relative_to("/data")
        p2 = (alt_root / rel_path).resolve()
        if p2.exists():
            return p2
    if root_str.startswith("/mnt/"):
        alt_root = Path("/data") / Path(root_str).relative_to("/mnt/data")
        p2 = (alt_root / rel_path).resolve()
        if p2.exists():
            return p2
    return p


# ---------------------------------------------------------------------------
# IR quality evaluation
# ---------------------------------------------------------------------------


def evaluate_ir_quality(image_path: str) -> tuple[float | None, float | None, float | None]:
    """Evaluate an IR image for thermal contrast, sharpness, and saturation clipping.

    Works for both 8-bit and 16-bit single-channel images.

    Returns:
        (variance, laplacian_variance, clipping_ratio) or (None, None, None) on failure.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None, None, None

    # If multi-channel, take first channel
    if img.ndim == 3:
        img = img[:, :, 0]

    # 1. Thermal Contrast (Variance)
    variance = float(np.var(img))

    # 2. Structural Sharpness (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(img, cv2.CV_64F).var())

    # 3. Saturation / Clipping Filter
    total_pixels = img.shape[0] * img.shape[1]
    if img.dtype == np.uint8:
        max_val = 255
    elif img.dtype == np.uint16:
        max_val = 65535
    else:
        max_val = float(np.max(img))

    clipped_pixels = int(np.sum((img == 0) | (img == max_val)))
    clipping_ratio = clipped_pixels / total_pixels

    return variance, laplacian_var, clipping_ratio


def _eval_one(ir_path: str) -> tuple[str, float | None, float | None, float | None]:
    """Worker helper for parallel evaluation."""
    var, lap_var, clip_ratio = evaluate_ir_quality(ir_path)
    return ir_path, var, lap_var, clip_ratio


def collect_ir_paths_from_manifests(
    refined_root: Path,
    tasks: tuple[str, ...] = SAR2IR_TASKS,
) -> list[tuple[str, str]]:
    """Return (input_path, target_path) pairs for sar2ir from manifest CSVs."""
    manifest_dir = refined_root / "manifests"
    csv_files = sorted(manifest_dir.glob("refined_manifest*.csv")) if manifest_dir.is_dir() else []

    pairs: list[tuple[str, str]] = []
    task_set = set(tasks)

    for csv_path in csv_files:
        with csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row_task = (row.get("task") or "").lower()
                if row_task not in task_set:
                    continue
                if (row.get("split") or "").lower() != "train":
                    continue
                inp, tgt = row.get("input", ""), row.get("target", "")
                if not inp or not tgt:
                    continue
                inp_resolved = str(_resolve_path(refined_root, inp))
                tgt_resolved = str(_resolve_path(refined_root, tgt))
                if Path(tgt_resolved).exists():
                    pairs.append((inp_resolved, tgt_resolved))

    return pairs


def collect_ir_paths_from_dir(ir_directory: str) -> list[str]:
    """Return list of IR image paths from a directory."""
    ir_dir = Path(ir_directory)
    if not ir_dir.is_dir():
        return []

    paths: list[str] = []
    for ext in IMAGE_EXTS:
        paths.extend(ir_dir.glob(f"*{ext}"))

    return sorted(str(p.resolve()) for p in paths)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute IR quality metrics for sar2ir dataset pruning."
    )
    parser.add_argument(
        "--ir_dir",
        type=str,
        default=None,
        help="Directory of IR images. If omitted, uses sar2ir from refined manifests.",
    )
    parser.add_argument(
        "--refined_root",
        type=str,
        default=str(DEFAULT_REFINED_ROOT),
        help="Refined dataset root when using manifest (default mode).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: datasets/.../manifests/ir_quality_metrics.csv",
    )
    parser.add_argument(
        "--include_input",
        action="store_true",
        help="Include input (SAR) path column when using manifest.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers for evaluation (default: 32).",
    )
    args = parser.parse_args()

    if not args.ir_dir:
        refined_root = Path(args.refined_root)
        if not refined_root.exists():
            refined_root = _PROJECT_ROOT / args.refined_root
        pairs = collect_ir_paths_from_manifests(refined_root)
        ir_paths = [tgt for _, tgt in pairs]
        input_map = {tgt: inp for inp, tgt in pairs} if args.include_input else {}
        if not ir_paths:
            print("No sar2ir pairs found in manifests.", file=sys.stderr)
            return 1
        print(f"Found {len(ir_paths)} sar2ir IR images from manifests.")
    else:
        ir_paths = collect_ir_paths_from_dir(args.ir_dir)
        input_map = {}
        if not ir_paths:
            print(f"No images found in {args.ir_dir}", file=sys.stderr)
            return 1
        print(f"Found {len(ir_paths)} images in {args.ir_dir}.")

    # Default output to manifests dir
    if args.output is None:
        out_path = DEFAULT_MANIFESTS / "ir_quality_metrics.csv"
    else:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = _PROJECT_ROOT / args.output

    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, float | None, float | None, float | None]] = []
    if args.workers <= 1:
        for ir_path in tqdm(ir_paths, desc="Evaluating IR quality", unit="img"):
            results.append(_eval_one(ir_path))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_eval_one, p): p for p in ir_paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating IR quality", unit="img"):
                results.append(fut.result())

    with out_path.open("w", newline="") as fh:
        cols = ["Path", "Filename", "Variance", "Laplacian_Variance", "Clipping_Ratio"]
        if input_map:
            cols.insert(0, "Input_Path")
        writer = csv.writer(fh)
        writer.writerow(cols)

        for ir_path, var, lap_var, clip_ratio in results:
            if var is not None:
                filename = os.path.basename(ir_path)
                row = [ir_path, filename, var, lap_var, clip_ratio]
                if input_map and ir_path in input_map:
                    row.insert(0, input_map[ir_path])
                writer.writerow(row)

    print(f"Done. Metrics saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
