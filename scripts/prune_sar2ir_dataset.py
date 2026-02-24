#!/usr/bin/env python3
"""Prune sar2ir dataset based on IR quality metrics.

Reads the metrics CSV from ir_quality_metrics.py and filters out low-quality
IR images based on configurable thresholds. Outputs paths to exclude (compatible
with --exclude_file).

Pruned output is saved under:
  datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/

Usage::

    # Prune using default thresholds
    python scripts/prune_sar2ir_dataset.py --metrics datasets/.../manifests/ir_quality_metrics.csv

    # Stricter thresholds
    python scripts/prune_sar2ir_dataset.py \
        --metrics datasets/.../manifests/ir_quality_metrics.csv \
        --min_variance 100 --min_laplacian 50 --max_clipping 0.05

    # Merge with existing exclude file
    python scripts/prune_sar2ir_dataset.py \
        --metrics datasets/.../manifests/ir_quality_metrics.csv \
        --merge datasets/.../manifests/bad_samples.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

MANIFESTS_DIR = (
    _PROJECT_ROOT / "datasets" / "BiliSakura" / "MACIV-T-2025-Structure-Refined" / "manifests"
)
DEFAULT_METRICS = MANIFESTS_DIR / "ir_quality_metrics.csv"
DEFAULT_OUTPUT = MANIFESTS_DIR / "pruned_sar2ir_exclude.txt"


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Pruning logic
# ---------------------------------------------------------------------------


def load_metrics(csv_path: Path) -> list[dict]:
    """Load metrics CSV. Expects columns: Path, Variance, Laplacian_Variance, Clipping_Ratio."""
    rows: list[dict] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in tqdm(reader, desc="Loading metrics", unit="row"):
            path = row.get("Path") or row.get("path", "")
            if not path:
                continue
            var = _parse_float(row.get("Variance") or row.get("variance", ""))
            lap = _parse_float(row.get("Laplacian_Variance") or row.get("laplacian_variance", ""))
            clip = _parse_float(row.get("Clipping_Ratio") or row.get("clipping_ratio", ""))
            if var is None and lap is None and clip is None:
                continue
            rows.append({
                "path": path,
                "variance": var,
                "laplacian": lap,
                "clipping": clip,
            })
    return rows


def filter_bad_by_thresholds(
    rows: list[dict],
    min_variance: float | None = None,
    min_laplacian: float | None = None,
    max_clipping: float | None = None,
) -> list[str]:
    """Return paths of images that fail any threshold (should be excluded)."""
    bad_paths: list[str] = []

    for r in tqdm(rows, desc="Filtering by thresholds", unit="row"):
        path = r["path"]
        var, lap, clip = r["variance"], r["laplacian"], r["clipping"]

        bad = False
        if min_variance is not None and var is not None and var < min_variance:
            bad = True
        if min_laplacian is not None and lap is not None and lap < min_laplacian:
            bad = True
        if max_clipping is not None and clip is not None and clip > max_clipping:
            bad = True

        if bad:
            bad_paths.append(path)

    return bad_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune sar2ir dataset based on IR quality metrics."
    )
    parser.add_argument(
        "--metrics", "-m",
        type=str,
        default=str(DEFAULT_METRICS),
        help="Path to metrics CSV from ir_quality_metrics.py.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output file listing paths to exclude. Default: manifests/pruned_sar2ir_exclude.txt",
    )
    parser.add_argument(
        "--min_variance",
        type=float,
        default=None,
        help="Minimum variance (thermal contrast). Images below are pruned.",
    )
    parser.add_argument(
        "--min_laplacian",
        type=float,
        default=None,
        help="Minimum Laplacian variance (sharpness). Images below are pruned.",
    )
    parser.add_argument(
        "--max_clipping",
        type=float,
        default=None,
        help="Maximum clipping ratio. Images above are pruned (e.g. 0.05 = 5%%).",
    )
    parser.add_argument(
        "--merge",
        type=str,
        default=None,
        help="Merge with existing exclude file. Output will include both pruned + merged paths.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print stats only, do not write output file.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.is_file():
        metrics_path = _PROJECT_ROOT / args.metrics
    if not metrics_path.is_file():
        print(f"Metrics file not found: {args.metrics}", file=sys.stderr)
        print("Run ir_quality_metrics.py first.", file=sys.stderr)
        return 1

    rows = load_metrics(metrics_path)
    if not rows:
        print("No valid rows in metrics CSV.", file=sys.stderr)
        return 1

    # If no thresholds given, use sensible defaults (prune obvious bad samples)
    min_var = args.min_variance
    min_lap = args.min_laplacian
    max_clip = args.max_clipping

    if min_var is None and min_lap is None and max_clip is None:
        # Default: prune very flat (low variance), very blurry (low laplacian), or heavily clipped
        min_var = 10.0
        min_lap = 1.0
        max_clip = 0.15
        print("Using default thresholds: min_variance=10, min_laplacian=1, max_clipping=0.15")
        print("Override with --min_variance, --min_laplacian, --max_clipping")

    bad_paths = filter_bad_by_thresholds(rows, min_var, min_lap, max_clip)

    # Merge with existing exclude file if requested
    all_exclude: set[str] = set(bad_paths)
    if args.merge:
        merge_path = Path(args.merge)
        if not merge_path.is_file():
            merge_path = _PROJECT_ROOT / args.merge
        if merge_path.is_file():
            with merge_path.open() as fh:
                for line in fh:
                    p = line.strip()
                    if p:
                        all_exclude.add(p)
            print(f"Merged {len(all_exclude) - len(bad_paths)} paths from {merge_path}")

    all_exclude_sorted = sorted(all_exclude)

    print(f"Total metrics rows : {len(rows)}")
    print(f"Pruned (bad IR)   : {len(bad_paths)}")
    print(f"Total to exclude  : {len(all_exclude_sorted)}")

    if args.dry_run:
        print("(dry run, no file written)")
        return 0

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fh:
        for p in tqdm(all_exclude_sorted, desc="Writing exclude file", unit="path"):
            fh.write(p + "\n")

    print(f"Output written to : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
