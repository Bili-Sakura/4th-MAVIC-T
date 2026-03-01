#!/usr/bin/env python3
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Rewrite absolute paths in manifest files to use a user-specified dataset root.

Manifest files under datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/
(paired_val_*.txt, paired_sar2rgb_sup.txt, bad_samples.txt) may contain absolute paths
or relative paths. Relative paths are resolved against the dataset root.
(e.g. /mnt/data/expansion/...). This script rewrites them so paths resolve to your
dataset location.

Note: refined_manifest.csv and refined_manifest_crop_aug.csv use relative paths
already and are not modified.

Usage:

    # Rewrite manifests to use PROJECT_ROOT (from paths.env or env)
    python scripts/rewrite_manifest_paths.py

    # Rewrite to a specific dataset root (e.g. after cloning to a new machine)
    python scripts/rewrite_manifest_paths.py --dataset_root /path/to/MACIV-T-2025-Structure-Refined

    # Dry run (print changes without writing)
    python scripts/rewrite_manifest_paths.py --dry_run

    # Only process specific files
    python scripts/rewrite_manifest_paths.py --files paired_val_sar2rgb.txt bad_samples.txt

The script detects paths containing "MACIV-T-2025-Structure-Refined", extracts the
relative part (e.g. sar2rgb/train/input/file.tif), and prepends the target dataset root.
Supports both path patterns found in manifests (including symlinked locations):
  - /mnt/data/expansion/datasets/BiliSakura/MACIV-T-2025-Structure-Refined/
  - /mnt/data/data/hf_datasets/BiliSakura/MACIV-T-2025-Structure-Refined/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Default dataset root: project_root/datasets/BiliSakura/MACIV-T-2025-Structure-Refined
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATASET_ROOT = _PROJECT_ROOT / "datasets" / "BiliSakura" / "MACIV-T-2025-Structure-Refined"

# Manifest files that may contain absolute or relative paths
DEFAULT_MANIFEST_FILES = (
    "paired_val_sar2eo.txt",
    "paired_val_sar2rgb.txt",
    "paired_val_sar2ir.txt",
    "paired_val_rgb2ir.txt",
    "paired_sar2rgb_sup.txt",
    "bad_samples.txt",
)

# Pattern: path contains MACIV-T-2025-Structure-Refined, extract relative part after it.
# Manifests may contain links from two locations:
#   /mnt/data/expansion/datasets/BiliSakura/MACIV-T-2025-Structure-Refined/
#   /mnt/data/data/hf_datasets/BiliSakura/MACIV-T-2025-Structure-Refined/
DATASET_ANCHOR = "MACIV-T-2025-Structure-Refined"
_RELATIVE_PATTERN = re.compile(
    r".*" + re.escape(DATASET_ANCHOR) + r"[/\\]?(.*)$",
    re.IGNORECASE,
)


def _extract_relative(path_str: str) -> str | None:
    """Extract the part after MACIV-T-2025-Structure-Refined, or None if not matched.
    Handles both path patterns:
      /mnt/data/expansion/datasets/BiliSakura/MACIV-T-2025-Structure-Refined/...
      /mnt/data/data/hf_datasets/BiliSakura/MACIV-T-2025-Structure-Refined/...
    """
    path_str = path_str.strip()
    m = _RELATIVE_PATTERN.match(path_str)
    if m:
        rel = m.group(1).replace("\\", "/")
        # Remove leading slash if present
        if rel.startswith("/"):
            rel = rel[1:]
        return rel
    return None


def _rewrite_path(path_str: str, dataset_root: Path) -> str:
    """Rewrite a path to use dataset_root. Handles absolute paths (extract rel) and
    relative paths (prepend dataset_root)."""
    rel = _extract_relative(path_str)
    if rel is not None:
        return str((dataset_root / rel).resolve())
    # Already relative (e.g. sar2rgb_sup/train/input/xxx.png) - prepend dataset_root
    p = Path(path_str)
    if not p.is_absolute() and path_str.strip():
        return str((dataset_root / path_str.strip()).resolve())
    return path_str


def rewrite_manifest_file(
    manifest_path: Path,
    dataset_root: Path,
    *,
    tab_separated: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Rewrite paths in a manifest file. Returns (lines_processed, paths_rewritten)."""
    if not manifest_path.is_file():
        return 0, 0

    text = manifest_path.read_text()
    lines = text.splitlines()
    new_lines: list[str] = []
    paths_rewritten = 0

    for line in lines:
        orig = line
        if tab_separated:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                p1, p2 = parts[0].strip(), parts[1].strip()
                np1 = _rewrite_path(p1, dataset_root)
                np2 = _rewrite_path(p2, dataset_root)
                if np1 != p1 or np2 != p2:
                    paths_rewritten += (1 if np1 != p1 else 0) + (1 if np2 != p2 else 0)
                new_lines.append(f"{np1}\t{np2}")
            else:
                new_lines.append(line)
        else:
            # One path per line (e.g. bad_samples.txt)
            new_path = _rewrite_path(line, dataset_root)
            if new_path != line.strip():
                paths_rewritten += 1
            new_lines.append(new_path)

    if not dry_run and paths_rewritten > 0:
        manifest_path.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""))

    return len(lines), paths_rewritten


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite absolute paths in manifest files to use a user dataset root."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help="Target dataset root (default: PROJECT_ROOT/datasets/BiliSakura/MACIV-T-2025-Structure-Refined)",
    )
    parser.add_argument(
        "--manifests_dir",
        type=Path,
        default=None,
        help="Directory containing manifest files (default: repo datasets/BiliSakura/.../manifests)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help=f"Manifest filenames to process (default: {list(DEFAULT_MANIFEST_FILES)})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be changed without writing",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if dataset_root is None:
        import os
        project_root = os.environ.get("PROJECT_ROOT")
        if project_root:
            dataset_root = Path(project_root) / "datasets" / "BiliSakura" / "MACIV-T-2025-Structure-Refined"
        else:
            dataset_root = _DEFAULT_DATASET_ROOT

    manifests_dir = args.manifests_dir
    if manifests_dir is None:
        # Default: manifests live in repo under dataset structure
        manifests_dir = _DEFAULT_DATASET_ROOT / "manifests"

    if not manifests_dir.is_dir():
        print(f"Error: manifests directory not found: {manifests_dir}", file=sys.stderr)
        sys.exit(1)

    files = args.files or list(DEFAULT_MANIFEST_FILES)
    tab_separated = {
        "paired_val_sar2eo.txt",
        "paired_val_sar2rgb.txt",
        "paired_val_sar2ir.txt",
        "paired_val_rgb2ir.txt",
        "paired_sar2rgb_sup.txt",
    }

    print(f"Dataset root: {dataset_root}")
    print(f"Manifests dir: {manifests_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    total_lines = 0
    total_rewritten = 0

    for fn in files:
        manifest_path = manifests_dir / fn
        if not manifest_path.is_file():
            print(f"  {fn}: not found, skipping")
            continue
        n_lines, n_rewritten = rewrite_manifest_file(
            manifest_path,
            dataset_root,
            tab_separated=(fn in tab_separated),
            dry_run=args.dry_run,
        )
        total_lines += n_lines
        total_rewritten += n_rewritten
        status = " (dry run)" if args.dry_run and n_rewritten else ""
        print(f"  {fn}: {n_lines} lines, {n_rewritten} paths rewritten{status}")

    print()
    print(f"Total: {total_rewritten} paths rewritten across {total_lines} lines.")
    if args.dry_run and total_rewritten > 0:
        print("Run without --dry_run to apply changes.")


if __name__ == "__main__":
    main()
