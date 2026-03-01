#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Create submission.zip from MACIV-T-2025-Submissions folder.

The submission must contain 4 subdirectories (sar2eo, sar2rgb, rgb2ir, sar2ir)
and readme.txt at root. See official-docs/submission.md.

Usage::

    python scripts/create_submission_zip.py \\
        --submission_root datasets/BiliSakura/MACIV-T-2025-Submissions \\
        --model_name ddbm \\
        --output submission.zip
"""

import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("sar2eo", "sar2rgb", "rgb2ir", "sar2ir")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--submission_root",
        type=str,
        default=str(_PROJECT_ROOT / "datasets/BiliSakura/MACIV-T-2025-Submissions"),
    )
    p.add_argument("--model_name", type=str, default="ddbm")
    p.add_argument("--output", type=str, default="submission.zip")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.submission_root)
    out_path = Path(args.output)
    model = args.model_name

    if out_path.is_dir():
        print(f"Error: output {out_path} is a directory, expected zip path.", file=sys.stderr)
        sys.exit(1)

    work_dir = out_path.parent / (out_path.stem + "_tmp")
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        def _ignore(path, names):
            return [n for n in names if n == "readme.txt" or n.endswith(".npz")]

        for task in TASKS:
            src = root / task / model
            dst = work_dir / task
            if not src.is_dir():
                print(f"Warning: missing {src}, skipping {task}", file=sys.stderr)
                continue
            shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)

        # Copy readme from first available task
        readme_src = None
        for task in TASKS:
            cand = root / task / model / "readme.txt"
            if cand.is_file():
                readme_src = cand
                break
        if readme_src:
            shutil.copy(readme_src, work_dir / "readme.txt")
        else:
            (work_dir / "readme.txt").write_text(
                "runtime per image [s] : 0.00\n"
                "CPU[1] / GPU[0] : 0\n"
                "Extra Data [1] / No Extra Data [0] : 0\n"
                "Other description : DDBM submission.\n"
            )

        base = str(out_path.resolve().with_suffix(""))
        shutil.make_archive(base, "zip", work_dir)
        print(f"Created {base}.zip")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
