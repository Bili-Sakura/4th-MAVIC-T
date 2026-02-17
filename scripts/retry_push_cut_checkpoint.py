#!/usr/bin/env python3
"""Retry pushing a CUT checkpoint to the Hub (e.g. after a ReadTimeout).

Usage::

    # Push the failed sar2eo_large checkpoint-epoch-1 (default)
    python scripts/retry_push_cut_checkpoint.py

    # Custom paths
    python scripts/retry_push_cut_checkpoint.py \\
        --save_dir ckpt/4th-MAVIC-T-ckpt-0217/sar2eo_large/cut/sar2eo/checkpoint-epoch-1 \\
        --hub_model_id BiliSakura/4th-MAVIC-T-ckpt-0217 \\
        --path_in_repo cut/sar2eo/checkpoint-epoch-1 \\
        --timeout 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.training_utils import push_checkpoint_to_hub


def main():
    parser = argparse.ArgumentParser(description="Retry pushing a CUT checkpoint to the Hub.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ckpt/4th-MAVIC-T-ckpt-0217/sar2eo_large/cut/sar2eo/checkpoint-epoch-1",
        help="Local checkpoint directory to upload.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="BiliSakura/4th-MAVIC-T-ckpt-0217",
        help="Hub repository ID (e.g. BiliSakura/4th-MAVIC-T-ckpt-0217).",
    )
    parser.add_argument(
        "--path_in_repo",
        type=str,
        default="cut/sar2eo/checkpoint-epoch-1",
        help="Path inside the repo (e.g. cut/sar2eo/checkpoint-epoch-1).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="HTTP request timeout in seconds (default 600).",
    )
    args = parser.parse_args()

    save_path = project_root / args.save_dir
    if not save_path.exists():
        print(f"Error: checkpoint path does not exist: {save_path}")
        sys.exit(1)

    print(f"Pushing: {save_path}")
    print(f"Hub: {args.hub_model_id} (path_in_repo={args.path_in_repo})")
    print(f"Timeout: {args.timeout}s")
    push_checkpoint_to_hub(
        save_dir=str(save_path),
        hub_model_id=args.hub_model_id,
        commit_message=f"cut sar2eo checkpoint-epoch-1 (retry)",
        path_in_repo=args.path_in_repo,
        request_timeout=args.timeout,
    )
    print("Done.")


if __name__ == "__main__":
    main()
