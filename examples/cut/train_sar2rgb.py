#!/usr/bin/env python
"""Train CUT baseline for **sar2rgb** (SAR → RGB, 1-band → 3-band, 1024×1024).

Usage::

    # Single GPU
    python -m examples.cut.train_sar2rgb

    # Multi-GPU via accelerate
    accelerate launch -m examples.cut.train_sar2rgb --train_batch_size 4

All default hyper-parameters live in :func:`config.sar2rgb_config`.
Any field can be overridden from the command line (``--field value``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import sar2rgb_config, TaskConfig  # noqa: E402
from .trainer import CUTTrainer  # noqa: E402


def parse_overrides() -> dict:
    parser = argparse.ArgumentParser(description="Train CUT – sar2rgb")
    for field_name, field_val in vars(sar2rgb_config()).items():
        ftype = type(field_val) if field_val is not None else str
        if isinstance(field_val, bool):
            parser.add_argument(f"--{field_name}", type=lambda v: v.lower() in ("true", "1", "yes"), default=field_val)
        else:
            parser.add_argument(f"--{field_name}", type=ftype, default=field_val)
    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}


def main():
    overrides = parse_overrides()
    cfg = sar2rgb_config(**overrides)

    # ------------------------------------------------------------------
    # Task-specific modifications can be added here.
    # For example:  cfg.netG = "resnet_6blocks"
    # ------------------------------------------------------------------

    trainer = CUTTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
