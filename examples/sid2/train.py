#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Train standalone SID2 baseline for MAVIC-T tasks.

Usage:
    python -m examples.sid2.train --task rgb2ir
    accelerate launch -m examples.sid2.train --task sar2ir --train_batch_size 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union, get_args, get_origin, get_type_hints

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import (  # noqa: E402
    TaskConfig,
    rgb2ir_config,
    sar2eo_config,
    sar2ir_config,
    sar2rgb_config,
)
from .trainer import SID2Trainer  # noqa: E402


_TASK_TO_CONFIG = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def _bool_arg(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _resolve_arg_type(field_name: str, field_val):
    if isinstance(field_val, bool):
        return _bool_arg
    if field_val is not None:
        return type(field_val)

    hints = get_type_hints(TaskConfig)
    annotation = hints.get(field_name, str)
    origin = get_origin(annotation)
    if origin is Union:
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]
    if annotation is bool:
        return _bool_arg
    if annotation in (int, float, str):
        return annotation
    return str


def parse_args():
    parser = argparse.ArgumentParser(description="Train standalone SID2 baseline")
    parser.add_argument("--task", choices=sorted(_TASK_TO_CONFIG.keys()), required=True)

    known, _ = parser.parse_known_args()
    cfg_builder = _TASK_TO_CONFIG[known.task]
    default_cfg = cfg_builder()

    for field_name, field_val in vars(default_cfg).items():
        ftype = _resolve_arg_type(field_name, field_val)
        parser.add_argument(f"--{field_name}", type=ftype, default=field_val)
    return parser.parse_args()


def main():
    args = parse_args()
    args_dict = vars(args)
    task = args_dict.pop("task")

    cfg_builder = _TASK_TO_CONFIG[task]
    cfg = cfg_builder(**args_dict)

    trainer = SID2Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
