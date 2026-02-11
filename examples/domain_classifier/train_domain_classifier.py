#!/usr/bin/env python
"""Train a ResNet-18 target-domain real/fake classifier.

Usage examples::

    # Train IR-domain classifier (default)
    python -m examples.domain_classifier.train_domain_classifier

    # Train EO-domain classifier
    python -m examples.domain_classifier.train_domain_classifier --target_domain eo

    # Resume from latest checkpoint in the run directory
    python -m examples.domain_classifier.train_domain_classifier --resume_from_checkpoint latest
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Union, get_args, get_origin, get_type_hints

# Ensure project root is importable when executed as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import DomainClassifierConfig  # noqa: E402
from .trainer import DomainClassifierTrainer  # noqa: E402


def _bool_arg(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _resolve_arg_type(field_name: str, default_value):
    if isinstance(default_value, bool):
        return _bool_arg
    if default_value is not None:
        return type(default_value)

    hints = get_type_hints(DomainClassifierConfig)
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


def parse_overrides() -> dict:
    parser = argparse.ArgumentParser(
        description="Pre-train target-domain ResNet-18 real/fake classifier"
    )
    defaults = DomainClassifierConfig()
    for field_name, field_value in vars(defaults).items():
        parser.add_argument(
            f"--{field_name}",
            type=_resolve_arg_type(field_name, field_value),
            default=field_value,
        )
    args = parser.parse_args()
    return vars(args)


def main() -> None:
    overrides = parse_overrides()
    cfg = DomainClassifierConfig(**overrides)
    trainer = DomainClassifierTrainer(cfg)
    result = trainer.train()
    print(result)


if __name__ == "__main__":
    main()

