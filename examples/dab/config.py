# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Task-specific configurations for DAB baseline training."""

from __future__ import annotations

from dataclasses import dataclass

from examples.bibbdm.config import (
    TaskConfig as _BiBBDMTaskConfig,
    sar2eo_config as _bibbdm_sar2eo_config,
    rgb2ir_config as _bibbdm_rgb2ir_config,
    sar2ir_config as _bibbdm_sar2ir_config,
    sar2rgb_config as _bibbdm_sar2rgb_config,
)


@dataclass
class TaskConfig(_BiBBDMTaskConfig):
    """Configuration for a single DAB image-to-image translation task."""

    # DAB-specific defaults
    condition_mode: str = "concat"
    objective: str = "grad"  # options: "grad", "noise", "ysubx"
    loss_type: str = "l1"
    max_var: float = 1.0
    sample_step: int = 1000
    num_inference_steps: int = 200
    sample_type: str = "linear"


def _to_dab_config(base_cfg: _BiBBDMTaskConfig) -> TaskConfig:
    cfg = TaskConfig(**vars(base_cfg))
    cfg.condition_mode = "concat"
    cfg.objective = "grad"
    cfg.loss_type = "l1"
    cfg.max_var = 1.0
    cfg.sample_step = 1000
    cfg.num_inference_steps = 200
    cfg.sample_type = "linear"
    return cfg


def sar2eo_config(**overrides) -> TaskConfig:
    cfg = _to_dab_config(_bibbdm_sar2eo_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    cfg = _to_dab_config(_bibbdm_rgb2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    cfg = _to_dab_config(_bibbdm_sar2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    cfg = _to_dab_config(_bibbdm_sar2rgb_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
