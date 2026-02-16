"""Task-specific configurations for BDBM baseline training."""

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
    """Configuration for a single BDBM image-to-image translation task."""

    # BDBM-specific defaults
    condition_mode: str = "dual"
    objective: str = "noise"  # options: "noise", "sum", "both"
    loss_type: str = "l2"
    max_var: float = 1.0
    sample_step: int = 200
    num_inference_steps: int = 200


def _to_bdbm_config(base_cfg: _BiBBDMTaskConfig) -> TaskConfig:
    cfg = TaskConfig(**vars(base_cfg))
    cfg.condition_mode = "dual"
    cfg.objective = "noise"
    cfg.loss_type = "l2"
    cfg.max_var = 1.0
    cfg.sample_step = 200
    cfg.num_inference_steps = 200
    return cfg


def sar2eo_config(**overrides) -> TaskConfig:
    cfg = _to_bdbm_config(_bibbdm_sar2eo_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    cfg = _to_bdbm_config(_bibbdm_rgb2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    cfg = _to_bdbm_config(_bibbdm_sar2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    cfg = _to_bdbm_config(_bibbdm_sar2rgb_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
