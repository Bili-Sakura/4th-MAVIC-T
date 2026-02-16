"""Task-specific configurations for CDTSDE baseline training."""

from __future__ import annotations

from dataclasses import dataclass

from examples.ddbm.config import (
    TaskConfig as _DDBMTaskConfig,
    sar2eo_config as _ddbm_sar2eo_config,
    rgb2ir_config as _ddbm_rgb2ir_config,
    sar2ir_config as _ddbm_sar2ir_config,
    sar2rgb_config as _ddbm_sar2rgb_config,
)


@dataclass
class TaskConfig(_DDBMTaskConfig):
    """Configuration for a single CDTSDE image-to-image translation task."""

    # CDTSDE scheduler
    cdtsde_num_train_timesteps: int = 1000
    beta_schedule: str = "linear"
    beta_start: float = 0.00085
    beta_end: float = 0.0120
    eta_schedule: str = "trunc_12"
    eta_start: float = 0.001
    eta_end: float = 0.999

    # CDTSDE model
    lambda_hidden_channels: int = 16

    # CDTSDE inference
    num_inference_steps: int = 50
    stochastic: bool = True
    apply_domain_shift: bool = True


def _to_cdtsde_config(base_cfg: _DDBMTaskConfig) -> TaskConfig:
    cfg = TaskConfig(**vars(base_cfg))
    cfg.cdtsde_num_train_timesteps = 1000
    cfg.beta_schedule = "linear"
    cfg.beta_start = 0.00085
    cfg.beta_end = 0.0120
    cfg.eta_schedule = "trunc_12"
    cfg.eta_start = 0.001
    cfg.eta_end = 0.999
    cfg.lambda_hidden_channels = 16
    cfg.num_inference_steps = 50
    cfg.stochastic = True
    cfg.apply_domain_shift = True
    return cfg


def sar2eo_config(**overrides) -> TaskConfig:
    cfg = _to_cdtsde_config(_ddbm_sar2eo_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    cfg = _to_cdtsde_config(_ddbm_rgb2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    cfg = _to_cdtsde_config(_ddbm_sar2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    cfg = _to_cdtsde_config(_ddbm_sar2rgb_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg

