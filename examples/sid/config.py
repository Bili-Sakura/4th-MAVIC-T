# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Task-specific configurations for standalone SiD baseline training."""

from __future__ import annotations

from examples.ddbm.config import (
    TaskConfig,
    rgb2ir_config as _ddbm_rgb2ir_config,
    sar2eo_config as _ddbm_sar2eo_config,
    sar2ir_config as _ddbm_sar2ir_config,
    sar2rgb_config as _ddbm_sar2rgb_config,
)


def _apply_sid_defaults(cfg: TaskConfig) -> TaskConfig:
    """Convert a DDBM task config into a standalone SiD baseline config."""
    cfg.unet_type = "sid"
    cfg.use_multiscale_loss = True
    cfg.multiscale_base_resolution = 32

    # SiD scheduler parameters
    setattr(cfg, "prediction_type", "v")
    setattr(cfg, "logsnr_min", -15.0)
    setattr(cfg, "logsnr_max", 15.0)
    setattr(cfg, "noise_d", 64.0)
    setattr(cfg, "clip_sample", True)
    setattr(cfg, "num_train_timesteps", 1000)
    return cfg


def sar2eo_config(**overrides) -> TaskConfig:
    cfg = _apply_sid_defaults(_ddbm_sar2eo_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    cfg = _apply_sid_defaults(_ddbm_rgb2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    cfg = _apply_sid_defaults(_ddbm_sar2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    cfg = _apply_sid_defaults(_ddbm_sar2rgb_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
