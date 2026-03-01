# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Task-specific configurations for standalone SiD2 baseline training."""

from __future__ import annotations

from examples.ddbm.config import (
    TaskConfig,
    rgb2ir_config as _ddbm_rgb2ir_config,
    sar2eo_config as _ddbm_sar2eo_config,
    sar2ir_config as _ddbm_sar2ir_config,
    sar2rgb_config as _ddbm_sar2rgb_config,
)


def _default_sigmoid_bias(resolution: int) -> float:
    """
    Resolution-aware bias from SiD2 appendix defaults (A.4).

    The paper reports:
      - 128²: b = 0
      - 256²: b = -1
      - 512²: b = -3
    and large-resolution sweeps suggest ~-4 for 1024².
    """
    if resolution >= 768:
        return -4.0
    if resolution >= 384:
        return -3.0
    if resolution >= 192:
        return -1.0
    return 0.0


def _apply_sid2_defaults(cfg: TaskConfig) -> TaskConfig:
    """Convert a DDBM task config into a standalone SiD2 baseline config."""
    cfg.unet_type = "sid"
    cfg.use_multiscale_loss = False
    cfg.multiscale_base_resolution = 32

    # SiD2 scheduler/objective defaults.
    setattr(cfg, "prediction_type", "v")
    setattr(cfg, "logsnr_min", -15.0)
    setattr(cfg, "logsnr_max", 15.0)
    setattr(cfg, "schedule_type", "cosine_interpolated")
    setattr(cfg, "noise_d", 64.0)  # used when schedule_type='shifted_cosine'
    setattr(cfg, "clip_sample", True)
    setattr(cfg, "num_train_timesteps", 1000)

    # SiD2 sigmoid-weighted x-space objective.
    setattr(cfg, "sid2_sigmoid_bias", _default_sigmoid_bias(int(cfg.resolution)))
    setattr(cfg, "sid2_include_dlogsnr", True)

    # Paper appendix uses cosine_interpolated_low_32_high_512 at 512².
    # We generalize by resolution while preserving the same ratio.
    low = max(1.0, float(cfg.resolution) / 16.0)
    high = float(cfg.resolution)
    setattr(cfg, "interpolated_noise_d_low", low)
    setattr(cfg, "interpolated_noise_d_high", high)
    return cfg


def sar2eo_config(**overrides) -> TaskConfig:
    cfg = _apply_sid2_defaults(_ddbm_sar2eo_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    cfg = _apply_sid2_defaults(_ddbm_rgb2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    cfg = _apply_sid2_defaults(_ddbm_sar2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    cfg = _apply_sid2_defaults(_ddbm_sar2rgb_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
