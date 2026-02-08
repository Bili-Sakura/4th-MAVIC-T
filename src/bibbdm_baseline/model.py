"""Backward-compatibility shim — imports from ``models.bibbdm_unet``.

New code should import directly from :mod:`src.bibbdm_baseline.models`.
"""

from .models.bibbdm_unet import BiBBDMUNet, create_model, _out_channels_for_objective  # noqa: F401

__all__ = ["BiBBDMUNet", "create_model", "_out_channels_for_objective"]
