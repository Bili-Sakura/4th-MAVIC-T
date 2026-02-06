"""Backward-compatibility shim — imports from ``models.unet``.

New code should import directly from :mod:`src.ddbm_baseline.models`.
"""

from .models.unet import DDBMUNet, create_model, _channel_mult_for_resolution  # noqa: F401

__all__ = ["DDBMUNet", "create_model", "_channel_mult_for_resolution"]
