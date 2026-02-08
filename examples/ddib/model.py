"""Backward-compatibility shim — imports from ``models.unet``.

New code should import directly from :mod:`src.models`.
"""

from src.models.unet_ddib import DDIBUNet, create_model, _channel_mult_for_resolution  # noqa: F401

__all__ = ["DDIBUNet", "create_model", "_channel_mult_for_resolution"]
