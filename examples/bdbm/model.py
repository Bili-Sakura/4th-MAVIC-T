"""Backward-compatibility shim for BDBM model imports."""

from src.models.unet_bdbm import BDBMUNet, create_model, _out_channels_for_objective  # noqa: F401

__all__ = ["BDBMUNet", "create_model", "_out_channels_for_objective"]
