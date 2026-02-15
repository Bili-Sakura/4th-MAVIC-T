"""Backward-compatibility shim for DBIM model imports."""

from src.models.unet_dbim import DBIMUNet, create_dbim_model  # noqa: F401

create_model = create_dbim_model

__all__ = ["DBIMUNet", "create_dbim_model", "create_model"]
