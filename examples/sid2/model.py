"""Backward-compatibility shim for SID2 model imports."""

from src.models.unet_sid import SiDUNet, create_sid_model  # noqa: F401

create_model = create_sid_model

__all__ = ["SiDUNet", "create_sid_model", "create_model"]
