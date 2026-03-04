# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Backward-compatibility shim for SID model imports."""

from src.models.unet.unet_sid import SiDUNet, create_sid_model  # noqa: F401

create_model = create_sid_model

__all__ = ["SiDUNet", "create_sid_model", "create_model"]
