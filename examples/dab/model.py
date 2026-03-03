# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Backward-compatibility shim for DAB model imports."""

from src.models.unet_dab import DABUNet, create_model  # noqa: F401

__all__ = ["DABUNet", "create_model"]
