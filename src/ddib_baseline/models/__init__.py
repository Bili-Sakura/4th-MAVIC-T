"""DDIB model components.

Provides the ``DDIBUNet`` wrapper around ``diffusers.UNet2DModel``
and the ``create_model`` factory function.
"""

from .unet import DDIBUNet, create_model

__all__ = ["DDIBUNet", "create_model"]
