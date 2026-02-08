"""I2SB model components.

Provides the ``I2SBUNet`` wrapper around ``diffusers.UNet2DModel``
and the ``create_model`` factory function.
"""

from .i2sb_unet import I2SBUNet, create_model

__all__ = ["I2SBUNet", "create_model"]
