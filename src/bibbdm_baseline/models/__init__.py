"""BiBBDM model components.

Provides the ``BiBBDMUNet`` wrapper around ``diffusers.UNet2DModel``
and the ``create_model`` factory function.
"""

from .bibbdm_unet import BiBBDMUNet, create_model

__all__ = ["BiBBDMUNet", "create_model"]
