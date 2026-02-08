"""Model architectures for MAVIC-T baselines."""

from .unet_ddbm import DDBMUNet
from .unet_ddbm import create_model as create_ddbm_model
from .unet_bibbdm import BiBBDMUNet
from .unet_bibbdm import create_model as create_bibbdm_model
from .unet_ddib import DDIBUNet
from .unet_ddib import create_model as create_ddib_model
from .unet_i2sb import I2SBUNet
from .unet_i2sb import create_model as create_i2sb_model
from .cut_model import (
    CUTGenerator,
    PatchGANDiscriminator,
    PatchSampleMLP,
    GANLoss,
    PatchNCELoss,
    create_generator,
    create_discriminator,
    create_patch_sample_mlp,
)
from .pix2pix_turbo import Pix2PixTurbo
from .cyclegan_turbo import (
    CycleGANTurbo,
    VAE_encode,
    VAE_decode,
    initialize_unet,
    initialize_vae,
)

__all__ = [
    "DDBMUNet", "create_ddbm_model",
    "BiBBDMUNet", "create_bibbdm_model",
    "DDIBUNet", "create_ddib_model",
    "I2SBUNet", "create_i2sb_model",
    "CUTGenerator", "PatchGANDiscriminator", "PatchSampleMLP",
    "GANLoss", "PatchNCELoss",
    "create_generator", "create_discriminator", "create_patch_sample_mlp",
    "Pix2PixTurbo",
    "CycleGANTurbo", "VAE_encode", "VAE_decode",
    "initialize_unet", "initialize_vae",
]
