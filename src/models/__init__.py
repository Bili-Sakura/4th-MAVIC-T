"""Model architectures for MAVIC-T baselines."""

from .unet_ddbm import (
    DDBMUNet,
    EDMUNet,
    EDM2UNet,
    VDMUNet,
    SiDUNet,
    create_model as create_ddbm_model,
    get_unet_type_config,
    SUPPORTED_UNET_TYPES,
    UNET_TYPE_ADM,
    UNET_TYPE_EDM,
    UNET_TYPE_EDM2,
    UNET_TYPE_VDM,
    UNET_TYPE_SID,
)
from .unet_dbim import (
    DBIMUNet,
    DBIMEDMUNet,
    DBIMEDM2UNet,
    DBIMVDMUNet,
    DBIMSiDUNet,
    create_dbim_model,
)
from .unet_bibbdm import (
    BiBBDMUNet,
    EDMBiBBDMUNet,
    EDM2BiBBDMUNet,
    VDMBiBBDMUNet,
    SiDBiBBDMUNet,
)
from .unet_bibbdm import create_model as create_bibbdm_model
from .unet_ddib import DDIBUNet
from .unet_ddib import create_model as create_ddib_model
from .unet_i2sb import (
    I2SBUNet,
    EDMI2SBUNet,
    EDM2I2SBUNet,
    VDMI2SBUNet,
    SiDI2SBUNet,
)
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
    "DDBMUNet", "EDMUNet", "EDM2UNet", "VDMUNet", "SiDUNet",
    "create_ddbm_model",
    "get_unet_type_config",
    "SUPPORTED_UNET_TYPES",
    "UNET_TYPE_ADM",
    "UNET_TYPE_EDM",
    "UNET_TYPE_EDM2",
    "UNET_TYPE_VDM",
    "UNET_TYPE_SID",
    "DBIMUNet", "DBIMEDMUNet", "DBIMEDM2UNet", "DBIMVDMUNet", "DBIMSiDUNet",
    "create_dbim_model",
    "BiBBDMUNet", "EDMBiBBDMUNet", "EDM2BiBBDMUNet", "VDMBiBBDMUNet", "SiDBiBBDMUNet",
    "create_bibbdm_model",
    "DDIBUNet", "create_ddib_model",
    "I2SBUNet", "EDMI2SBUNet", "EDM2I2SBUNet", "VDMI2SBUNet", "SiDI2SBUNet",
    "create_i2sb_model",
    "CUTGenerator", "PatchGANDiscriminator", "PatchSampleMLP",
    "GANLoss", "PatchNCELoss",
    "create_generator", "create_discriminator", "create_patch_sample_mlp",
    "Pix2PixTurbo",
    "CycleGANTurbo", "VAE_encode", "VAE_decode",
    "initialize_unet", "initialize_vae",
]