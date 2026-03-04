# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Model architectures for MAVIC-T baselines."""

from .unet_ddbm import (
    DDBMUNet,
    EDMUNet,
    EDM2UNet,
    VDMUNet,
    create_model as create_ddbm_model,
    get_unet_type_config,
    SUPPORTED_UNET_TYPES,
    UNET_TYPE_ADM,
    UNET_TYPE_EDM,
    UNET_TYPE_EDM2,
    UNET_TYPE_VDM,
    DIT_TYPE_PIXNERD,
    UNET_TYPE_PIXNERD,
    UNET_TYPE_PIXELDIT,
)
from .pixnerd_backbone import PixNerdBackbone
from .pixeldit_backbone import PixelDiTBackbone
    UNET_TYPE_SIT,
)
from .pixnerd_backbone import PixNerdBackbone
from .sit_backbone import SiTBackbone
from .unet_sid import SiDUNet, create_sid_model
from .unet_dbim import (
    DBIMUNet,
    DBIMEDMUNet,
    DBIMEDM2UNet,
    DBIMVDMUNet,
    create_dbim_model,
)
from .unet_bibbdm import (
    BiBBDMUNet,
    EDMBiBBDMUNet,
    EDM2BiBBDMUNet,
    VDMBiBBDMUNet,
)
from .unet_bibbdm import create_model as create_bibbdm_model
from .unet_bdbm import BDBMUNet
from .unet_bdbm import create_model as create_bdbm_model
from .unet_ddib import DDIBUNet
from .unet_ddib import create_model as create_ddib_model
from .unet_i2sb import (
    I2SBUNet,
    EDMI2SBUNet,
    EDM2I2SBUNet,
    VDMI2SBUNet,
)
from .unet_i2sb import create_model as create_i2sb_model
from .unet_cdtsde import CDTSDEUNet
from .unet_cdtsde import create_model as create_cdtsde_model
from .unet_dab import DABUNet
from .unet_dab import create_model as create_dab_model
from .unet_unidb import UniDBConditionalUNet
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
from .stegogan_model import (
    StegoGANGeneratorA,
    StegoGANGeneratorB,
    StegoGANDiscriminator,
    StegoGANLoss,
    create_generator_a as create_stegogan_generator_a,
    create_generator_b as create_stegogan_generator_b,
    create_discriminator as create_stegogan_discriminator,
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
    "DDBMUNet", "EDMUNet", "EDM2UNet", "VDMUNet",
    "PixNerdBackbone",
    "PixelDiTBackbone",
    "SiTBackbone",
    "create_ddbm_model",
    "SiDUNet", "create_sid_model",
    "get_unet_type_config",
    "SUPPORTED_UNET_TYPES",
    "UNET_TYPE_ADM",
    "UNET_TYPE_EDM",
    "UNET_TYPE_EDM2",
    "UNET_TYPE_VDM",
    "DIT_TYPE_PIXNERD",
    "UNET_TYPE_PIXNERD",
    "UNET_TYPE_PIXELDIT",
    "UNET_TYPE_SIT",
    "DBIMUNet", "DBIMEDMUNet", "DBIMEDM2UNet", "DBIMVDMUNet",
    "create_dbim_model",
    "BiBBDMUNet", "EDMBiBBDMUNet", "EDM2BiBBDMUNet", "VDMBiBBDMUNet",
    "create_bibbdm_model",
    "BDBMUNet", "create_bdbm_model",
    "DDIBUNet", "create_ddib_model",
    "I2SBUNet", "EDMI2SBUNet", "EDM2I2SBUNet", "VDMI2SBUNet",
    "create_i2sb_model",
    "CDTSDEUNet", "create_cdtsde_model",
    "DABUNet", "create_dab_model",
    "UniDBConditionalUNet",
    "CUTGenerator", "PatchGANDiscriminator", "PatchSampleMLP",
    "GANLoss", "PatchNCELoss",
    "create_generator", "create_discriminator", "create_patch_sample_mlp",
    "StegoGANGeneratorA", "StegoGANGeneratorB", "StegoGANDiscriminator",
    "StegoGANLoss",
    "create_stegogan_generator_a", "create_stegogan_generator_b",
    "create_stegogan_discriminator",
    "Pix2PixTurbo",
    "CycleGANTurbo", "VAE_encode", "VAE_decode",
    "initialize_unet", "initialize_vae",
]
