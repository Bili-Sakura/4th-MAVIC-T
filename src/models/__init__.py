# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Model architectures for MAVIC-T baselines."""

from .unet import (
    # Canonical backbone classes
    UNet2DWrapper,
    EDMUNet2D,
    EDM2UNet2D,
    VDMUNet2D,
    # Unique architectures
    CDTSDEUNet,
    UniDBConditionalUNet,
    # Generic factory
    create_model,
    # Type constants
    get_backbone_config,
    get_unet_type_config,  # backward-compat alias
    SUPPORTED_UNET_TYPES,
    SUPPORTED_DIT_TYPES,
    SUPPORTED_BACKBONE_TYPES,
    UNET_TYPE_ADM,
    UNET_TYPE_EDM,
    UNET_TYPE_EDM2,
    UNET_TYPE_VDM,
    DIT_TYPE_PIXNERD,
    DIT_TYPE_PIXELDIT,
    DIT_TYPE_SIT,
    # Shared utilities
    _build_block_types,
    _channel_mult_for_resolution,
    _parse_layers_per_block,
    _parse_create_model_args,
    # Backward-compat aliases
    DDBMUNet,
    DBIMUNet,
    I2SBUNet,
    BiBBDMUNet,
    SiDUNet,
    BDBMUNet,
    DABUNet,
    DDIBUNet,
    EDMUNet,
    DBIMEDMUNet,
    EDMBiBBDMUNet,
    EDMI2SBUNet,
    VDMUNet,
    DBIMVDMUNet,
    VDMBiBBDMUNet,
    VDMI2SBUNet,
    EDM2UNet,
    DBIMEDM2UNet,
    EDM2BiBBDMUNet,
    EDM2I2SBUNet,
    create_ddbm_model,
    create_dbim_model,
)
from .dit import (
    PixNerdBackbone,
    PixelDiTBackbone,
    SiTBackbone,
)
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
    # Canonical backbone classes
    "UNet2DWrapper", "EDMUNet2D", "EDM2UNet2D", "VDMUNet2D",
    # Unique architectures
    "CDTSDEUNet", "UniDBConditionalUNet",
    # DiT backbones
    "PixNerdBackbone", "PixelDiTBackbone", "SiTBackbone",
    # Factory
    "create_model", "create_ddbm_model", "create_dbim_model",
    # Type constants
    "get_backbone_config", "get_unet_type_config",
    "SUPPORTED_UNET_TYPES", "SUPPORTED_DIT_TYPES", "SUPPORTED_BACKBONE_TYPES",
    "UNET_TYPE_ADM", "UNET_TYPE_EDM", "UNET_TYPE_EDM2", "UNET_TYPE_VDM",
    "DIT_TYPE_PIXNERD", "DIT_TYPE_PIXELDIT", "DIT_TYPE_SIT",
    # Shared utilities
    "_build_block_types", "_channel_mult_for_resolution",
    "_parse_layers_per_block", "_parse_create_model_args",
    # Backward-compat aliases
    "DDBMUNet", "DBIMUNet", "I2SBUNet", "BiBBDMUNet", "SiDUNet",
    "BDBMUNet", "DABUNet", "DDIBUNet",
    "EDMUNet", "DBIMEDMUNet", "EDMBiBBDMUNet", "EDMI2SBUNet",
    "VDMUNet", "DBIMVDMUNet", "VDMBiBBDMUNet", "VDMI2SBUNet",
    "EDM2UNet", "DBIMEDM2UNet", "EDM2BiBBDMUNet", "EDM2I2SBUNet",
    # GAN models
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
