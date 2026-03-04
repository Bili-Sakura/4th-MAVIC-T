# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""UNet backbone models for MAVIC-T baselines.

Following the diffusers philosophy, backbone classes are method-agnostic.
The canonical classes live in :mod:`unet_2d`:

* :class:`UNet2DWrapper` — ADM-style UNet (replaces per-method wrappers).
* :class:`EDMUNet2D` — Fourier time embedding variant.
* :class:`VDMUNet2D` — logSNR normalization variant.
* :class:`EDM2UNet2D` — EDM2 preconditioning (disabled).

Method-specific type aliases (e.g. ``DDBMUNet``, ``BiBBDMUNet``) are
provided for backward compatibility but all resolve to the same generic
backbone class.
"""

from .unet_2d import (
    # Canonical backbone classes
    UNet2DWrapper,
    EDMUNet2D,
    EDM2UNet2D,
    VDMUNet2D,
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
    _raise_backbone_placeholder,
    _raise_unet_placeholder,  # backward-compat alias
)
from .unet_cdtsde import CDTSDEUNet
from .unet_unidb import UniDBConditionalUNet

# ---------------------------------------------------------------------------
# Backward-compat aliases: old per-method class names → generic backbones
# ---------------------------------------------------------------------------
DDBMUNet = UNet2DWrapper
DBIMUNet = UNet2DWrapper
I2SBUNet = UNet2DWrapper
BiBBDMUNet = UNet2DWrapper
SiDUNet = UNet2DWrapper
BDBMUNet = UNet2DWrapper
DABUNet = UNet2DWrapper
DDIBUNet = UNet2DWrapper

EDMUNet = EDMUNet2D
DBIMEDMUNet = EDMUNet2D
EDMBiBBDMUNet = EDMUNet2D
EDMI2SBUNet = EDMUNet2D

VDMUNet = VDMUNet2D
DBIMVDMUNet = VDMUNet2D
VDMBiBBDMUNet = VDMUNet2D
VDMI2SBUNet = VDMUNet2D

EDM2UNet = EDM2UNet2D
DBIMEDM2UNet = EDM2UNet2D
EDM2BiBBDMUNet = EDM2UNet2D
EDM2I2SBUNet = EDM2UNet2D

# Backward-compat factory aliases
create_ddbm_model = create_model
create_dbim_model = create_model

__all__ = [
    # Canonical classes
    "UNet2DWrapper", "EDMUNet2D", "EDM2UNet2D", "VDMUNet2D",
    # Unique architectures
    "CDTSDEUNet", "UniDBConditionalUNet",
    # Factory
    "create_model",
    # Type constants
    "get_backbone_config", "get_unet_type_config",
    "SUPPORTED_UNET_TYPES", "SUPPORTED_DIT_TYPES", "SUPPORTED_BACKBONE_TYPES",
    "UNET_TYPE_ADM", "UNET_TYPE_EDM", "UNET_TYPE_EDM2", "UNET_TYPE_VDM",
    "DIT_TYPE_PIXNERD", "DIT_TYPE_PIXELDIT", "DIT_TYPE_SIT",
    # Shared utilities
    "_build_block_types", "_channel_mult_for_resolution",
    "_parse_layers_per_block", "_parse_create_model_args",
    "_raise_backbone_placeholder", "_raise_unet_placeholder",
    # Backward-compat aliases
    "DDBMUNet", "DBIMUNet", "I2SBUNet", "BiBBDMUNet", "SiDUNet",
    "BDBMUNet", "DABUNet", "DDIBUNet",
    "EDMUNet", "DBIMEDMUNet", "EDMBiBBDMUNet", "EDMI2SBUNet",
    "VDMUNet", "DBIMVDMUNet", "VDMBiBBDMUNet", "VDMI2SBUNet",
    "EDM2UNet", "DBIMEDM2UNet", "EDM2BiBBDMUNet", "EDM2I2SBUNet",
    "create_ddbm_model", "create_dbim_model",
]
