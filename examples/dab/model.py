# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DAB model factory.

Provides :func:`create_dab_model` (and backward-compatible alias
:func:`create_model`) which builds a UNet backbone configured for DAB.

Imports the generic backbone classes from :mod:`src.models.unet.unet_2d`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch.nn as nn

from src.models.unet.unet_2d import (
    UNet2DWrapper,
    _channel_mult_for_resolution,
    _parse_create_model_args,
    UNET_TYPE_ADM,
)

# Backward-compat alias
DABUNet = UNet2DWrapper

SUPPORTED_UNET_TYPES = (UNET_TYPE_ADM,)


def create_dab_model(
    image_size: int = 256,
    in_channels: int = 3,
    num_channels: int = 128,
    num_res_blocks: int = 2,
    attention_resolutions: str = "32,16,8",
    dropout: float = 0.0,
    condition_mode: Optional[str] = "concat",
    channel_mult: str = "",
    objective: str = "grad",
    backbone_type: str = UNET_TYPE_ADM,
    conditioning_channels: Optional[int] = None,
    **_: Any,
) -> nn.Module:
    """Factory for DAB-compatible UNet models."""
    if backbone_type not in SUPPORTED_UNET_TYPES:
        raise ValueError(
            f"backbone_type '{backbone_type}' not supported. Use one of: {SUPPORTED_UNET_TYPES}"
        )

    attn_indices, cm_tuple = _parse_create_model_args(
        image_size,
        attention_resolutions,
        channel_mult,
    )
    return UNet2DWrapper(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=in_channels,
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=cm_tuple,
        conditioning_channels=conditioning_channels,
    )


# Backward-compat alias
create_model = create_dab_model

__all__ = [
    "DABUNet",
    "create_dab_model",
    "create_model",
]
