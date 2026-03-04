# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""CDTSDE model factory.

Provides :func:`create_cdtsde_model` (and backward-compatible alias
:func:`create_model`) which builds a CDTSDEUNet with lambda field.

The :class:`CDTSDEUNet` class lives in :mod:`src.models.unet.unet_cdtsde`
(unique architecture with spatial lambda field).  Shared parsing utilities
come from :mod:`src.models.unet.unet_2d`.
"""

from __future__ import annotations

from typing import Optional

from src.models.unet.unet_cdtsde import CDTSDEUNet
from src.models.unet.unet_2d import _parse_create_model_args


def create_cdtsde_model(
    image_size: int = 256,
    in_channels: int = 3,
    num_channels: int = 128,
    num_res_blocks: int = 2,
    attention_resolutions: str = "32,16,8",
    dropout: float = 0.0,
    condition_mode: Optional[str] = "concat",
    channel_mult: str = "",
    lambda_hidden_channels: int = 16,
) -> CDTSDEUNet:
    """Factory for CDTSDE UNet model."""
    attn_indices, cm_tuple = _parse_create_model_args(
        image_size=image_size,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
    )
    return CDTSDEUNet(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=cm_tuple,
        lambda_hidden_channels=lambda_hidden_channels,
    )


# Backward-compat alias
create_model = create_cdtsde_model

__all__ = ["CDTSDEUNet", "create_cdtsde_model", "create_model"]

