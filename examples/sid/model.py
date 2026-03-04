# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""SID model factory.

Provides :func:`create_sid_model` (and backward-compatible alias
:func:`create_model`) which builds a UNet backbone for Simple Diffusion.

Imports the generic backbone classes from :mod:`src.models.unet.unet_2d`.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch.nn as nn

from src.models.unet.unet_2d import (
    UNet2DWrapper,
    _channel_mult_for_resolution,
    _parse_create_model_args,
    _parse_layers_per_block,
)

# Backward-compat alias
SiDUNet = UNet2DWrapper


def create_sid_model(
    image_size: int = 256,
    in_channels: int = 3,
    out_channels: Optional[int] = None,
    condition_channels: Optional[int] = None,
    num_channels: int = 128,
    num_res_blocks: Union[int, str, Tuple[int, ...]] = 2,
    attention_resolutions: str = "64,32",
    dropout: float = 0.0,
    condition_mode: Optional[str] = "concat",
    channel_mult: str = "",
    attention_head_dim: Optional[int] = 64,
) -> nn.Module:
    """Factory for standalone SID UNet."""
    attn_indices, cm_tuple = _parse_create_model_args(
        image_size, attention_resolutions, channel_mult
    )
    cm_effective = cm_tuple if cm_tuple is not None else _channel_mult_for_resolution(image_size)
    parsed_num_res_blocks = _parse_layers_per_block(
        num_res_blocks,
        num_levels=len(cm_effective),
        allow_variable=True,
    )
    return UNet2DWrapper(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        condition_channels=condition_channels,
        model_channels=num_channels,
        num_res_blocks=parsed_num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=cm_tuple,
        attention_head_dim=attention_head_dim,
    )


# Backward-compat alias
create_model = create_sid_model

__all__ = ["SiDUNet", "create_sid_model", "create_model"]
