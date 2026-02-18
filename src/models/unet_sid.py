"""Standalone Simple Diffusion (SiD) UNet."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .unet_ddbm import (
    _build_block_types,
    _channel_mult_for_resolution,
    _parse_create_model_args,
    _parse_layers_per_block,
)


class SiDUNet(ModelMixin, ConfigMixin):
    """Simple Diffusion UNet using ``UNet2DModel`` with concat conditioning."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        condition_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: Union[int, Tuple[int, ...]] = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
        attention_head_dim: Optional[int] = 64,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if condition_channels is None:
            condition_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_channels = condition_channels
        self.condition_mode = condition_mode

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        unet_in_channels = (
            in_channels + condition_channels if condition_mode == "concat" else in_channels
        )
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        layers_per_block = _parse_layers_per_block(
            num_res_blocks,
            num_levels=len(channel_mult),
            allow_variable=True,
        )

        unet_kwargs: dict = dict(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=layers_per_block,
            dropout=dropout,
            mid_block_type="UNetMidBlock2D",
        )
        if attention_head_dim is not None:
            unet_kwargs["attention_head_dim"] = attention_head_dim
        self.unet = UNet2DModel(**unet_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)
        return self.unet(x, timestep).sample


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
    return SiDUNet(
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
