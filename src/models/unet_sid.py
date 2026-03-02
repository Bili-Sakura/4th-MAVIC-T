# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Simple Diffusion (SiD) model using the native diffusers ``UNet2DModel``.

The concat-conditioning contract (``model(cat([x, xT], dim=1), t)``) is
handled by callers (trainer / pipeline) so that the model itself is a plain
``UNet2DModel`` with no custom wrapper.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from diffusers import UNet2DModel

from .unet_ddbm import (
    _build_block_types,
    _channel_mult_for_resolution,
    _parse_create_model_args,
    _parse_layers_per_block,
)

# Backward-compatibility alias: SiDUNet is the native diffusers UNet2DModel.
SiDUNet = UNet2DModel


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
) -> UNet2DModel:
    """Create a native diffusers ``UNet2DModel`` for SiD conditional translation.

    When ``condition_mode='concat'`` (default) the returned model expects the
    noisy sample and the condition image to be pre-concatenated along the
    channel axis before being passed to the model::

        model_input = torch.cat([noisy_sample, condition], dim=1)
        pred = model(model_input, timestep).sample

    Parameters
    ----------
    image_size : int
        Spatial resolution (height == width).
    in_channels : int
        Number of channels of the *target* (noisy sample) image.
    out_channels : int or None
        Output channels. Defaults to ``in_channels``.
    condition_channels : int or None
        Channels of the condition image. Defaults to ``in_channels``.
    num_channels : int
        Base channel count of the UNet.
    num_res_blocks : int, str, or tuple of int
        Residual blocks per resolution level.
    attention_resolutions : str
        Comma-separated spatial resolutions at which attention is applied.
    dropout : float
        Dropout probability.
    condition_mode : str or None
        ``'concat'`` to include condition channels in the model's input
        (default), or ``None`` for an unconditional model.
    channel_mult : str
        Comma-separated per-level channel multipliers.
    attention_head_dim : int or None
        Dimension per attention head.
    """
    if out_channels is None:
        out_channels = in_channels
    if condition_channels is None:
        condition_channels = in_channels

    attn_indices, cm_tuple = _parse_create_model_args(
        image_size, attention_resolutions, channel_mult
    )
    cm_effective = cm_tuple if cm_tuple is not None else _channel_mult_for_resolution(image_size)
    parsed_num_res_blocks = _parse_layers_per_block(
        num_res_blocks,
        num_levels=len(cm_effective),
        allow_variable=True,
    )

    unet_in_channels = (
        in_channels + condition_channels if condition_mode == "concat" else in_channels
    )
    block_out_channels = tuple(num_channels * m for m in cm_effective)
    down_block_types, up_block_types = _build_block_types(cm_effective, attn_indices)

    unet_kwargs: dict = dict(
        sample_size=image_size,
        in_channels=unet_in_channels,
        out_channels=out_channels,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        layers_per_block=parsed_num_res_blocks,
        dropout=dropout,
        mid_block_type="UNetMidBlock2D",
    )
    if attention_head_dim is not None:
        unet_kwargs["attention_head_dim"] = attention_head_dim
    return UNet2DModel(**unet_kwargs)
