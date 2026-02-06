"""DDBM-compatible UNet model built on ``diffusers.UNet2DModel``.

The vendor DDBM UNet accepts ``(x, timestep, xT=…)`` where ``xT`` is the
source/condition image.  With ``condition_mode='concat'`` the model
internally concatenates ``x`` and ``xT`` along the channel axis.

This module replicates that contract using a standard ``UNet2DModel`` from
the Hugging Face *diffusers* library.  A thin wrapper class
:class:`DDBMUNet` concatenates source and noisy sample before forwarding to
the underlying ``UNet2DModel``, so the rest of the training / sampling code
can call ``model(x, t, xT=source)`` just like the vendor code.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import UNet2DModel


def _channel_mult_for_resolution(resolution: int) -> Tuple[int, ...]:
    """Return a sensible default channel multiplier tuple."""
    return {
        512: (1, 1, 2, 2, 4, 4),
        256: (1, 1, 2, 2, 4, 4),
        128: (1, 1, 2, 3, 4),
        64:  (1, 2, 3, 4),
        32:  (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


class DDBMUNet(nn.Module):
    """Wrapper around ``UNet2DModel`` that accepts the DDBM calling convention.

    Parameters
    ----------
    image_size : int
        Spatial resolution (height == width).
    in_channels : int
        Number of channels of the *target* image (and of the noisy sample).
        When ``condition_mode='concat'``, the underlying UNet receives
        ``2 * in_channels`` input channels.
    model_channels : int
        Base channel count of the UNet.
    num_res_blocks : int
        Residual blocks per resolution level.
    attention_resolutions : tuple of int
        Down-block indices where attention is applied (0-indexed).
    dropout : float
        Dropout probability.
    condition_mode : str or None
        ``'concat'`` to concatenate source image along channels, or ``None``
        for unconditional mode.
    channel_mult : tuple of int or None
        Per-level channel multipliers. Auto-detected if ``None``.
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels

        # Build block_out_channels from model_channels and channel_mult
        block_out_channels = tuple(model_channels * m for m in channel_mult)

        # Convert attention_resolutions to down_block indices
        down_block_types = []
        for i in range(len(channel_mult)):
            if i in attention_resolutions:
                down_block_types.append("AttnDownBlock2D")
            else:
                down_block_types.append("DownBlock2D")

        up_block_types = []
        for i in range(len(channel_mult)):
            if (len(channel_mult) - 1 - i) in attention_resolutions:
                up_block_types.append("AttnUpBlock2D")
            else:
                up_block_types.append("UpBlock2D")

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass matching vendor DDBM UNet calling convention.

        Parameters
        ----------
        x : Tensor  (B, C, H, W)
            Pre-conditioned noisy sample (``c_in * noisy``).
        timestep : Tensor  (B,)
            Rescaled log-sigma timestep.
        xT : Tensor or None  (B, C, H, W)
            Source/condition image.

        Returns
        -------
        Tensor  (B, C, H, W)
            Raw model output (before ``c_out / c_skip`` application).
        """
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)
        return self.unet(x, timestep).sample


def create_model(
    image_size: int = 256,
    in_channels: int = 3,
    num_channels: int = 128,
    num_res_blocks: int = 2,
    attention_resolutions: str = "32,16,8",
    dropout: float = 0.0,
    condition_mode: Optional[str] = "concat",
    channel_mult: str = "",
    **kwargs,
) -> DDBMUNet:
    """Factory matching the vendor ``create_model`` signature.

    Parses string-based arguments (``attention_resolutions``, ``channel_mult``)
    into the tuples that :class:`DDBMUNet` expects.
    """
    # Parse attention_resolutions → down-block indices
    attn_indices: Tuple[int, ...] = ()
    if attention_resolutions:
        if isinstance(attention_resolutions, str):
            attn_res_list = [int(r) for r in attention_resolutions.split(",")]
        else:
            attn_res_list = list(attention_resolutions)
        # Convert absolute resolutions to down-block indices (0-indexed)
        attn_indices = tuple(
            i for i, _ in enumerate(
                _channel_mult_for_resolution(image_size) if not channel_mult else range(100)
            )
        )
        # Actually, map resolution to block index: block i has resolution image_size / 2^i
        attn_indices = []
        cm = None
        if channel_mult and isinstance(channel_mult, str) and channel_mult != "":
            cm = tuple(int(c) for c in channel_mult.split(","))
        elif channel_mult and isinstance(channel_mult, tuple):
            cm = channel_mult
        else:
            cm = _channel_mult_for_resolution(image_size)

        for i in range(len(cm)):
            block_res = image_size // (2 ** i)
            if block_res in attn_res_list:
                attn_indices.append(i)
        attn_indices = tuple(attn_indices)

    # Parse channel_mult
    cm_tuple: Optional[Tuple[int, ...]] = None
    if channel_mult and isinstance(channel_mult, str) and channel_mult != "":
        cm_tuple = tuple(int(c) for c in channel_mult.split(","))
    elif isinstance(channel_mult, tuple) and channel_mult:
        cm_tuple = channel_mult

    return DDBMUNet(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=cm_tuple,
    )
