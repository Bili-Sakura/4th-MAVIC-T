"""BiBBDM-compatible UNet model built on ``diffusers.UNet2DModel``.

The BiBBDM UNet accepts ``(x_t, timesteps, context=…)`` where ``x_t`` is the
noisy sample and ``context`` is an optional conditioning signal.  With
``condition_mode='concat'`` the model internally concatenates ``x_t`` and
``context`` along the channel axis.

For dual-learning objectives (``dlns``, ``dlab``, ``dlgab``), the model
outputs ``2 * in_channels`` to predict both components simultaneously.

Supported UNet types (via ``unet_type`` in :func:`create_model`):
- ``adm``: ADM-style diffusers UNet2DModel (default, implemented).
- ``edm``, ``edm2``, ``vdm``, ``sid``: placeholders (see :mod:`src.models.unet_ddbm`).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .unet_ddbm import (
    SUPPORTED_UNET_TYPES,
    UNET_TYPE_ADM,
    _raise_unet_placeholder,
)


def _channel_mult_for_resolution(resolution: int) -> Tuple[int, ...]:
    """Return a sensible default channel multiplier tuple."""
    return {
        512: (1, 1, 2, 2, 4, 4),
        256: (1, 1, 2, 2, 4, 4),
        128: (1, 1, 2, 3, 4),
        64:  (1, 2, 3, 4),
        32:  (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


class BiBBDMUNet(ModelMixin, ConfigMixin):
    """Wrapper around ``UNet2DModel`` for BiBBDM.

    Inherits from :class:`~diffusers.ModelMixin` and
    :class:`~diffusers.ConfigMixin` so that instances can be persisted and
    restored with ``save_pretrained`` / ``from_pretrained``.

    Parameters
    ----------
    image_size : int
        Spatial resolution (height == width).
    in_channels : int
        Number of channels of the noisy sample.
    out_channels : int or None
        Number of output channels.  ``None`` defaults to ``in_channels``.
        For dual-learning objectives set to ``2 * in_channels``.
    model_channels : int
        Base channel count of the UNet.
    num_res_blocks : int
        Residual blocks per resolution level.
    attention_resolutions : tuple of int
        Down-block indices where attention is applied (0-indexed).
    dropout : float
        Dropout probability.
    condition_mode : str or None
        ``'concat'`` to concatenate context along channels, or ``None``
        for unconditional mode.
    channel_mult : tuple of int or None
        Per-level channel multipliers.  Auto-detected if ``None``.
    """

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
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
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels

        block_out_channels = tuple(model_channels * m for m in channel_mult)

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
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass matching BiBBDM UNet calling convention.

        Parameters
        ----------
        x_t : Tensor  (B, C, H, W)
            Noisy sample at timestep *t*.
        timesteps : Tensor  (B,)
            Integer timestep indices.
        context : Tensor or None  (B, C, H, W)
            Optional conditioning signal (e.g. source image).

        Returns
        -------
        Tensor  (B, out_channels, H, W)
            Predicted objective reconstruction.
        """
        if self.condition_mode == "concat" and context is not None:
            x_t = torch.cat([x_t, context], dim=1)
        return self.unet(x_t, timesteps).sample


def _out_channels_for_objective(objective: str, in_channels: int) -> int:
    """Return the UNet output channels for the given BiBBDM objective."""
    if objective in ("dlns", "dlab", "dlgab"):
        return 2 * in_channels
    return in_channels


def create_model(
    image_size: int = 256,
    in_channels: int = 3,
    num_channels: int = 128,
    num_res_blocks: int = 2,
    attention_resolutions: str = "32,16,8",
    dropout: float = 0.0,
    condition_mode: Optional[str] = "concat",
    channel_mult: str = "",
    objective: str = "dlns",
    unet_type: str = UNET_TYPE_ADM,
    **kwargs: Any,
) -> Union[BiBBDMUNet, nn.Module]:
    """Factory for :class:`BiBBDMUNet`.

    Parses string-based arguments (``attention_resolutions``, ``channel_mult``)
    into tuples and infers ``out_channels`` from the *objective*.

    Parameters
    ----------
    unet_type : str
        Backbone architecture. One of: ``adm`` (default), ``edm``, ``edm2``,
        ``vdm``, ``sid``. Only ``adm`` is implemented; others raise
        :exc:`NotImplementedError`.
    """
    if unet_type not in SUPPORTED_UNET_TYPES:
        raise ValueError(
            f"unet_type '{unet_type}' not supported. Use one of: {SUPPORTED_UNET_TYPES}"
        )
    if unet_type != UNET_TYPE_ADM:
        _raise_unet_placeholder("BiBBDM", unet_type)

    out_channels = _out_channels_for_objective(objective, in_channels)

    # Parse attention_resolutions → down-block indices
    attn_indices: Tuple[int, ...] = ()
    if attention_resolutions:
        if isinstance(attention_resolutions, str):
            attn_res_list = [int(r) for r in attention_resolutions.split(",")]
        else:
            attn_res_list = list(attention_resolutions)

        cm = None
        if channel_mult and isinstance(channel_mult, str) and channel_mult != "":
            cm = tuple(int(c) for c in channel_mult.split(","))
        elif channel_mult and isinstance(channel_mult, tuple):
            cm = channel_mult
        else:
            cm = _channel_mult_for_resolution(image_size)

        attn_indices = tuple(
            i for i in range(len(cm))
            if image_size // (2 ** i) in attn_res_list
        )

    # Parse channel_mult
    cm_tuple: Optional[Tuple[int, ...]] = None
    if channel_mult and isinstance(channel_mult, str) and channel_mult != "":
        cm_tuple = tuple(int(c) for c in channel_mult.split(","))
    elif isinstance(channel_mult, tuple) and channel_mult:
        cm_tuple = channel_mult

    return BiBBDMUNet(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=cm_tuple,
    )
