# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DDIB model factory.

Provides :func:`create_ddib_model` (and backward-compatible alias
:func:`create_model`) which builds an unconditional UNet backbone for DDIB.

DDIB uses different default channel multipliers than the generic backbone
(e.g. 256px → (1,1,2,2,4,4) instead of (1,2,2,4)).

Imports the generic backbone class from :mod:`src.models.unet.unet_2d`.
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.models.unet.unet_2d import UNet2DWrapper

# Backward-compat alias
DDIBUNet = UNet2DWrapper


def _channel_mult_for_resolution(resolution: int) -> Tuple[int, ...]:
    """Return a sensible default channel multiplier tuple for DDIB.

    DDIB uses deeper architectures than the generic backbone defaults.
    """
    return {
        512: (1, 1, 2, 2, 4, 4),
        256: (1, 1, 2, 2, 4, 4),
        128: (1, 1, 2, 3, 4),
        64:  (1, 2, 3, 4),
        32:  (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


def create_ddib_model(
    image_size: int = 256,
    in_channels: int = 3,
    num_channels: int = 128,
    num_res_blocks: int = 2,
    attention_resolutions: str = "32,16,8",
    dropout: float = 0.0,
    learn_sigma: bool = False,
    channel_mult: str = "",
    **kwargs,
) -> UNet2DWrapper:
    """Factory for DDIB unconditional UNet models.

    Parses string-based arguments (``attention_resolutions``, ``channel_mult``)
    into the tuples that :class:`UNet2DWrapper` expects.
    """
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

    out_channels = in_channels * 2 if learn_sigma else in_channels

    return UNet2DWrapper(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=None,
        channel_mult=cm_tuple,
    )


# Backward-compat alias
create_model = create_ddib_model

__all__ = [
    "DDIBUNet",
    "create_ddib_model",
    "create_model",
    "_channel_mult_for_resolution",
]
