# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""BiBBDM model factory.

Provides :func:`create_bibbdm_model` (and backward-compatible alias
:func:`create_model`) which builds a UNet backbone configured for the
BiBBDM dual-learning objectives.

Imports the generic backbone classes from :mod:`src.models.unet.unet_2d`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch.nn as nn

from src.models.unet.unet_2d import (
    UNet2DWrapper,
    EDMUNet2D,
    VDMUNet2D,
    _channel_mult_for_resolution,
    _parse_create_model_args,
    _parse_layers_per_block,
    get_unet_type_config,
    SUPPORTED_UNET_TYPES,
    UNET_TYPE_ADM,
    UNET_TYPE_EDM,
    UNET_TYPE_EDM2,
    UNET_TYPE_VDM,
)

# Backward-compat alias
BiBBDMUNet = UNet2DWrapper


def _out_channels_for_objective(objective: str, in_channels: int) -> int:
    """Return the UNet output channels for the given BiBBDM objective."""
    if objective in ("dlns", "dlab", "dlgab"):
        return 2 * in_channels
    return in_channels


_BIBBDM_CLASS_MAP = {
    UNET_TYPE_ADM: UNet2DWrapper,
    UNET_TYPE_EDM: EDMUNet2D,
    UNET_TYPE_VDM: VDMUNet2D,
}


def create_bibbdm_model(
    image_size: int = 256,
    in_channels: int = 3,
    num_channels: int = 128,
    num_res_blocks: Union[int, str, Tuple[int, ...]] = 2,
    attention_resolutions: str = "32,16,8",
    dropout: float = 0.0,
    condition_mode: Optional[str] = "concat",
    channel_mult: str = "",
    objective: str = "dlns",
    unet_type: str = UNET_TYPE_ADM,
    **kwargs: Any,
) -> nn.Module:
    """Factory for BiBBDM-compatible UNet models.

    Parses string-based arguments (``attention_resolutions``, ``channel_mult``)
    into tuples and infers ``out_channels`` from the *objective*.

    Parameters
    ----------
    unet_type : str
        Backbone architecture. One of: ``adm`` (default), ``edm``, ``vdm``.
        Note: ``edm2`` is disabled due to pipeline incompatibility.
    """
    if unet_type == UNET_TYPE_EDM2:
        cfg = get_unet_type_config(UNET_TYPE_EDM2)
        raise ValueError(
            f"unet_type 'edm2' is disabled. {cfg.get('issue', 'Incompatible with pipeline.')}"
        )
    if unet_type not in SUPPORTED_UNET_TYPES:
        raise ValueError(
            f"unet_type '{unet_type}' not supported. Use one of: {SUPPORTED_UNET_TYPES}"
        )

    out_channels = _out_channels_for_objective(objective, in_channels)

    attn_indices, cm_tuple = _parse_create_model_args(
        image_size, attention_resolutions, channel_mult
    )

    cm_effective = cm_tuple if cm_tuple is not None else _channel_mult_for_resolution(image_size)
    parsed_num_res_blocks = _parse_layers_per_block(
        num_res_blocks,
        num_levels=len(cm_effective),
        allow_variable=False,
    )

    common_kwargs = dict(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=num_channels,
        num_res_blocks=parsed_num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=cm_tuple,
    )

    cls = _BIBBDM_CLASS_MAP[unet_type]

    if unet_type == UNET_TYPE_VDM:
        if "gamma_min" in kwargs:
            common_kwargs["gamma_min"] = kwargs["gamma_min"]
        if "gamma_max" in kwargs:
            common_kwargs["gamma_max"] = kwargs["gamma_max"]

    return cls(**common_kwargs)


# Backward-compat alias
create_model = create_bibbdm_model

__all__ = [
    "BiBBDMUNet",
    "create_bibbdm_model",
    "create_model",
    "_out_channels_for_objective",
]
