"""I2SB-compatible UNet model built on ``diffusers.UNet2DModel``.

The I2SB UNet accepts ``(x, timestep, cond=…)`` where ``cond`` is the
source/condition image.  With ``condition_mode='concat'`` the model
internally concatenates ``x`` and ``cond`` along the channel axis.

This module replicates that contract using a standard ``UNet2DModel`` from
the Hugging Face *diffusers* library.  A thin wrapper class
:class:`I2SBUNet` concatenates source and noisy sample before forwarding to
the underlying ``UNet2DModel``, so the rest of the training / sampling code
can call ``model(x, t, cond=source)`` just like the vendor code.

Supported UNet types (via ``unet_type`` in :func:`create_model`):
- ``adm``: ADM-style diffusers UNet2DModel (default).
- ``edm``: EDM/DDPM++ style with Fourier time embedding.
- ``edm2``: DISABLED. See unet_ddbm.get_unet_type_config("edm2") for the issue.
- ``vdm``: VDM with logSNR time normalization.
- ``sid``: Simple Diffusion using UNet2DModel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import ModelMixin, UNet2DConditionModel, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .unet_ddbm import (
    SUPPORTED_UNET_TYPES,
    UNET_TYPE_ADM,
    UNET_TYPE_EDM,
    UNET_TYPE_EDM2,
    UNET_TYPE_VDM,
    UNET_TYPE_SID,
    _build_block_types,
    _parse_layers_per_block,
    _parse_create_model_args,
    get_unet_type_config,
)


def _channel_mult_for_resolution(resolution: int) -> Tuple[int, ...]:
    """Return a sensible default channel multiplier tuple.

    ADM-style: 256px=4 stages (256→16), 512px=5 stages (512→16), 1024px=6 stages.
    """
    return {
        1024: (1, 1, 2, 2, 4, 4),
        512: (1, 2, 4, 4, 8),
        256: (1, 2, 2, 4),
        128: (1, 1, 2, 3, 4),
        64:  (1, 2, 3, 4),
        32:  (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


class I2SBUNet(ModelMixin, ConfigMixin):
    """Wrapper around ``UNet2DModel`` that accepts the I2SB calling convention.

    Inherits from :class:`~diffusers.ModelMixin` and
    :class:`~diffusers.ConfigMixin` so that instances can be persisted and
    restored with ``save_pretrained`` / ``from_pretrained``.

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

    @register_to_config
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
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass matching I2SB UNet calling convention.

        Parameters
        ----------
        x : Tensor  (B, C, H, W)
            Noisy sample.
        timestep : Tensor  (B,)
            Timestep embedding.
        cond : Tensor or None  (B, C, H, W)
            Source/condition image.

        Returns
        -------
        Tensor  (B, C, H, W)
            Predicted noise label.
        """
        if self.condition_mode == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)
        return self.unet(x, timestep).sample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load UNet; if ema_unet has no config.json, load config from unet and weights from ema_unet."""
        path = Path(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder", "unet")
        if subfolder == "ema_unet" and not (path / "ema_unet" / "config.json").exists():
            unet = super().from_pretrained(path, subfolder="unet", **{k: v for k, v in kwargs.items() if k != "subfolder"})
            ema_path = path / "ema_unet" / "diffusion_pytorch_model.safetensors"
            if ema_path.exists():
                from safetensors.torch import load_file
                state = load_file(str(ema_path))
                unet.load_state_dict(state, strict=True)
            return unet
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


class EDMI2SBUNet(ModelMixin, ConfigMixin):
    """EDM-style I2SB UNet with Fourier time embedding."""

    @register_to_config
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
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
            time_embedding_type="fourier",
        )

    def forward(self, x, timestep, cond=None):
        if self.condition_mode == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)
        return self.unet(x, timestep).sample


class EDM2I2SBUNet(ModelMixin, ConfigMixin):
    """EDM2-style I2SB UNet with Fourier embedding and preconditioning.

    DISABLED: Incompatible with pipeline contract (see unet_ddbm.EDM2UNet). Use adm or edm.
    """

    @register_to_config
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
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        self.sigma_data = sigma_data

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
            time_embedding_type="fourier",
        )

    def forward(self, x, timestep, cond=None):
        sigma = timestep.float().reshape(-1, 1, 1, 1)
        sd2 = self.sigma_data ** 2
        c_skip = sd2 / (sigma ** 2 + sd2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + sd2).sqrt()
        c_in = 1.0 / (sd2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4.0

        x_precond = c_in * x
        if self.condition_mode == "concat" and cond is not None:
            x_precond = torch.cat([x_precond, cond], dim=1)
        F_x = self.unet(x_precond, c_noise).sample
        return c_skip * x + c_out * F_x


class VDMI2SBUNet(ModelMixin, ConfigMixin):
    """VDM-style I2SB UNet with logSNR time normalization."""

    @register_to_config
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
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        unet_in_channels = in_channels * 2 if condition_mode == "concat" else in_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(self, x, timestep, cond=None):
        t_normalized = (timestep.float() - self.gamma_min) / (self.gamma_max - self.gamma_min)
        if self.condition_mode == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)
        return self.unet(x, t_normalized).sample


class SiDI2SBUNet(ModelMixin, ConfigMixin):
    """Simple Diffusion I2SB UNet using native UNet2DModel."""

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: Union[int, Tuple[int, ...]] = 2,
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
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        layers_per_block = _parse_layers_per_block(
            num_res_blocks,
            num_levels=len(channel_mult),
            allow_variable=True,
        )

        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=in_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=layers_per_block,
            dropout=dropout,
            mid_block_type="UNetMidBlock2D",
        )

    def forward(self, x, timestep, cond=None):
        if self.condition_mode == "concat" and cond is not None:
            x = torch.cat([x, cond], dim=1)
        return self.unet(
            sample=x,
            timestep=timestep,
            encoder_hidden_states=None,
        ).sample


_I2SB_CLASS_MAP = {
    UNET_TYPE_ADM: I2SBUNet,
    UNET_TYPE_EDM: EDMI2SBUNet,
    UNET_TYPE_VDM: VDMI2SBUNet,
    UNET_TYPE_SID: SiDI2SBUNet,
}


def create_model(
    image_size: int = 256,
    in_channels: int = 3,
    num_channels: int = 128,
    num_res_blocks: Union[int, str, Tuple[int, ...]] = 2,
    attention_resolutions: str = "32,16,8",
    dropout: float = 0.0,
    condition_mode: Optional[str] = "concat",
    channel_mult: str = "",
    unet_type: str = UNET_TYPE_ADM,
    **kwargs: Any,
) -> nn.Module:
    """Factory for I2SB-compatible UNet models.

    Parses string-based arguments (``attention_resolutions``, ``channel_mult``)
    into the tuples that the wrapper classes expect.

    Parameters
    ----------
    unet_type : str
        Backbone architecture. One of: ``adm`` (default), ``edm``, ``vdm``,
        ``sid``. Note: ``edm2`` is disabled due to pipeline incompatibility.
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

    attn_indices, cm_tuple = _parse_create_model_args(
        image_size, attention_resolutions, channel_mult
    )

    cm_effective = cm_tuple if cm_tuple is not None else _channel_mult_for_resolution(image_size)
    parsed_num_res_blocks = _parse_layers_per_block(
        num_res_blocks,
        num_levels=len(cm_effective),
        allow_variable=(unet_type == UNET_TYPE_SID),
    )

    common_kwargs = dict(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        num_res_blocks=parsed_num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=cm_tuple,
    )

    cls = _I2SB_CLASS_MAP[unet_type]

    if unet_type == UNET_TYPE_VDM:
        if "gamma_min" in kwargs:
            common_kwargs["gamma_min"] = kwargs["gamma_min"]
        if "gamma_max" in kwargs:
            common_kwargs["gamma_max"] = kwargs["gamma_max"]

    return cls(**common_kwargs)