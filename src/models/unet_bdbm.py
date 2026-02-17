"""BDBM-compatible UNet wrappers built on ``diffusers.UNet2DModel``.

This module ports the core UNet contract from ``libs/BDBM`` into the
project's diffusers-native style.

Differences vs BiBBDM:
- BDBM commonly uses ``condition_mode='dual'`` where the context packs both
  endpoint slots (2 * C channels), resulting in 3 * C UNet input channels.
- BDBM objectives are ``noise``, ``sum``, ``both`` where ``both`` predicts
  concatenated endpoint tensors and therefore doubles output channels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .unet_ddbm import _build_block_types, _parse_create_model_args


UNET_TYPE_ADM = "adm"
SUPPORTED_UNET_TYPES = (UNET_TYPE_ADM,)


def _channel_mult_for_resolution(resolution: int) -> Tuple[int, ...]:
    """Return sensible channel multiplier defaults by image size."""
    return {
        1024: (1, 1, 2, 2, 4, 4),
        512: (1, 1, 2, 2, 4, 4),
        256: (1, 1, 2, 2, 4, 4),
        128: (1, 1, 2, 3, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


class BDBMUNet(ModelMixin, ConfigMixin):
    """Diffusers UNet wrapper matching BDBM's ``(x_t, t, context=...)`` contract."""

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
        condition_mode: Optional[str] = "dual",
        channel_mult: Optional[Tuple[int, ...]] = None,
        conditioning_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        if conditioning_channels is None:
            if condition_mode == "concat":
                conditioning_channels = in_channels
            elif condition_mode == "dual":
                conditioning_channels = 2 * in_channels
            else:
                conditioning_channels = 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_mode = condition_mode
        self.conditioning_channels = conditioning_channels

        unet_in_channels = in_channels + conditioning_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for BDBM UNet objective prediction."""
        if self.condition_mode in ("concat", "dual"):
            if context is None:
                raise ValueError(
                    f"condition_mode='{self.condition_mode}' requires context input."
                )
            x_t = torch.cat([x_t, context], dim=1)

        expected = int(self.unet.config.in_channels)
        if x_t.shape[1] != expected:
            raise ValueError(
                f"UNet expected {expected} input channels, got {x_t.shape[1]}."
            )
        return self.unet(x_t, timesteps).sample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load UNet with EMA fallback when ``ema_unet`` lacks config.json."""
        path = Path(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder", "unet")
        if subfolder == "ema_unet" and not (path / "ema_unet" / "config.json").exists():
            unet = super().from_pretrained(
                path,
                subfolder="unet",
                **{k: v for k, v in kwargs.items() if k != "subfolder"},
            )
            ema_path = path / "ema_unet" / "diffusion_pytorch_model.safetensors"
            if ema_path.exists():
                from safetensors.torch import load_file

                unet.load_state_dict(load_file(str(ema_path)), strict=True)
            return unet
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


def _out_channels_for_objective(objective: str, in_channels: int) -> int:
    """Return output channels for BDBM objective."""
    if objective == "both":
        return 2 * in_channels
    return in_channels


def create_model(
    image_size: int = 256,
    in_channels: int = 3,
    num_channels: int = 128,
    num_res_blocks: int = 2,
    attention_resolutions: str = "32,16,8",
    dropout: float = 0.0,
    condition_mode: Optional[str] = "dual",
    channel_mult: str = "",
    objective: str = "noise",
    unet_type: str = UNET_TYPE_ADM,
    conditioning_channels: Optional[int] = None,
    **_: Any,
) -> nn.Module:
    """Factory for BDBM-compatible UNet models."""
    if unet_type not in SUPPORTED_UNET_TYPES:
        raise ValueError(
            f"unet_type '{unet_type}' not supported. Use one of: {SUPPORTED_UNET_TYPES}"
        )

    attn_indices, cm_tuple = _parse_create_model_args(
        image_size,
        attention_resolutions,
        channel_mult,
    )
    out_channels = _out_channels_for_objective(objective, in_channels)
    return BDBMUNet(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=cm_tuple,
        conditioning_channels=conditioning_channels,
    )


__all__ = [
    "BDBMUNet",
    "create_model",
    "_out_channels_for_objective",
    "UNET_TYPE_ADM",
    "SUPPORTED_UNET_TYPES",
]
