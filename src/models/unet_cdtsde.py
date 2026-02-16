"""CDTSDE-compatible UNet model built on ``diffusers.UNet2DModel``.

This module adapts the core idea from CDTSDE (adaptive domain-shift field
``Lambda_t``) into this repository's diffusers-style baseline structure.

Like other bridge baselines here, the UNet follows the DDBM calling
convention ``model(x_t, timestep, xT=source)`` with optional channel
concatenation conditioning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config


def _build_block_types(
    channel_mult: Tuple[int, ...],
    attention_resolutions: Tuple[int, ...],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Build down/up block type tuples from channel multipliers."""
    down_block_types = []
    for i in range(len(channel_mult)):
        down_block_types.append("AttnDownBlock2D" if i in attention_resolutions else "DownBlock2D")

    up_block_types = []
    for i in range(len(channel_mult)):
        rev_i = len(channel_mult) - 1 - i
        up_block_types.append("AttnUpBlock2D" if rev_i in attention_resolutions else "UpBlock2D")

    return tuple(down_block_types), tuple(up_block_types)


def _channel_mult_for_resolution(resolution: int) -> Tuple[int, ...]:
    """Return default channel multipliers by resolution."""
    return {
        512: (1, 1, 2, 2, 4, 4),
        256: (1, 1, 2, 2, 4, 4),
        128: (1, 1, 2, 3, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


def _parse_create_model_args(
    image_size: int,
    attention_resolutions: Union[str, Tuple[int, ...]],
    channel_mult: Union[str, Tuple[int, ...], None],
) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
    """Parse string-based config fields to tuples."""
    cm_tuple: Optional[Tuple[int, ...]] = None
    if channel_mult and isinstance(channel_mult, str) and channel_mult != "":
        cm_tuple = tuple(int(c) for c in channel_mult.split(","))
    elif isinstance(channel_mult, tuple) and channel_mult:
        cm_tuple = channel_mult

    attn_indices: Tuple[int, ...] = ()
    if attention_resolutions:
        if isinstance(attention_resolutions, str):
            attn_res_list = [int(r) for r in attention_resolutions.split(",")]
        else:
            attn_res_list = list(attention_resolutions)

        cm = cm_tuple if cm_tuple else _channel_mult_for_resolution(image_size)
        attn_indices = tuple(
            i for i in range(len(cm)) if image_size // (2 ** i) in attn_res_list
        )

    return attn_indices, cm_tuple


class _LambdaField(nn.Module):
    """Predicts a channel-wise spatial ``Lambda_t`` field in [0, 1]."""

    def __init__(self, out_channels: int, hidden_channels: int = 16):
        super().__init__()
        def _group_count(ch: int) -> int:
            for g in (8, 4, 2, 1):
                if g <= ch and ch % g == 0:
                    return g
            return 1

        groups1 = _group_count(hidden_channels)
        groups2 = _group_count(hidden_channels * 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=5, padding=2),
            nn.GroupNorm(groups1, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(groups2, hidden_channels * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _positional_bias(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, h, 1)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, w)
        bias = 0.5 * (torch.sin(torch.pi * ys) + torch.cos(torch.pi * xs))
        return bias.unsqueeze(1)

    def forward(self, lambda_linear: torch.Tensor, target_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        bsz, channels, height, width = target_shape
        lam = lambda_linear
        if lam.ndim == 1:
            lam = lam.view(bsz, 1, 1, 1)
        elif lam.ndim == 2:
            lam = lam.view(bsz, 1, 1, 1)
        elif lam.ndim != 4:
            raise ValueError(f"Unsupported lambda_linear shape: {tuple(lam.shape)}")

        lam = lam.clamp(0.0, 1.0)
        lam_grid = lam.expand(bsz, 1, height, width)
        pos = self._positional_bias(height, width, device=lam.device, dtype=lam.dtype).expand_as(lam_grid)

        raw = self.net(lam_grid + 0.2 * pos)
        mod = torch.sigmoid(raw)
        boundary = lam_grid
        # Boundary-preserving modulation: f(0)=0, f(1)=1 while allowing spatial variation.
        lam_hat = boundary + boundary * (1.0 - boundary) * (2.0 * mod - 1.0)
        lam_hat = torch.sigmoid(lam_hat * 6.0 - 3.0).clamp(0.0, 1.0)

        if lam_hat.shape[1] != channels:
            lam_hat = lam_hat[:, :1].expand(bsz, channels, height, width)
        return lam_hat


class CDTSDEUNet(ModelMixin, ConfigMixin):
    """Diffusers-native UNet wrapper with CDTSDE lambda field."""

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
        lambda_hidden_channels: int = 16,
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
        self.lambda_field = _LambdaField(
            out_channels=in_channels,
            hidden_channels=lambda_hidden_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)
        return self.unet(x, timestep).sample

    def predict_lambda(
        self,
        lambda_linear: torch.Tensor,
        target_shape: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Predict spatially varying ``Lambda_t`` for a target tensor shape."""
        return self.lambda_field(lambda_linear, target_shape)

    def mix_target_source(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        lambda_linear: torch.Tensor,
    ) -> torch.Tensor:
        """Mix target/source with dynamic spatial lambda (CDTSDE core step)."""
        lambda_hat = self.predict_lambda(lambda_linear, target.shape)
        return lambda_hat * source + (1.0 - lambda_hat) * target

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load UNet, handling ema_unet subfolder without local config."""
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

                state = load_file(str(ema_path))
                unet.load_state_dict(state, strict=True)
            return unet
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


def create_model(
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


__all__ = [
    "CDTSDEUNet",
    "create_model",
]

