"""DDIB-compatible unconditional UNet model built on ``diffusers.UNet2DModel``.

Unlike the DDBM baseline which uses source-conditioned (``concat``) UNets, DDIB
trains *unconditional* diffusion models on each domain independently.  At
translation time two such models are combined: one encodes the source image to
a latent via DDIM reverse sampling, and the other decodes the latent to the
target domain via DDIM forward sampling.

This module provides:

* :class:`DDIBUNet` – thin wrapper around ``diffusers.UNet2DModel``.
* :func:`create_model` – factory matching the ``guided_diffusion`` style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config


def _channel_mult_for_resolution(resolution: int) -> Tuple[int, ...]:
    """Return a sensible default channel multiplier tuple."""
    return {
        512: (1, 1, 2, 2, 4, 4),
        256: (1, 1, 2, 2, 4, 4),
        128: (1, 1, 2, 3, 4),
        64:  (1, 2, 3, 4),
        32:  (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


class DDIBUNet(ModelMixin, ConfigMixin):
    """Unconditional UNet for DDIB diffusion models.

    Inherits from :class:`~diffusers.ModelMixin` and
    :class:`~diffusers.ConfigMixin` so that instances can be persisted and
    restored with ``save_pretrained`` / ``from_pretrained``.

    Parameters
    ----------
    image_size : int
        Spatial resolution (height == width).
    in_channels : int
        Number of channels of the input image.
    model_channels : int
        Base channel count of the UNet.
    num_res_blocks : int
        Residual blocks per resolution level.
    attention_resolutions : tuple of int
        Down-block indices where attention is applied (0-indexed).
    dropout : float
        Dropout probability.
    learn_sigma : bool
        If ``True`` the model predicts both mean and variance (doubled output channels).
    channel_mult : tuple of int or None
        Per-level channel multipliers.  Auto-detected if ``None``.
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
        learn_sigma: bool = False,
        channel_mult: Optional[Tuple[int, ...]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.learn_sigma = learn_sigma

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        out_channels = in_channels * 2 if learn_sigma else in_channels

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
            in_channels=in_channels,
            out_channels=out_channels,
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
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor  (B, C, H, W)
            Noisy sample.
        timestep : Tensor  (B,)
            Integer timestep indices.

        Returns
        -------
        Tensor  (B, C, H, W)  or  (B, 2*C, H, W) when ``learn_sigma=True``
            Model prediction (noise or noise + log-variance).
        """
        return self.unet(x, timestep).sample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load UNet; if ema subfolder has no config.json, load config from base unet and weights from ema."""
        path = Path(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder", "unet")
        config_subfolder = subfolder.replace("ema_unet", "unet")
        ema_subfolder = subfolder
        if subfolder != config_subfolder and not (path / ema_subfolder / "config.json").exists():
            unet = super().from_pretrained(path, subfolder=config_subfolder, **{k: v for k, v in kwargs.items() if k != "subfolder"})
            ema_path = path / ema_subfolder / "diffusion_pytorch_model.safetensors"
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
    learn_sigma: bool = False,
    channel_mult: str = "",
    **kwargs,
) -> DDIBUNet:
    """Factory matching the ``guided_diffusion.script_util.create_model`` signature.

    Parses string-based arguments (``attention_resolutions``, ``channel_mult``)
    into the tuples that :class:`DDIBUNet` expects.
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

    return DDIBUNet(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        learn_sigma=learn_sigma,
        channel_mult=cm_tuple,
    )
