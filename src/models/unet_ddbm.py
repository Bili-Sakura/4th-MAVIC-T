"""DDBM-compatible UNet model built on ``diffusers.UNet2DModel``.

The vendor DDBM UNet accepts ``(x, timestep, xT=…)`` where ``xT`` is the
source/condition image.  With ``condition_mode='concat'`` the model
internally concatenates ``x`` and ``xT`` along the channel axis.

This module replicates that contract using a standard ``UNet2DModel`` from
the Hugging Face *diffusers* library.  A thin wrapper class
:class:`DDBMUNet` concatenates source and noisy sample before forwarding to
the underlying ``UNet2DModel``, so the rest of the training / sampling code
can call ``model(x, t, xT=source)`` just like the vendor code.

Supported UNet types (via ``unet_type`` in :func:`create_model`):
- ``adm``: ADM-style diffusers UNet2DModel (default).
- ``edm``: EDM/DDPM++ style using UNet2DModel with Fourier time embedding.
- ``edm2``: EDM2 magnitude-preserving style with Fourier embedding and preconditioning.
- ``vdm``: Variational Diffusion Model with logSNR time normalization.
- ``sid``: Simple Diffusion using UNet2DModel (following libs/simpleDiffusion).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config


# ---------------------------------------------------------------------------
# UNet type registry and config placeholders
# ---------------------------------------------------------------------------

UNET_TYPE_ADM = "adm"
UNET_TYPE_EDM = "edm"
UNET_TYPE_EDM2 = "edm2"
UNET_TYPE_VDM = "vdm"
UNET_TYPE_SID = "sid"

SUPPORTED_UNET_TYPES = (UNET_TYPE_ADM, UNET_TYPE_EDM, UNET_TYPE_EDM2, UNET_TYPE_VDM, UNET_TYPE_SID)


def get_unet_type_config(unet_type: str) -> Dict[str, Any]:
    """Return a config hint dict for the given UNet type.

    Used for documentation and validation.
    """
    configs = {
        UNET_TYPE_ADM: {
            "source": "diffusers.UNet2DModel",
            "description": "ADM-style UNet with positional/sinusoidal time embedding, GroupNorm, ResNet blocks.",
            "implemented": True,
        },
        UNET_TYPE_EDM: {
            "source": "diffusers.UNet2DModel (time_embedding_type='fourier')",
            "description": "EDM/DDPM++ style UNet with Fourier time embedding via diffusers.",
            "implemented": True,
        },
        UNET_TYPE_EDM2: {
            "source": "diffusers.UNet2DModel (time_embedding_type='fourier') + EDM2 preconditioning",
            "description": "EDM2-style UNet with Fourier embedding and magnitude-preserving preconditioning.",
            "implemented": True,
        },
        UNET_TYPE_VDM: {
            "source": "diffusers.UNet2DModel + logSNR normalization",
            "description": "VDM-style UNet with logSNR (gamma) time normalization.",
            "implemented": True,
        },
        UNET_TYPE_SID: {
            "source": "diffusers.UNet2DModel (following libs/simpleDiffusion)",
            "description": "Simple Diffusion UNet using native diffusers UNet2DModel.",
            "implemented": True,
        },
    }
    if unet_type not in configs:
        raise ValueError(
            f"Unknown unet_type '{unet_type}'. Supported: {tuple(configs.keys())}"
        )
    return configs[unet_type].copy()


def _raise_unet_placeholder(baseline: str, unet_type: str) -> None:
    """Raise ValueError for unknown unet_type in a given baseline."""
    raise ValueError(
        f"Unknown unet_type '{unet_type}' for {baseline}. "
        f"Supported: {SUPPORTED_UNET_TYPES}"
    )


def _build_block_types(
    channel_mult: Tuple[int, ...],
    attention_resolutions: Tuple[int, ...],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Build down_block_types and up_block_types from channel_mult and attention indices."""
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
    return tuple(down_block_types), tuple(up_block_types)


def _channel_mult_for_resolution(resolution: int) -> Tuple[int, ...]:
    """Return a sensible default channel multiplier tuple."""
    return {
        512: (1, 1, 2, 2, 4, 4),
        256: (1, 1, 2, 2, 4, 4),
        128: (1, 1, 2, 3, 4),
        64:  (1, 2, 3, 4),
        32:  (1, 2, 3, 4),
    }.get(resolution, (1, 2, 3, 4))


class DDBMUNet(ModelMixin, ConfigMixin):
    """Wrapper around ``UNet2DModel`` that accepts the DDBM calling convention.

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


# ---------------------------------------------------------------------------
# EDM UNet – Fourier time embedding (replaces libs/DDBM SongUNet)
# ---------------------------------------------------------------------------


class EDMUNet(ModelMixin, ConfigMixin):
    """EDM/DDPM++ style UNet using ``UNet2DModel`` with Fourier time embedding.

    Ported from ``libs/DDBM/ddbm/models/edm_unet.SongUNet``. The original
    SongUNet uses random Fourier features for time conditioning; this version
    uses the native ``time_embedding_type='fourier'`` in diffusers.

    Parameters are identical to :class:`DDBMUNet`.
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
            time_embedding_type="fourier",
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


# ---------------------------------------------------------------------------
# EDM2 UNet – Fourier embedding + magnitude-preserving preconditioning
# ---------------------------------------------------------------------------


class EDM2UNet(ModelMixin, ConfigMixin):
    """EDM2 magnitude-preserving UNet with preconditioning wrapper.

    Ported from ``libs/edm2/training/networks_edm2.Precond``. The underlying
    UNet uses Fourier time embedding via ``UNet2DModel``. The forward pass
    applies EDM2 preconditioning (``c_skip``, ``c_out``, ``c_in``,
    ``c_noise``) following Equation 7 of Karras et al. (2024).

    Parameters
    ----------
    sigma_data : float
        Expected standard deviation of the training data (default 0.5).
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

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with EDM2 preconditioning.

        ``timestep`` is interpreted as sigma (noise level). The model applies:
        ``D(x; sigma) = c_skip * x + c_out * F(c_in * x; c_noise)``
        where ``c_noise = ln(sigma) / 4``.
        """
        sigma = timestep.float().reshape(-1, 1, 1, 1)
        sd2 = self.sigma_data ** 2

        c_skip = sd2 / (sigma ** 2 + sd2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + sd2).sqrt()
        c_in = 1.0 / (sd2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4.0

        x_precond = c_in * x
        if self.condition_mode == "concat" and xT is not None:
            x_precond = torch.cat([x_precond, xT], dim=1)

        F_x = self.unet(x_precond, c_noise).sample
        return c_skip * x + c_out * F_x


# ---------------------------------------------------------------------------
# VDM UNet – logSNR (gamma) time normalization
# ---------------------------------------------------------------------------


class VDMUNet(ModelMixin, ConfigMixin):
    """Variational Diffusion Model UNet with logSNR time normalization.

    Ported from ``libs/vdm/model_vdm.ScoreUNet`` (Jax/Flax). The VDM score
    model receives ``gamma = logSNR(t)`` as its time input; this wrapper
    normalizes gamma to ``[0, 1]`` before passing it to the underlying
    ``UNet2DModel``.

    Parameters
    ----------
    gamma_min : float
        Minimum logSNR value (default -13.3).
    gamma_max : float
        Maximum logSNR value (default 5.0).
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

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with logSNR normalization.

        ``timestep`` is interpreted as logSNR (gamma). It is normalized to
        ``[0, 1]`` via ``t = (gamma - gamma_min) / (gamma_max - gamma_min)``
        before being passed to the UNet.
        """
        t_normalized = (timestep.float() - self.gamma_min) / (self.gamma_max - self.gamma_min)
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)
        return self.unet(x, t_normalized).sample


# ---------------------------------------------------------------------------
# SiD UNet – Simple Diffusion (following libs/simpleDiffusion)
# ---------------------------------------------------------------------------


class SiDUNet(ModelMixin, ConfigMixin):
    """Simple Diffusion UNet using native ``UNet2DModel``.

    Ported from ``libs/simpleDiffusion/nets/unet.UNet2D`` which itself is a
    thin subclass of ``diffusers.UNet2DModel``. Identical to :class:`DDBMUNet`
    in structure, provided as a distinct class for type clarity and potential
    future customization (e.g. shifted cosine schedule integration).

    Parameters are identical to :class:`DDBMUNet`.
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
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)
        return self.unet(x, timestep).sample


def _parse_create_model_args(
    image_size: int,
    attention_resolutions: Union[str, Tuple[int, ...]],
    channel_mult: Union[str, Tuple[int, ...], None],
) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
    """Parse string-based attention_resolutions and channel_mult into tuples.

    Returns (attn_indices, cm_tuple).
    """
    # Parse channel_mult
    cm_tuple: Optional[Tuple[int, ...]] = None
    if channel_mult and isinstance(channel_mult, str) and channel_mult != "":
        cm_tuple = tuple(int(c) for c in channel_mult.split(","))
    elif isinstance(channel_mult, tuple) and channel_mult:
        cm_tuple = channel_mult

    # Parse attention_resolutions → down-block indices
    attn_indices: Tuple[int, ...] = ()
    if attention_resolutions:
        if isinstance(attention_resolutions, str):
            attn_res_list = [int(r) for r in attention_resolutions.split(",")]
        else:
            attn_res_list = list(attention_resolutions)

        cm = cm_tuple if cm_tuple else _channel_mult_for_resolution(image_size)
        attn_indices = tuple(
            i for i in range(len(cm))
            if image_size // (2 ** i) in attn_res_list
        )

    return attn_indices, cm_tuple


_UNET_CLASS_MAP: Dict[str, type] = {
    UNET_TYPE_ADM: DDBMUNet,
    UNET_TYPE_EDM: EDMUNet,
    UNET_TYPE_EDM2: EDM2UNet,
    UNET_TYPE_VDM: VDMUNet,
    UNET_TYPE_SID: SiDUNet,
}


def create_model(
    image_size: int = 256,
    in_channels: int = 3,
    num_channels: int = 128,
    num_res_blocks: int = 2,
    attention_resolutions: str = "32,16,8",
    dropout: float = 0.0,
    condition_mode: Optional[str] = "concat",
    channel_mult: str = "",
    unet_type: str = UNET_TYPE_ADM,
    **kwargs: Any,
) -> Union[DDBMUNet, EDMUNet, EDM2UNet, VDMUNet, SiDUNet]:
    """Factory for DDBM-compatible UNet models.

    Parses string-based arguments (``attention_resolutions``, ``channel_mult``)
    into the tuples that the wrapper classes expect.

    Parameters
    ----------
    unet_type : str
        Backbone architecture. One of: ``adm`` (default), ``edm``, ``edm2``,
        ``vdm``, ``sid``.
    """
    if unet_type not in SUPPORTED_UNET_TYPES:
        raise ValueError(
            f"unet_type '{unet_type}' not supported. Use one of: {SUPPORTED_UNET_TYPES}"
        )

    attn_indices, cm_tuple = _parse_create_model_args(
        image_size, attention_resolutions, channel_mult
    )

    common_kwargs = dict(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=cm_tuple,
    )

    cls = _UNET_CLASS_MAP[unet_type]

    # EDM2 and VDM accept extra init parameters via kwargs
    if unet_type == UNET_TYPE_EDM2:
        if "sigma_data" in kwargs:
            common_kwargs["sigma_data"] = kwargs["sigma_data"]
    elif unet_type == UNET_TYPE_VDM:
        if "gamma_min" in kwargs:
            common_kwargs["gamma_min"] = kwargs["gamma_min"]
        if "gamma_max" in kwargs:
            common_kwargs["gamma_max"] = kwargs["gamma_max"]

    return cls(**common_kwargs)
