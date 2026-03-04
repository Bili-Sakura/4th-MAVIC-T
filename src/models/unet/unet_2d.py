# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Generic backbone models for diffusion bridge methods.

Following the `diffusers <https://github.com/huggingface/diffusers>`_ philosophy,
this module provides **method-agnostic** backbone architectures.  Method-specific
model creation and initialization (e.g. for DDBM, BiBBDM, I2SB, DDIB, etc.) is
handled by factory helpers in each method's own ``examples/<method>/model.py``
file, **not** here.

Both UNet and DiT are usable backbones for diffusion bridge (and other) baseline
methods.  They are **not** hybridized — each has its own type constants.

Backbone types supported via ``backbone_type`` in :func:`create_model`:

UNet backbones (wrapping ``diffusers.UNet2DModel``):
- ``adm`` (:class:`UNet2DWrapper`) — ADM-style UNet with sinusoidal time embedding.
- ``edm`` (:class:`EDMUNet2D`) — EDM/DDPM++ style with Fourier time embedding.
- ``edm2`` (:class:`EDM2UNet2D`) — DISABLED, see docstring below.
- ``vdm`` (:class:`VDMUNet2D`) — VDM with logSNR time normalization.

DiT backbones (dispatched to ``src.models.dit``):
- ``pixnerd`` — PixNerd DiT + NerfBlock.
- ``pixeldit`` — PixelDiT dual-level DiT.
- ``sit`` — SiT (Scalable Interpolant Transformer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from diffusers import ModelMixin, UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config


# ---------------------------------------------------------------------------
# Backbone type registry
# ---------------------------------------------------------------------------

# UNet backbone types
UNET_TYPE_ADM = "adm"
UNET_TYPE_EDM = "edm"
UNET_TYPE_EDM2 = "edm2"
UNET_TYPE_VDM = "vdm"

SUPPORTED_UNET_TYPES = (UNET_TYPE_ADM, UNET_TYPE_EDM, UNET_TYPE_VDM)

# DiT backbone types
DIT_TYPE_PIXNERD = "pixnerd"
DIT_TYPE_PIXELDIT = "pixeldit"
DIT_TYPE_SIT = "sit"

SUPPORTED_DIT_TYPES = (DIT_TYPE_PIXNERD, DIT_TYPE_PIXELDIT, DIT_TYPE_SIT)

# Combined backbone types for validation
SUPPORTED_BACKBONE_TYPES = SUPPORTED_UNET_TYPES + SUPPORTED_DIT_TYPES


def get_backbone_config(backbone_type: str) -> Dict[str, Any]:
    """Return a config hint dict for the given backbone type.

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
            "implemented": False,
            "issue": (
                "EDM2 is incompatible with DDBM/BiBBDM pipelines. The pipeline passes (c_in*x_t, "
                "rescaled_log_sigma) and applies c_skip/c_out externally, but EDM2 expects raw x, "
                "sigma as timestep, and applies its own preconditioning internally. Use adm or edm instead."
            ),
        },
        UNET_TYPE_VDM: {
            "source": "diffusers.UNet2DModel + logSNR normalization",
            "description": "VDM-style UNet with logSNR (gamma) time normalization.",
            "implemented": True,
        },
        DIT_TYPE_PIXNERD: {
            "source": "PixNerd DiT + NerfBlock (pure PyTorch)",
            "description": (
                "PixNerd pixel-space DiT with neural field decoder blocks. "
                "Uses self-attention for patch-level reasoning and hypernetwork "
                "MLP (NerfBlock) for per-pixel refinement. Backbone only — the "
                "original PixNerd flow-matching scheduler is NOT used."
            ),
            "implemented": True,
        },
        DIT_TYPE_PIXELDIT: {
            "source": "PixelDiT dual-level DiT (pure PyTorch)",
            "description": (
                "PixelDiT pixel-space dual-level DiT with patch-level semantic "
                "blocks and pixel-level transformer blocks using pixel-wise AdaLN "
                "and token compaction. Backbone only — the original PixelDiT "
                "flow-matching scheduler is NOT used."
            ),
            "implemented": True,
        },
        DIT_TYPE_SIT: {
            "source": "SiT DiT blocks (pure PyTorch)",
            "description": (
                "SiT (Scalable Interpolant Transformer) backbone with adaLN-Zero "
                "conditioning, multi-head self-attention, and sin-cos 2-D positional "
                "embeddings. Adapted from Ma et al. (2024) for image-to-image "
                "diffusion bridges. Class-label conditioning removed; source image "
                "concatenated along channels in concat mode."
            ),
            "implemented": True,
        },
    }
    if backbone_type not in configs:
        raise ValueError(
            f"Unknown backbone_type '{backbone_type}'. Supported: {tuple(configs.keys())}"
        )
    return configs[backbone_type].copy()


# Backward-compat alias
get_unet_type_config = get_backbone_config


def _raise_backbone_placeholder(baseline: str, backbone_type: str) -> None:
    """Raise ValueError for unknown backbone_type in a given baseline."""
    raise ValueError(
        f"Unknown backbone_type '{backbone_type}' for {baseline}. "
        f"Supported: {SUPPORTED_BACKBONE_TYPES}"
    )


# Backward-compat alias
_raise_unet_placeholder = _raise_backbone_placeholder


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

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


def _parse_layers_per_block(
    num_res_blocks: Union[int, str, Sequence[int]],
    num_levels: int,
    *,
    allow_variable: bool,
) -> Union[int, Tuple[int, ...]]:
    """Parse ``num_res_blocks`` into diffusers-compatible ``layers_per_block``.

    ``UNet2DModel`` only supports a scalar int for ``layers_per_block``.
    """
    if isinstance(num_res_blocks, int):
        if num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be a positive integer.")
        return num_res_blocks

    if isinstance(num_res_blocks, str):
        values = tuple(int(v.strip()) for v in num_res_blocks.split(",") if v.strip())
    elif isinstance(num_res_blocks, Sequence):
        values = tuple(int(v) for v in num_res_blocks)
    else:
        raise TypeError(
            "num_res_blocks must be an int, comma-separated string, or integer sequence."
        )

    if len(values) == 0:
        raise ValueError("num_res_blocks sequence cannot be empty.")
    if any(v <= 0 for v in values):
        raise ValueError("All num_res_blocks values must be positive integers.")
    if not allow_variable:
        raise ValueError("Variable num_res_blocks is not supported for this baseline. Use a single integer.")
    if len(values) != num_levels:
        raise ValueError(
            f"num_res_blocks has {len(values)} entries, but architecture has {num_levels} levels."
        )
    return values


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


# ---------------------------------------------------------------------------
# UNet2DWrapper — generic ADM-style UNet backbone
# ---------------------------------------------------------------------------


class UNet2DWrapper(ModelMixin, ConfigMixin):
    """Generic wrapper around ``UNet2DModel`` with optional channel-concat conditioning.

    This is the base backbone class for all bridge-based diffusion methods.
    Following the diffusers philosophy, it is **method-agnostic**: the same
    class is used by DDBM, DBIM, BiBBDM, I2SB, DDIB, SiD, BDBM, DAB, etc.

    Conditioning is handled via channel concatenation (when
    ``condition_mode='concat'`` or ``'dual'``).  Unconditional mode is
    supported by setting ``condition_mode=None``.

    Parameters
    ----------
    image_size : int
        Spatial resolution (height == width).
    in_channels : int
        Number of channels of the *target* image (and of the noisy sample).
    out_channels : int or None
        Number of output channels.  ``None`` defaults to ``in_channels``.
        For dual-learning objectives set to ``2 * in_channels``.
    model_channels : int
        Base channel count of the UNet.
    num_res_blocks : int or tuple of int
        Residual blocks per resolution level.
    attention_resolutions : tuple of int
        Down-block indices where attention is applied (0-indexed).
    dropout : float
        Dropout probability.
    condition_mode : str or None
        ``'concat'`` to concatenate condition along channels,
        ``'dual'`` for dual-endpoint conditioning (2× condition channels),
        or ``None`` for unconditional mode.
    channel_mult : tuple of int or None
        Per-level channel multipliers.  Auto-detected if ``None``.
    attention_head_dim : int or None
        Dimension per attention head. 64 stabilizes training (ADM-style).
    conditioning_channels : int or None
        Explicit conditioning channel count. If ``None``, derived from
        ``condition_mode`` and ``in_channels``.
    learn_sigma : bool
        If ``True``, doubles the output channels for variance prediction.
    mid_block_type : str or None
        Override diffusers mid-block type (e.g. ``'UNetMidBlock2D'``).
    """

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: Union[int, Tuple[int, ...]] = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
        attention_head_dim: Optional[int] = 64,
        conditioning_channels: Optional[int] = None,
        learn_sigma: bool = False,
        mid_block_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        # Resolve output channels
        if out_channels is None:
            out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels
        self.learn_sigma = learn_sigma

        # Resolve conditioning channels
        if conditioning_channels is None:
            if condition_mode == "concat":
                conditioning_channels = in_channels
            elif condition_mode == "dual":
                conditioning_channels = 2 * in_channels
            else:
                conditioning_channels = 0
        self.conditioning_channels = conditioning_channels

        unet_in_channels = in_channels + conditioning_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        layers_per_block = _parse_layers_per_block(
            num_res_blocks,
            num_levels=len(channel_mult),
            allow_variable=True,
        )

        unet_kwargs: dict = dict(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=layers_per_block,
            dropout=dropout,
        )
        if attention_head_dim is not None:
            unet_kwargs["attention_head_dim"] = attention_head_dim
        if mid_block_type is not None:
            unet_kwargs["mid_block_type"] = mid_block_type

        self.unet = UNet2DModel(**unet_kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass with optional channel-concat conditioning.

        Parameters
        ----------
        sample : Tensor  (B, C, H, W)
            Noisy sample (pre-conditioned if the scheduler requires it).
        timestep : Tensor  (B,)
            Timestep or rescaled time signal.
        condition : Tensor or None  (B, C_cond, H, W)
            Conditioning signal.  Concatenated along channels when
            ``condition_mode`` is ``'concat'`` or ``'dual'``.
            For backward compatibility, also accepted as keyword args
            ``xT``, ``cond``, or ``context``.

        Returns
        -------
        Tensor  (B, out_channels, H, W)
        """
        # Backward-compat: accept legacy keyword arg names
        if condition is None:
            for key in ('xT', 'cond', 'context'):
                if key in kwargs:
                    condition = kwargs[key]
                    break

        if self.condition_mode in ("concat", "dual") and condition is not None:
            sample = torch.cat([sample, condition], dim=1)

        return self.unet(sample, timestep).sample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load UNet; if ema_unet has no config.json, load config from unet and weights from ema_unet."""
        path = Path(pretrained_model_name_or_path)
        subfolder = kwargs.get("subfolder", "unet")
        if subfolder == "ema_unet" and not (path / "ema_unet" / "config.json").exists():
            config = cls.load_config(path / "unet")
            unet = cls.from_config(config)
            ema_path = path / "ema_unet" / "diffusion_pytorch_model.safetensors"
            if ema_path.exists():
                from safetensors.torch import load_file
                state = load_file(str(ema_path))
                unet.load_state_dict(state, strict=True)
            else:
                raise FileNotFoundError(f"EMA weights not found at: {ema_path}")
            torch_dtype = kwargs.get("torch_dtype")
            if torch_dtype is not None:
                unet = unet.to(dtype=torch_dtype)
            return unet
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


# ---------------------------------------------------------------------------
# EDMUNet2D — Fourier time embedding variant
# ---------------------------------------------------------------------------


class EDMUNet2D(ModelMixin, ConfigMixin):
    """EDM/DDPM++ style UNet using ``UNet2DModel`` with Fourier time embedding.

    Parameters are identical to :class:`UNet2DWrapper` (except no
    ``learn_sigma`` / ``mid_block_type``).
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
        attention_head_dim: Optional[int] = 64,
        conditioning_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        if conditioning_channels is None:
            if condition_mode == "concat":
                conditioning_channels = in_channels
            elif condition_mode == "dual":
                conditioning_channels = 2 * in_channels
            else:
                conditioning_channels = 0
        self.conditioning_channels = conditioning_channels

        unet_in_channels = in_channels + conditioning_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        unet_kwargs: dict = dict(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
            time_embedding_type="fourier",
        )
        if attention_head_dim is not None:
            unet_kwargs["attention_head_dim"] = attention_head_dim

        self.unet = UNet2DModel(**unet_kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if condition is None:
            for key in ('xT', 'cond', 'context'):
                if key in kwargs:
                    condition = kwargs[key]
                    break
        if self.condition_mode in ("concat", "dual") and condition is not None:
            sample = torch.cat([sample, condition], dim=1)
        return self.unet(sample, timestep).sample


# ---------------------------------------------------------------------------
# EDM2UNet2D — Fourier + preconditioning (DISABLED)
# ---------------------------------------------------------------------------


class EDM2UNet2D(ModelMixin, ConfigMixin):
    """EDM2 magnitude-preserving UNet with preconditioning wrapper.

    DISABLED: Incompatible with DDBM/BiBBDM pipelines (see module-level annotation).
    Use ``adm`` or ``edm`` backbones instead.

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
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
        sigma_data: float = 0.5,
        conditioning_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        self.sigma_data = sigma_data

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        if conditioning_channels is None:
            if condition_mode == "concat":
                conditioning_channels = in_channels
            elif condition_mode == "dual":
                conditioning_channels = 2 * in_channels
            else:
                conditioning_channels = 0
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
            time_embedding_type="fourier",
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward with EDM2 preconditioning."""
        if condition is None:
            for key in ('xT', 'cond', 'context'):
                if key in kwargs:
                    condition = kwargs[key]
                    break

        sigma = timestep.float().reshape(-1, 1, 1, 1)
        sd2 = self.sigma_data ** 2

        c_skip = sd2 / (sigma ** 2 + sd2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + sd2).sqrt()
        c_in = 1.0 / (sd2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4.0

        x_precond = c_in * sample
        if self.condition_mode in ("concat", "dual") and condition is not None:
            x_precond = torch.cat([x_precond, condition], dim=1)

        F_x = self.unet(x_precond, c_noise).sample
        return c_skip * sample + c_out * F_x


# ---------------------------------------------------------------------------
# VDMUNet2D — logSNR normalization
# ---------------------------------------------------------------------------


class VDMUNet2D(ModelMixin, ConfigMixin):
    """Variational Diffusion Model UNet with logSNR time normalization.

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
        out_channels: Optional[int] = None,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (1,),
        dropout: float = 0.0,
        condition_mode: Optional[str] = "concat",
        channel_mult: Optional[Tuple[int, ...]] = None,
        attention_head_dim: Optional[int] = 64,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        conditioning_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.condition_mode = condition_mode
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        if channel_mult is None:
            channel_mult = _channel_mult_for_resolution(image_size)

        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        if conditioning_channels is None:
            if condition_mode == "concat":
                conditioning_channels = in_channels
            elif condition_mode == "dual":
                conditioning_channels = 2 * in_channels
            else:
                conditioning_channels = 0
        self.conditioning_channels = conditioning_channels

        unet_in_channels = in_channels + conditioning_channels
        block_out_channels = tuple(model_channels * m for m in channel_mult)
        down_block_types, up_block_types = _build_block_types(channel_mult, attention_resolutions)

        unet_kwargs: dict = dict(
            sample_size=image_size,
            in_channels=unet_in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            layers_per_block=num_res_blocks,
            dropout=dropout,
        )
        if attention_head_dim is not None:
            unet_kwargs["attention_head_dim"] = attention_head_dim

        self.unet = UNet2DModel(**unet_kwargs)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward with logSNR normalization."""
        if condition is None:
            for key in ('xT', 'cond', 'context'):
                if key in kwargs:
                    condition = kwargs[key]
                    break

        t_normalized = (timestep.float() - self.gamma_min) / (self.gamma_max - self.gamma_min)
        if self.condition_mode in ("concat", "dual") and condition is not None:
            sample = torch.cat([sample, condition], dim=1)
        return self.unet(sample, t_normalized).sample


# ---------------------------------------------------------------------------
# Generic backbone factory
# ---------------------------------------------------------------------------

_UNET_CLASS_MAP: Dict[str, type] = {
    UNET_TYPE_ADM: UNet2DWrapper,
    UNET_TYPE_EDM: EDMUNet2D,
    UNET_TYPE_VDM: VDMUNet2D,
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
    backbone_type: str = UNET_TYPE_ADM,
    attention_head_dim: Optional[int] = 64,
    **kwargs: Any,
) -> nn.Module:
    """Factory for generic backbone models.

    Parses string-based arguments (``attention_resolutions``, ``channel_mult``)
    into the tuples that the wrapper classes expect.

    Parameters
    ----------
    backbone_type : str
        Backbone architecture. One of: ``adm`` (default), ``edm``, ``vdm``,
        ``pixnerd``, ``pixeldit``, ``sit``.
        Note: ``edm2`` is disabled due to pipeline incompatibility.
    """
    # Backward compat: accept old kwarg name ``unet_type``
    if "unet_type" in kwargs:
        backbone_type = kwargs.pop("unet_type")

    if backbone_type == UNET_TYPE_EDM2:
        cfg = get_backbone_config(UNET_TYPE_EDM2)
        raise ValueError(
            f"backbone_type 'edm2' is disabled. {cfg.get('issue', 'Incompatible with pipeline.')}"
        )
    if backbone_type not in SUPPORTED_BACKBONE_TYPES:
        raise ValueError(
            f"backbone_type '{backbone_type}' not supported. Use one of: {SUPPORTED_BACKBONE_TYPES}"
        )

    # PixNerd uses a completely different parameter set from UNet backbones.
    if backbone_type == DIT_TYPE_PIXNERD:
        from ..dit.pixnerd import PixNerdBackbone
        return PixNerdBackbone(
            image_size=image_size,
            in_channels=in_channels,
            hidden_size=kwargs.get("pixnerd_hidden_size", 1152),
            hidden_size_x=kwargs.get("pixnerd_hidden_size_x", 64),
            nerf_mlp_ratio=kwargs.get("pixnerd_nerf_mlp_ratio", 4),
            num_blocks=kwargs.get("pixnerd_num_blocks", 18),
            num_cond_blocks=kwargs.get("pixnerd_num_cond_blocks", 4),
            patch_size=kwargs.get("pixnerd_patch_size", 2),
            num_groups=kwargs.get("pixnerd_num_groups", 12),
            condition_mode=condition_mode,
            dropout=dropout,
        )

    # PixelDiT uses a completely different parameter set from UNet backbones.
    if backbone_type == DIT_TYPE_PIXELDIT:
        from ..dit.pixeldit import PixelDiTBackbone
        return PixelDiTBackbone(
            image_size=image_size,
            in_channels=in_channels,
            hidden_size=kwargs.get("pixeldit_hidden_size", 1152),
            pixel_dim=kwargs.get("pixeldit_pixel_dim", 16),
            patch_depth=kwargs.get("pixeldit_patch_depth", 26),
            pixel_depth=kwargs.get("pixeldit_pixel_depth", 4),
            num_heads=kwargs.get("pixeldit_num_heads", 16),
            pixel_num_heads=kwargs.get("pixeldit_pixel_num_heads", 16),
            patch_size=kwargs.get("pixeldit_patch_size", 16),
            mlp_ratio=kwargs.get("pixeldit_mlp_ratio", 4.0),
            condition_mode=condition_mode,
            dropout=dropout,
        )

    # SiT uses a completely different parameter set from UNet backbones.
    if backbone_type == DIT_TYPE_SIT:
        from ..dit.sit import SiTBackbone
        return SiTBackbone(
            image_size=image_size,
            patch_size=kwargs.get("sit_patch_size", 2),
            in_channels=in_channels,
            hidden_size=kwargs.get("sit_hidden_size", 1152),
            depth=kwargs.get("sit_depth", 28),
            num_heads=kwargs.get("sit_num_heads", 16),
            mlp_ratio=kwargs.get("sit_mlp_ratio", 4.0),
            condition_mode=condition_mode,
            dropout=dropout,
        )

    attn_indices, cm_tuple = _parse_create_model_args(
        image_size, attention_resolutions, channel_mult
    )

    cm_effective = cm_tuple if cm_tuple is not None else _channel_mult_for_resolution(image_size)
    parsed_num_res_blocks = _parse_layers_per_block(
        num_res_blocks,
        num_levels=len(cm_effective),
        allow_variable=False,
    )

    common_kwargs: dict = dict(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        num_res_blocks=parsed_num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=cm_tuple,
        attention_head_dim=attention_head_dim,
    )

    # Pass through optional params from kwargs
    for opt_key in ('out_channels', 'conditioning_channels', 'learn_sigma', 'mid_block_type'):
        if opt_key in kwargs:
            common_kwargs[opt_key] = kwargs[opt_key]

    cls = _UNET_CLASS_MAP[backbone_type]

    # VDM accepts extra init parameters via kwargs
    if backbone_type == UNET_TYPE_VDM:
        if "gamma_min" in kwargs:
            common_kwargs["gamma_min"] = kwargs["gamma_min"]
        if "gamma_max" in kwargs:
            common_kwargs["gamma_max"] = kwargs["gamma_max"]

    return cls(**common_kwargs)
