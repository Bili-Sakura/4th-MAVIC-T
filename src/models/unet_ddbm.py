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
- ``adm``: ADM-style diffusers UNet2DModel (default, implemented).
- ``edm``: EDM/DDPM++ style from libs/DDBM (SongUNet), placeholder.
- ``edm2``: EDM2 magnitude-preserving UNet from libs/edm2, placeholder.
- ``vdm``: Variational Diffusion Model from libs/vdm, placeholder.
- ``sid``: Simple Diffusion from libs/simpleDiffusion, placeholder.
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

    Used for documentation and validation. Actual implementations may
    require additional parameters.
    """
    configs = {
        UNET_TYPE_ADM: {
            "source": "diffusers.UNet2DModel",
            "description": "ADM-style UNet with positional/sinusoidal time embedding, GroupNorm, ResNet blocks.",
            "implemented": True,
        },
        UNET_TYPE_EDM: {
            "source": "libs/DDBM/ddbm/models/edm_unet.SongUNet",
            "description": "EDM/DDPM++/NCSN++ UNetBlock with Fourier embedding, [1,3,3,1] resample filter.",
            "implemented": False,
        },
        UNET_TYPE_EDM2: {
            "source": "libs/edm2/training/networks_edm2.UNet",
            "description": "EDM2 magnitude-preserving UNet with MPConv, mp_silu, Precond wrapper.",
            "implemented": False,
        },
        UNET_TYPE_VDM: {
            "source": "libs/vdm/model_vdm.py",
            "description": "Variational Diffusion Model (logSNR = f(t)), Jax/Flax reference.",
            "implemented": False,
        },
        UNET_TYPE_SID: {
            "source": "libs/simpleDiffusion/nets/unet.py",
            "description": "Simple Diffusion UNet with shifted cosine schedule, wavelet decomposition.",
            "implemented": False,
        },
    }
    if unet_type not in configs:
        raise ValueError(
            f"Unknown unet_type '{unet_type}'. Supported: {tuple(configs.keys())}"
        )
    return configs[unet_type].copy()


def _raise_unet_placeholder(baseline: str, unet_type: str) -> None:
    """Raise NotImplementedError for unimplemented unet_type in a given baseline.

    Shared by DDBM, BiBBDM, I2SB to avoid duplicating placeholder logic.
    """
    config = get_unet_type_config(unet_type)
    raise NotImplementedError(
        f"{unet_type.upper()} UNet not yet implemented for {baseline}. "
        f"See {config['source']}. Only ADM (unet_type='adm') is supported."
    )


def _create_model_edm(
    image_size: int,
    in_channels: int,
    num_channels: int,
    num_res_blocks: int,
    attention_resolutions: str,
    dropout: float,
    condition_mode: Optional[str],
    channel_mult: str,
    **kwargs: Any,
) -> None:
    """Placeholder: EDM-style UNet (SongUNet) from libs/DDBM.

    See libs/DDBM/ddbm/models/edm_unet.SongUNet and script_util.create_model(unet_type='edm').
    """
    raise NotImplementedError(
        "EDM (SongUNet) UNet not yet implemented in this codebase. "
        "Use libs/DDBM/ddbm.utils.script_util.create_model(unet_type='edm') "
        "or port SongUNet into src/models/unet_ddbm.py."
    )


def _create_model_edm2(
    image_size: int,
    in_channels: int,
    num_channels: int,
    num_res_blocks: int,
    attention_resolutions: str,
    dropout: float,
    condition_mode: Optional[str],
    channel_mult: str,
    **kwargs: Any,
) -> None:
    """Placeholder: EDM2 magnitude-preserving UNet from libs/edm2.

    See libs/edm2/training/networks_edm2.UNet and Precond. Requires adaptation
    for bridge (xT conditioning) and DDBM preconditioning.
    """
    raise NotImplementedError(
        "EDM2 UNet not yet implemented in this codebase. "
        "See libs/edm2/training/networks_edm2.py. Adapt for bridge conditioning "
        "and DDBM c_skip/c_out/c_in preconditioning."
    )


def _create_model_vdm(
    image_size: int,
    in_channels: int,
    num_channels: int,
    num_res_blocks: int,
    attention_resolutions: str,
    dropout: float,
    condition_mode: Optional[str],
    channel_mult: str,
    **kwargs: Any,
) -> None:
    """Placeholder: Variational Diffusion Model from libs/vdm.

    See libs/vdm/model_vdm.py. VDM uses continuous-time formulation with
    logSNR = f(t). Reference implementation is Jax/Flax.
    """
    raise NotImplementedError(
        "VDM UNet not yet implemented in this codebase. "
        "See libs/vdm and README. Reference is Jax/Flax; PyTorch port needed."
    )


def _create_model_sid(
    image_size: int,
    in_channels: int,
    num_channels: int,
    num_res_blocks: int,
    attention_resolutions: str,
    dropout: float,
    condition_mode: Optional[str],
    channel_mult: str,
    **kwargs: Any,
) -> None:
    """Placeholder: Simple Diffusion UNet from libs/simpleDiffusion.

    See libs/simpleDiffusion/nets/unet.py. SiD uses shifted cosine schedule
    and optional wavelet decomposition. UNet/U-ViT backbones available.
    """
    raise NotImplementedError(
        "Simple Diffusion (SiD) UNet not yet implemented in this codebase. "
        "See libs/simpleDiffusion/nets/unet.py and diffusion/simple_diffusion.py."
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
    unet_type: str = UNET_TYPE_ADM,
    **kwargs: Any,
) -> Union[DDBMUNet, nn.Module]:
    """Factory for DDBM-compatible UNet models.

    Parses string-based arguments (``attention_resolutions``, ``channel_mult``)
    into the tuples that :class:`DDBMUNet` expects.

    Parameters
    ----------
    unet_type : str
        Backbone architecture. One of: ``adm`` (default), ``edm``, ``edm2``,
        ``vdm``, ``sid``. Only ``adm`` is implemented; others raise
        :exc:`NotImplementedError` with guidance.
    """
    if unet_type not in SUPPORTED_UNET_TYPES:
        raise ValueError(
            f"unet_type '{unet_type}' not supported. Use one of: {SUPPORTED_UNET_TYPES}"
        )

    # Route to placeholder implementations for non-ADM types
    if unet_type == UNET_TYPE_EDM:
        _create_model_edm(
            image_size=image_size,
            in_channels=in_channels,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            condition_mode=condition_mode,
            channel_mult=channel_mult,
            **kwargs,
        )
    elif unet_type == UNET_TYPE_EDM2:
        _create_model_edm2(
            image_size=image_size,
            in_channels=in_channels,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            condition_mode=condition_mode,
            channel_mult=channel_mult,
            **kwargs,
        )
    elif unet_type == UNET_TYPE_VDM:
        _create_model_vdm(
            image_size=image_size,
            in_channels=in_channels,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            condition_mode=condition_mode,
            channel_mult=channel_mult,
            **kwargs,
        )
    elif unet_type == UNET_TYPE_SID:
        _create_model_sid(
            image_size=image_size,
            in_channels=in_channels,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            condition_mode=condition_mode,
            channel_mult=channel_mult,
            **kwargs,
        )

    # ADM (default): diffusers UNet2DModel via DDBMUNet
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
