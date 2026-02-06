# Copyright 2024 The DDBM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Model creation utilities for DDBM.

This module provides functions for creating DDBM UNet models with various
configurations. For diffusion scheduling and sampling, use DDBMScheduler
and DDBMPipeline from the main ddbm package.

Example:
    >>> from ddbm.utils.script_util import create_model, model_defaults
    >>> from ddbm import DDBMScheduler, DDBMPipeline
    >>>
    >>> # Create model
    >>> model = create_model(
    ...     image_size=64,
    ...     in_channels=3,
    ...     num_channels=128,
    ...     num_res_blocks=2,
    ... )
    >>>
    >>> # Create scheduler and pipeline
    >>> scheduler = DDBMScheduler(pred_mode='vp', sigma_max=1.0)
    >>> pipeline = DDBMPipeline(unet=model, scheduler=scheduler)
"""

import argparse
from typing import Optional, Tuple, Union

from ..models.unet import UNetModel
from ..models.edm_unet import SongUNet

# Number of classes for ImageNet (used for class-conditional training)
NUM_CLASSES = 1000


def model_defaults():
    """
    Returns default parameters for DDBM model creation.

    These defaults are suitable for image-to-image translation tasks
    at 64x64 resolution.

    Returns:
        dict: Default model configuration parameters.
    """
    return dict(
        image_size=64,
        in_channels=3,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        unet_type='adm',
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        attention_type='flash',
        learn_sigma=False,
        condition_mode='concat',
    )


def model_and_diffusion_defaults():
    """
    Returns default parameters for both model and diffusion scheduler.
    
    Note: Diffusion parameters are provided for backward compatibility.
    For new code, use DDBMScheduler directly with its own configuration.

    Returns:
        dict: Combined model and diffusion default parameters.
    """
    defaults = model_defaults()
    # Diffusion defaults (for scheduler configuration)
    defaults.update(dict(
        sigma_data=0.5,
        sigma_min=0.002,
        sigma_max=80.0,
        beta_d=2,
        beta_min=0.1,
        cov_xy=0.,
        pred_mode='vp',
        weight_schedule="bridge_karras",
    ))
    return defaults


def create_model(
    image_size: int,
    in_channels: int,
    num_channels: int,
    num_res_blocks: int,
    unet_type: str = "adm",
    channel_mult: Union[str, Tuple] = "",
    learn_sigma: bool = False,
    class_cond: bool = False,
    use_checkpoint: bool = False,
    attention_resolutions: str = "16",
    num_heads: int = 1,
    num_head_channels: int = -1,
    num_heads_upsample: int = -1,
    use_scale_shift_norm: bool = False,
    dropout: float = 0,
    resblock_updown: bool = False,
    use_fp16: bool = False,
    use_new_attention_order: bool = False,
    attention_type: str = 'flash',
    condition_mode: Optional[str] = None,
):
    """
    Create a DDBM UNet model.

    Args:
        image_size: Resolution of input images (e.g., 64, 128, 256).
        in_channels: Number of input image channels (typically 3 for RGB).
        num_channels: Base number of model channels.
        num_res_blocks: Number of residual blocks per resolution level.
        unet_type: UNet architecture type - 'adm' (ADM-style) or 'edm' (EDM-style).
        channel_mult: Channel multiplier per resolution level.
        learn_sigma: Whether to predict sigma along with the denoised sample.
        class_cond: Whether to condition on class labels.
        use_checkpoint: Whether to use gradient checkpointing.
        attention_resolutions: Comma-separated resolutions for attention layers.
        num_heads: Number of attention heads.
        num_head_channels: Channels per attention head (-1 to use num_heads).
        num_heads_upsample: Heads for upsampling blocks (-1 to use num_heads).
        use_scale_shift_norm: Whether to use scale-shift normalization.
        dropout: Dropout probability.
        resblock_updown: Whether to use residual blocks for up/downsampling.
        use_fp16: Whether to use FP16 precision.
        use_new_attention_order: Whether to use reordered attention.
        attention_type: Type of attention ('flash', 'vanilla', etc.).
        condition_mode: How to condition on source image ('concat' or None).

    Returns:
        UNetModel or SongUNet: The created model.

    Example:
        >>> model = create_model(
        ...     image_size=64,
        ...     in_channels=3,
        ...     num_channels=128,
        ...     num_res_blocks=2,
        ...     unet_type='adm',
        ...     condition_mode='concat',
        ... )
    """
    # Handle channel multiplier
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"Unsupported image size: {image_size}")
    elif isinstance(channel_mult, str):
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    # Parse attention resolutions
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if unet_type == 'adm':
        return UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=(in_channels if not learn_sigma else in_channels * 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            attention_type=attention_type,
            condition_mode=condition_mode,
        )
    elif unet_type == 'edm':
        return SongUNet(
            img_resolution=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=(in_channels if not learn_sigma else in_channels * 2),
            num_blocks=4,
            attn_resolutions=[16],
            dropout=dropout,
            channel_mult=channel_mult,
            channel_mult_noise=2,
            embedding_type='fourier',
            encoder_type='residual',
            decoder_type='standard',
            resample_filter=[1, 3, 3, 1],
        )
    else:
        raise ValueError(f"Unsupported unet type: {unet_type}")


def create_model_and_diffusion(
    image_size,
    in_channels,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    attention_type,
    condition_mode,
    pred_mode,
    weight_schedule=None,
    sigma_data=0.5,
    sigma_min=0.002,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    cov_xy=0.,
    unet_type='adm',
):
    """
    Create a DDBM model and scheduler.
    
    Note: This function is provided for backward compatibility. For new code,
    use create_model() and DDBMScheduler separately:
    
        >>> from ddbm import DDBMScheduler
        >>> from ddbm.script_util import create_model
        >>> model = create_model(...)
        >>> scheduler = DDBMScheduler(...)

    Returns:
        tuple: (model, scheduler) where model is UNetModel/SongUNet and 
               scheduler is DDBMScheduler.
    """
    from .schedulers import DDBMScheduler
    
    model = create_model(
        image_size=image_size,
        in_channels=in_channels,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        unet_type=unet_type,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        attention_type=attention_type,
        condition_mode=condition_mode,
    )
    
    scheduler = DDBMScheduler(
        sigma_data=sigma_data,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        beta_d=beta_d,
        beta_min=beta_min,
        pred_mode=pred_mode,
    )
    
    return model, scheduler


# =============================================================================
# Argument parsing utilities
# =============================================================================

def add_dict_to_argparser(parser: argparse.ArgumentParser, default_dict: dict):
    """Add dictionary entries as command-line arguments."""
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """Convert argparse namespace to dictionary with selected keys."""
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    Parse boolean values from strings.
    
    Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
