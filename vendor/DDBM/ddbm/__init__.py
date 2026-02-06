# Copyright 2024 The DDBM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Denoising Diffusion Bridge Models (DDBM)

A Hugging Face diffusers-compatible implementation of DDBM for image-to-image
translation using diffusion bridges.

Paper: "Denoising Diffusion Bridge Models" (https://arxiv.org/abs/2309.16948)

Quick Start:
    >>> from ddbm import DDBMScheduler, DDBMPipeline
    >>>
    >>> # Create scheduler
    >>> scheduler = DDBMScheduler(pred_mode='vp', sigma_max=1.0)
    >>>
    >>> # Load your trained model
    >>> model = ...  # Your trained DDBM UNet
    >>>
    >>> # Create pipeline
    >>> pipeline = DDBMPipeline(unet=model, scheduler=scheduler)
    >>>
    >>> # Run inference
    >>> result = pipeline(source_image=your_image, num_inference_steps=40)
    >>> images = result.images

Components:
    - **Schedulers**: DDBMScheduler for bridge diffusion process
    - **Pipelines**: DDBMPipeline for image-to-image generation
    - **Models**: UNetModel (ADM-style), SongUNet (EDM-style)
    - **Utils**: Utility functions for training and inference
"""

__version__ = "0.4.0"

# =============================================================================
# Primary API (Diffusers-compatible components)
# =============================================================================
from .schedulers import DDBMScheduler, DDBMSchedulerOutput
from .pipelines import DDBMPipeline, DDBMPipelineOutput

# =============================================================================
# Exports
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Schedulers
    "DDBMScheduler",
    "DDBMSchedulerOutput",
    # Pipelines
    "DDBMPipeline",
    "DDBMPipelineOutput",
    # Models (lazy loaded)
    "UNetModel",
    "SongUNet",
    # Model creation utilities (lazy loaded)
    "create_model",
    "create_model_and_diffusion",
    "model_and_diffusion_defaults",
]


# =============================================================================
# Lazy imports for components with heavier dependencies
# =============================================================================
def __getattr__(name: str):
    """
    Lazy imports for components that require additional dependencies.
    
    This allows the core diffusers-compatible API (scheduler, pipeline) to work
    without installing all optional dependencies.
    """
    # Models
    if name == "UNetModel":
        from .models.unet import UNetModel
        return UNetModel
    if name == "SongUNet":
        from .models.edm_unet import SongUNet
        return SongUNet
    
    # Model creation utilities
    if name == "create_model":
        from .utils.script_util import create_model
        return create_model
    if name == "create_model_and_diffusion":
        from .utils.script_util import create_model_and_diffusion
        return create_model_and_diffusion
    if name == "model_and_diffusion_defaults":
        from .utils.script_util import model_and_diffusion_defaults
        return model_and_diffusion_defaults
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
