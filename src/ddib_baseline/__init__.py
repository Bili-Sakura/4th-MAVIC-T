"""DDIB baseline for MAVIC-T image-to-image translation tasks.

Dual Diffusion Implicit Bridges (DDIB, ICLR 2023) translates images between
two domains by training independent unconditional diffusion models on each
domain, then connecting them through a shared DDIM latent space.

Organised in a ``diffusers``-compatible layout:

- **schedulers** – ``DDIBScheduler``, ``DDIBSchedulerOutput``
- **pipelines**  – ``DDIBPipeline``, ``DDIBPipelineOutput``
- **models**     – ``DDIBUNet``, ``create_model``
- **utils**      – neural-network helpers, FP16 utilities
"""

# Schedulers
from .schedulers import DDIBScheduler, DDIBSchedulerOutput

# Pipelines
from .pipelines import DDIBPipeline, DDIBPipelineOutput

# Models
from .models import DDIBUNet, create_model

__all__ = [
    # Schedulers
    "DDIBScheduler",
    "DDIBSchedulerOutput",
    # Pipelines
    "DDIBPipeline",
    "DDIBPipelineOutput",
    # Models
    "DDIBUNet",
    "create_model",
]
