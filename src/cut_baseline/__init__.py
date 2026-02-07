"""CUT baseline for MAVIC-T image-to-image translation tasks.

Organised in a ``diffusers``-compatible layout:

- **models**      – ``CUTGenerator``, ``PatchGANDiscriminator``, ``PatchSampleMLP``
- **pipelines**   – ``CUTPipeline``, ``CUTPipelineOutput``
- **schedulers**  – ``CUTScheduler``, ``CUTSchedulerOutput``
- **utils**       – image helpers, image pool
"""

# Models
from .models import (
    CUTGenerator,
    PatchGANDiscriminator,
    PatchSampleMLP,
    GANLoss,
    PatchNCELoss,
    create_generator,
    create_discriminator,
    create_patch_sample_mlp,
)

# Pipelines
from .pipelines import CUTPipeline, CUTPipelineOutput

# Schedulers
from .schedulers import CUTScheduler, CUTSchedulerOutput

__all__ = [
    # Models
    "CUTGenerator",
    "PatchGANDiscriminator",
    "PatchSampleMLP",
    "GANLoss",
    "PatchNCELoss",
    "create_generator",
    "create_discriminator",
    "create_patch_sample_mlp",
    # Pipelines
    "CUTPipeline",
    "CUTPipelineOutput",
    # Schedulers
    "CUTScheduler",
    "CUTSchedulerOutput",
]
