"""I2SB baseline for MAVIC-T image-to-image translation tasks.

Organised in a ``diffusers``-compatible layout:

- **schedulers** – ``I2SBScheduler``, ``I2SBSchedulerOutput``
- **pipelines**  – ``I2SBPipeline``, ``I2SBPipelineOutput``
- **models**     – ``I2SBUNet``, ``create_model``
"""

# Schedulers
from .schedulers import I2SBScheduler, I2SBSchedulerOutput

# Pipelines
from .pipelines import I2SBPipeline, I2SBPipelineOutput

# Models
from .models import I2SBUNet, create_model

__all__ = [
    # Schedulers
    "I2SBScheduler",
    "I2SBSchedulerOutput",
    # Pipelines
    "I2SBPipeline",
    "I2SBPipelineOutput",
    # Models
    "I2SBUNet",
    "create_model",
]
