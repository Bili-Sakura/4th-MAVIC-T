"""Img2Image-Turbo (Pix2Pix-Turbo / CycleGAN-Turbo) for MAVIC-T image-to-image translation tasks.

.. note::
   **Lower priority**: the Img2Image-Turbo / Pix2Pix-Turbo method has been
   found less suitable for the MAVIC-T task compared to other baselines
   (CUT, DDBM).  Its code is retained for reference and future
   experimentation, but further implementation effort should focus on the
   other baselines first.

Organised in a ``diffusers``-compatible layout:

- **models**      – ``Pix2PixTurbo``, ``CycleGANTurbo``, VAE helpers, initialisation
- **pipelines**   – ``Pix2PixTurboPipeline``, ``CycleGANTurboPipeline``
- **schedulers**  – ``make_1step_sched`` (single-step DDPM)
- **utils**       – VAE forward functions, training datasets, DINO loss, image prep
"""

# Models
from .models import (
    Pix2PixTurbo,
    CycleGANTurbo,
    VAE_encode,
    VAE_decode,
    initialize_unet,
    initialize_vae,
)

# Pipelines
from .pipelines import (
    Pix2PixTurboPipeline,
    CycleGANTurboPipeline,
    TurboPipelineOutput,
)

# Schedulers
from .schedulers import make_1step_sched

# Utils
from .utils import (
    vae_encoder_fwd,
    vae_decoder_fwd,
    build_transform,
    PairedDataset,
    UnpairedDataset,
    canny_from_pil,
)

__all__ = [
    # Models
    "Pix2PixTurbo",
    "CycleGANTurbo",
    "VAE_encode",
    "VAE_decode",
    "initialize_unet",
    "initialize_vae",
    # Pipelines
    "Pix2PixTurboPipeline",
    "CycleGANTurboPipeline",
    "TurboPipelineOutput",
    # Schedulers
    "make_1step_sched",
    # Utils
    "vae_encoder_fwd",
    "vae_decoder_fwd",
    "build_transform",
    "PairedDataset",
    "UnpairedDataset",
    "canny_from_pil",
]
