"""Img2Image-Turbo utility functions.

Provides VAE helpers, training dataset classes, image preparation,
DINO-based structural similarity loss, latent target encoding, and
representation alignment placeholders – all ported from or extending
``vendor/Img2Image-Turbo``.
"""

from .vae_utils import vae_encoder_fwd, vae_decoder_fwd
from .training_utils import build_transform, PairedDataset, UnpairedDataset
from .image_prep import canny_from_pil
from .latent_target import LatentTargetEncoder
from .rep_alignment import (
    RepresentationAlignmentBase,
    SARCLIPAlignment,
    DINOv3SatAlignment,
)

__all__ = [
    # VAE helpers
    "vae_encoder_fwd",
    "vae_decoder_fwd",
    # Training datasets
    "build_transform",
    "PairedDataset",
    "UnpairedDataset",
    # Image prep
    "canny_from_pil",
    # Latent target encoder
    "LatentTargetEncoder",
    # Representation alignment
    "RepresentationAlignmentBase",
    "SARCLIPAlignment",
    "DINOv3SatAlignment",
]
