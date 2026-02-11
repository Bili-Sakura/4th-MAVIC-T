"""Backward-compatibility shim — imports from ``models.cut_model``.

New code should import directly from :mod:`src.models`.
"""

from src.models.cut_model import (  # noqa: F401
    CUTGenerator,
    PatchGANDiscriminator,
    PatchSampleMLP,
    GANLoss,
    PatchNCELoss,
    create_generator,
    create_discriminator,
    create_patch_sample_mlp,
)

__all__ = [
    "CUTGenerator",
    "PatchGANDiscriminator",
    "PatchSampleMLP",
    "GANLoss",
    "PatchNCELoss",
    "create_generator",
    "create_discriminator",
    "create_patch_sample_mlp",
]
