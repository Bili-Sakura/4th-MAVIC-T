# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""GAN-based model architectures (CUT, StegoGAN)."""

from .cut_model import (
    CUTGenerator,
    PatchGANDiscriminator,
    PatchSampleMLP,
    GANLoss,
    PatchNCELoss,
    create_generator,
    create_discriminator,
    create_patch_sample_mlp,
)
from .stegogan_model import (
    StegoGANGeneratorA,
    StegoGANGeneratorB,
    StegoGANDiscriminator,
    StegoGANLoss,
    create_generator_a,
    create_generator_b,
    create_discriminator as create_stegogan_discriminator,
)

__all__ = [
    "CUTGenerator", "PatchGANDiscriminator", "PatchSampleMLP",
    "GANLoss", "PatchNCELoss",
    "create_generator", "create_discriminator", "create_patch_sample_mlp",
    "StegoGANGeneratorA", "StegoGANGeneratorB", "StegoGANDiscriminator",
    "StegoGANLoss",
    "create_generator_a", "create_generator_b", "create_stegogan_discriminator",
]
