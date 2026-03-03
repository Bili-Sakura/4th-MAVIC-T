# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Backward-compatibility shim — imports from ``models.stegogan_model``.

New code should import directly from :mod:`src.models`.
"""

from src.models.stegogan_model import (  # noqa: F401
    StegoGANGeneratorA,
    StegoGANGeneratorB,
    StegoGANDiscriminator,
    StegoGANLoss,
    create_generator_a,
    create_generator_b,
    create_discriminator,
)

__all__ = [
    "StegoGANGeneratorA",
    "StegoGANGeneratorB",
    "StegoGANDiscriminator",
    "StegoGANLoss",
    "create_generator_a",
    "create_generator_b",
    "create_discriminator",
]
