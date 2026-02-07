"""Latent-space target encoder for ablation studies.

This module re-exports :class:`LatentTargetEncoder` from the shared
:mod:`src.latent_target` module for backward compatibility.  New code
should import directly from :mod:`src.latent_target`.
"""

from src.latent_target import LatentTargetEncoder  # noqa: F401
