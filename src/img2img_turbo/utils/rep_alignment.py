"""Representation alignment via pre-trained image encoders.

This module re-exports all classes from the shared :mod:`src.rep_alignment`
module for backward compatibility.  New code should import directly from
:mod:`src.rep_alignment`.
"""

from src.rep_alignment import (  # noqa: F401
    RepresentationAlignmentBase,
    SARCLIPAlignment,
    DINOv3SatAlignment,
    build_projector_mlp,
)
