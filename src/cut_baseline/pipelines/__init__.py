"""CUT pipeline components.

Provides the :class:`CUTPipeline` single-pass inference pipeline and
its :class:`CUTPipelineOutput` output dataclass.
"""

from .cut_pipeline import CUTPipeline, CUTPipelineOutput

__all__ = ["CUTPipeline", "CUTPipelineOutput"]
