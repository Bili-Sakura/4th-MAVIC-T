"""CUT pipeline components.

Provides the :class:`CUTPipeline` single-pass inference pipeline,
its :class:`CUTPipelineOutput` output dataclass, and the latent-space
variant :class:`CUTLatentPipeline`.
"""

from .cut_pipeline import CUTPipeline, CUTPipelineOutput
from .cut_latent_pipeline import CUTLatentPipeline, CUTLatentPipelineOutput

__all__ = [
    "CUTPipeline",
    "CUTPipelineOutput",
    "CUTLatentPipeline",
    "CUTLatentPipelineOutput",
]
