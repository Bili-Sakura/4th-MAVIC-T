# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""UniDB pipeline for image-to-image translation.

Based on https://github.com/2769433owo/UniDB-plusplus.
"""

from .pipeline_unidb import UniDBPipeline, UniDBPipelineOutput

__all__ = ["UniDBPipeline", "UniDBPipelineOutput"]
