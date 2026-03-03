# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DAB dataset wrappers.

DAB uses the same paired dataset structure as BiBBDM.
"""

from examples.bibbdm.dataset_wrapper import (  # noqa: F401
    MavicTBiBBDMDataset as MavicTDABDataset,
)
from examples.ddbm.dataset_wrapper import PairedValDataset  # noqa: F401

__all__ = ["MavicTDABDataset", "PairedValDataset"]
