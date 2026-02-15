"""BDBM dataset wrappers.

BDBM uses the same paired dataset structure as BiBBDM.
"""

from examples.bibbdm.dataset_wrapper import (  # noqa: F401
    MavicTBiBBDMDataset as MavicTBDBMDataset,
)
from examples.ddbm.dataset_wrapper import PairedValDataset  # noqa: F401

__all__ = ["MavicTBDBMDataset", "PairedValDataset"]
