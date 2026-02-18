"""SID dataset wrappers.

SID uses the same paired dataset structure as DDBM.
"""

from examples.ddbm.dataset_wrapper import (  # noqa: F401
    MavicTDDBMDataset as MavicTSIDDataset,
    PairedValDataset,
)

__all__ = ["MavicTSIDDataset", "PairedValDataset"]
