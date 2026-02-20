"""SID2 dataset wrappers.

SID2 uses the same paired dataset structure as SID / DDBM.
"""

from examples.ddbm.dataset_wrapper import (  # noqa: F401
    MavicTDDBMDataset as MavicTSID2Dataset,
    PairedValDataset,
)

__all__ = ["MavicTSID2Dataset", "PairedValDataset"]
