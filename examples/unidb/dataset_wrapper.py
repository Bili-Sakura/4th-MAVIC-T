"""UniDB dataset wrappers.

UniDB uses the same paired (target, source) structure as DDBM.
"""

from examples.ddbm.dataset_wrapper import (  # noqa: F401
    MavicTDDBMDataset as MavicTUniDBDataset,
    PairedValDataset,
)

__all__ = ["MavicTUniDBDataset", "PairedValDataset"]
