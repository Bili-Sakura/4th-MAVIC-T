# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""CDTSDE dataset wrappers.

CDTSDE uses the same paired dataset structure as DDBM.
"""

from examples.ddbm.dataset_wrapper import (  # noqa: F401
    MavicTDDBMDataset as MavicTCDTSDEDataset,
    PairedValDataset,
)

__all__ = ["MavicTCDTSDEDataset", "PairedValDataset"]

