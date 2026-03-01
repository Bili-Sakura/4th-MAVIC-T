# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""SID dataset wrappers.

SID uses the same paired dataset structure as DDBM.
"""

from examples.ddbm.dataset_wrapper import (  # noqa: F401
    MavicTDDBMDataset as MavicTSIDDataset,
    PairedValDataset,
)

__all__ = ["MavicTSIDDataset", "PairedValDataset"]
