# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""PyTorch Dataset wrapper around ``MavicTImageToImageDataset`` for StegoGAN.

This module reuses the :class:`MavicTCUTDataset` from the CUT example since
StegoGAN uses the same paired/unpaired data loading and augmentation pipeline.
"""

from examples.cut.dataset_wrapper import MavicTCUTDataset  # noqa: F401

# Re-export under a StegoGAN-specific name for clarity
MavicTStegoGANDataset = MavicTCUTDataset
