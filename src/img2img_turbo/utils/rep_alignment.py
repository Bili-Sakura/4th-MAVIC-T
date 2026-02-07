"""Representation alignment via pre-trained image encoders (placeholder).

This module defines the interfaces for representation alignment losses
used to inject features from frozen pre-trained encoders into the
Pix2Pix-Turbo training loop.

Two concrete strategies are planned:

* **SARCLIP alignment** – for SAR2EO, SAR2IR, SAR2RGB tasks.  A frozen
  SARCLIP image encoder extracts features from the input SAR image, and
  the alignment loss encourages the model's intermediate features to
  match.  Checkpoint: ``models/BiliSakura/SARCLIP``.

* **DINOv3-sat alignment** – for RGB2IR.  A frozen DINOv3-sat encoder
  extracts features from the input RGB image.
  Checkpoint: ``models/BiliSakura/DINOv3-sat``.

Both classes below are **placeholders**: they define the expected API but
raise ``NotImplementedError`` in their core methods so that concrete
implementations can be filled in once reference code is provided.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RepresentationAlignmentBase(ABC, nn.Module):
    """Abstract base for representation alignment modules.

    Sub-classes must implement :meth:`extract_features` and
    :meth:`compute_alignment_loss`.
    """

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path

    @abstractmethod
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from *images* using the frozen encoder.

        Parameters
        ----------
        images : Tensor (B, C, H, W)
            Input images in ``[-1, 1]``.

        Returns
        -------
        Tensor
            Feature tensor whose shape depends on the encoder.
        """

    @abstractmethod
    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a scalar alignment loss between two feature maps.

        Parameters
        ----------
        model_features : Tensor
            Features from the translation model's intermediate layers.
        encoder_features : Tensor
            Features from the frozen pre-trained encoder.

        Returns
        -------
        Tensor
            Scalar loss value.
        """


# ---------------------------------------------------------------------------
# SARCLIP alignment (SAR tasks)
# ---------------------------------------------------------------------------

class SARCLIPAlignment(RepresentationAlignmentBase):
    """Representation alignment using a pre-trained SARCLIP image encoder.

    Used for SAR2EO, SAR2IR, and SAR2RGB tasks.  The SARCLIP encoder
    processes the input SAR image and produces a feature representation;
    an alignment loss encourages the translation model to preserve the
    SAR-specific semantics captured by SARCLIP.

    Parameters
    ----------
    model_path : str
        Path to the SARCLIP checkpoint directory
        (e.g. ``./models/BiliSakura/SARCLIP``).

    .. note::
       This is a **placeholder**.  The concrete implementation will be
       provided once the reference SARCLIP repository is available.
    """

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        # TODO: Load the pre-trained SARCLIP image encoder from model_path
        # and freeze its weights.  Example:
        #   self.encoder = load_sarclip_encoder(model_path)
        #   self.encoder.requires_grad_(False)
        #   self.encoder.eval()

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract SARCLIP features from SAR input images.

        .. note:: Placeholder – raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "SARCLIPAlignment.extract_features() is a placeholder. "
            "Provide the concrete implementation using the SARCLIP "
            "reference repository."
        )

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SARCLIP representation alignment loss.

        .. note:: Placeholder – raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "SARCLIPAlignment.compute_alignment_loss() is a placeholder. "
            "Provide the concrete implementation using the SARCLIP "
            "reference repository."
        )


# ---------------------------------------------------------------------------
# DINOv3-sat alignment (RGB2IR task)
# ---------------------------------------------------------------------------

class DINOv3SatAlignment(RepresentationAlignmentBase):
    """Representation alignment using a pre-trained DINOv3-sat encoder.

    Used for the RGB2IR task.  The DINOv3-sat encoder processes the input
    RGB image and produces a feature representation; an alignment loss
    encourages the translation model to preserve the satellite-image
    semantics captured by DINOv3-sat.

    Parameters
    ----------
    model_path : str
        Path to the DINOv3-sat checkpoint directory
        (e.g. ``./models/BiliSakura/DINOv3-sat``).

    .. note::
       This is a **placeholder**.  The concrete implementation will be
       provided once the reference DINOv3-sat repository is available.
    """

    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        # TODO: Load the pre-trained DINOv3-sat encoder from model_path
        # and freeze its weights.  Example:
        #   self.encoder = load_dinov3_sat_encoder(model_path)
        #   self.encoder.requires_grad_(False)
        #   self.encoder.eval()

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINOv3-sat features from RGB input images.

        .. note:: Placeholder – raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "DINOv3SatAlignment.extract_features() is a placeholder. "
            "Provide the concrete implementation using the DINOv3-sat "
            "reference repository."
        )

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DINOv3-sat representation alignment loss.

        .. note:: Placeholder – raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "DINOv3SatAlignment.compute_alignment_loss() is a placeholder. "
            "Provide the concrete implementation using the DINOv3-sat "
            "reference repository."
        )
