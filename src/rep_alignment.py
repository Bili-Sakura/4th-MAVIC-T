"""Representation alignment via pre-trained image encoders.

This module implements representation alignment losses inspired by REPA
(REPresentation Alignment, see ``vendor/REPA``).  The technique is
architecture-agnostic: a frozen pre-trained encoder extracts features from
the **source** image, while a trainable projection head maps the
translation model's output features into the same embedding space.  A
negative-cosine-similarity loss encourages the model to preserve the
semantic content captured by the encoder.

Two concrete strategies are provided:

* **SARCLIP alignment** – for SAR2EO, SAR2IR, SAR2RGB tasks.  A frozen
  SARCLIP ViT-L/14 image encoder extracts features from the input SAR
  image.  Checkpoint: ``models/BiliSakura/SARCLIP``.

* **DINOv3-sat alignment** – for RGB2IR.  A frozen DINOv3-sat ViT-L
  encoder extracts features from the input RGB image.
  Checkpoint: ``models/BiliSakura/DINOv3-sat``.

The alignment loss follows REPA's formulation (negative cosine similarity
averaged over the batch) and uses a 3-layer MLP projection head identical
to the one in ``vendor/REPA/models/sit.py::build_mlp``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (adapted from vendor/REPA)
# ---------------------------------------------------------------------------

def _build_projector(input_dim: int, projector_dim: int, output_dim: int) -> nn.Sequential:
    """Build a 3-layer MLP projection head (mirrors ``vendor/REPA/models/sit.py::build_mlp``)."""
    return nn.Sequential(
        nn.Linear(input_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, output_dim),
    )


def _mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions (from ``vendor/REPA/loss.py``)."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def _normalize_to_01(images: torch.Tensor) -> torch.Tensor:
    """Convert images from ``[-1, 1]`` range to ``[0, 1]``."""
    return (images + 1.0) * 0.5


def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
    """Expand 1-channel images to 3 channels."""
    if images.shape[1] == 1:
        return images.repeat(1, 3, 1, 1)
    return images


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

SARCLIP_IMAGE_SIZE = 224
SARCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
SARCLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class SARCLIPAlignment(RepresentationAlignmentBase):
    """Representation alignment using a pre-trained SARCLIP image encoder.

    Used for SAR2EO, SAR2IR, and SAR2RGB tasks.  The SARCLIP encoder
    processes the input SAR image and produces a feature representation;
    an alignment loss encourages the translation model to preserve the
    SAR-specific semantics captured by SARCLIP.

    The encoder is loaded lazily on the first call to :meth:`extract_features`
    so that the constructor succeeds even when the checkpoint is unavailable
    (e.g. in unit tests).

    Parameters
    ----------
    model_path : str
        Path to the SARCLIP checkpoint directory
        (e.g. ``./models/BiliSakura/SARCLIP``).
    projector_dim : int
        Hidden dimension of the 3-layer MLP projection head (default 2048).
    encoder_dim : int
        Output dimension of the SARCLIP ViT-L/14 encoder (default 1024).
    """

    def __init__(
        self,
        model_path: str,
        projector_dim: int = 2048,
        encoder_dim: int = 1024,
    ) -> None:
        super().__init__(model_path)
        self._encoder: Optional[nn.Module] = None
        self._encoder_loaded = False
        self._projector_dim = projector_dim
        self.encoder_dim = encoder_dim
        # Projection head – built lazily via build_projector()
        self.projector: Optional[nn.Module] = None

    # ---- encoder loading (lazy) -------------------------------------------

    def _load_encoder(self) -> None:
        """Load the frozen SARCLIP visual encoder from *model_path*."""
        if self._encoder_loaded:
            return
        import open_clip
        model = open_clip.create_model(
            "ViT-L-14", pretrained=self.model_path,
        )
        self._encoder = model.visual
        self._encoder.requires_grad_(False)
        self._encoder.eval()
        self._encoder_loaded = True
        logger.info("Loaded SARCLIP encoder from %s", self.model_path)

    @property
    def encoder(self) -> nn.Module:
        """Return the frozen encoder, loading it on first access."""
        if not self._encoder_loaded:
            self._load_encoder()
        assert self._encoder is not None
        return self._encoder

    # ---- projection head --------------------------------------------------

    def build_projector(self, model_feature_dim: int) -> nn.Module:
        """Build the trainable projection head for a given model feature dim.

        Call this once you know the channel dimension of the translation
        model's output features.  The projector maps
        ``model_feature_dim → encoder_dim``.

        Returns the projection head so it can be added to the optimiser.
        """
        self.projector = _build_projector(
            model_feature_dim, self._projector_dim, self.encoder_dim,
        )
        return self.projector

    # ---- preprocessing ----------------------------------------------------

    @staticmethod
    def _preprocess(images: torch.Tensor) -> torch.Tensor:
        """Rescale from [-1, 1] to CLIP-normalised input."""
        x = _adapt_channels(_normalize_to_01(images))
        x = F.interpolate(x, size=SARCLIP_IMAGE_SIZE, mode="bicubic", align_corners=False)
        mean = torch.tensor(SARCLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(SARCLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    # ---- public API -------------------------------------------------------

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract SARCLIP features from SAR input images.

        Parameters
        ----------
        images : Tensor (B, C, H, W)
            Input images in ``[-1, 1]``.

        Returns
        -------
        Tensor (B, D)
            Global feature vector (CLS token) from SARCLIP ViT-L/14.
        """
        x = self._preprocess(images)
        enc = self.encoder
        device = next(enc.parameters()).device
        return enc(x.to(device))

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SARCLIP representation alignment loss.

        Uses negative cosine similarity following REPA (``vendor/REPA/loss.py``).

        Parameters
        ----------
        model_features : Tensor
            If 4-D ``(B, C, H, W)`` – spatial features are global-average-pooled
            to ``(B, C)`` before projection.  If 2-D ``(B, D)`` – used directly.
        encoder_features : Tensor (B, D_enc)
            Features from :meth:`extract_features`.

        Returns
        -------
        Tensor
            Scalar alignment loss (lower is better alignment).
        """
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])  # global average pool
        if self.projector is not None:
            model_features = self.projector(model_features)
        z_model = F.normalize(model_features, dim=-1)
        z_enc = F.normalize(encoder_features.detach(), dim=-1)
        return -(z_model * z_enc).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# DINOv3-sat alignment (RGB2IR task)
# ---------------------------------------------------------------------------

DINO_IMAGE_SIZE = 224
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


class DINOv3SatAlignment(RepresentationAlignmentBase):
    """Representation alignment using a pre-trained DINOv3-sat encoder.

    Used for the RGB2IR task.  The DINOv3-sat encoder processes the input
    RGB image and produces a feature representation; an alignment loss
    encourages the translation model to preserve the satellite-image
    semantics captured by DINOv3-sat.

    The encoder is loaded lazily on the first call to :meth:`extract_features`.

    Parameters
    ----------
    model_path : str
        Path to the DINOv3-sat checkpoint directory
        (e.g. ``./models/BiliSakura/DINOv3-sat``).
    projector_dim : int
        Hidden dimension of the 3-layer MLP projection head (default 2048).
    encoder_dim : int
        Output dimension of the DINOv3-sat ViT-L encoder (default 1024).
    """

    def __init__(
        self,
        model_path: str,
        projector_dim: int = 2048,
        encoder_dim: int = 1024,
    ) -> None:
        super().__init__(model_path)
        self._encoder: Optional[nn.Module] = None
        self._encoder_loaded = False
        self._projector_dim = projector_dim
        self.encoder_dim = encoder_dim
        self.projector: Optional[nn.Module] = None

    # ---- encoder loading (lazy) -------------------------------------------

    def _load_encoder(self) -> None:
        """Load the frozen DINOv3-sat encoder from *model_path*."""
        if self._encoder_loaded:
            return
        self._encoder = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14",
        )
        ckpt_path = self.model_path
        import os
        if os.path.isdir(ckpt_path):
            candidates = [
                os.path.join(ckpt_path, f)
                for f in os.listdir(ckpt_path)
                if f.endswith((".pth", ".pt", ".bin"))
            ]
            if candidates:
                ckpt_path = candidates[0]
        if os.path.isfile(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self._encoder.load_state_dict(state_dict, strict=False)
        self._encoder.requires_grad_(False)
        self._encoder.eval()
        self._encoder_loaded = True
        logger.info("Loaded DINOv3-sat encoder from %s", self.model_path)

    @property
    def encoder(self) -> nn.Module:
        """Return the frozen encoder, loading it on first access."""
        if not self._encoder_loaded:
            self._load_encoder()
        assert self._encoder is not None
        return self._encoder

    # ---- projection head --------------------------------------------------

    def build_projector(self, model_feature_dim: int) -> nn.Module:
        """Build the trainable projection head for a given model feature dim.

        Returns the projection head so it can be added to the optimiser.
        """
        self.projector = _build_projector(
            model_feature_dim, self._projector_dim, self.encoder_dim,
        )
        return self.projector

    # ---- preprocessing ----------------------------------------------------

    @staticmethod
    def _preprocess(images: torch.Tensor) -> torch.Tensor:
        """Rescale from [-1, 1] to ImageNet-normalised input."""
        x = _adapt_channels(_normalize_to_01(images))
        x = F.interpolate(x, size=DINO_IMAGE_SIZE, mode="bicubic", align_corners=False)
        mean = torch.tensor(DINO_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(DINO_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    # ---- public API -------------------------------------------------------

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINOv3-sat features from RGB input images.

        Parameters
        ----------
        images : Tensor (B, C, H, W)
            Input images in ``[-1, 1]``.

        Returns
        -------
        Tensor (B, D)
            CLS-token feature from DINOv3-sat ViT-L.
        """
        x = self._preprocess(images)
        enc = self.encoder
        device = next(enc.parameters()).device
        features = enc(x.to(device))
        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                features = features["x_norm_clstoken"]
            elif "cls_token" in features:
                features = features["cls_token"]
            else:
                raise KeyError(
                    f"DINOv3-sat encoder returned dict with keys {list(features.keys())}; "
                    "expected 'x_norm_clstoken' or 'cls_token'"
                )
        if features.ndim == 3:
            features = features[:, 0]
        return features

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DINOv3-sat representation alignment loss.

        Uses negative cosine similarity following REPA (``vendor/REPA/loss.py``).

        Parameters
        ----------
        model_features : Tensor
            If 4-D ``(B, C, H, W)`` – spatial features are global-average-pooled
            to ``(B, C)`` before projection.  If 2-D ``(B, D)`` – used directly.
        encoder_features : Tensor (B, D_enc)
            Features from :meth:`extract_features`.

        Returns
        -------
        Tensor
            Scalar alignment loss (lower is better alignment).
        """
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])  # global average pool
        if self.projector is not None:
            model_features = self.projector(model_features)
        z_model = F.normalize(model_features, dim=-1)
        z_enc = F.normalize(encoder_features.detach(), dim=-1)
        return -(z_model * z_enc).sum(dim=-1).mean()
