"""Representation alignment via pre-trained image encoders.

This module implements the REPA (REPresentation Alignment) technique
adapted from ``vendor/REPA`` for use in any baseline training loop
(Pix2Pix-Turbo, CUT, DDBM, etc.).  The core idea is architecture-agnostic:
a frozen pre-trained encoder extracts features from input images, and a
learnable projector MLP maps the translation model's features into the
encoder's representation space.  The alignment loss is the mean negative
cosine similarity between the projected model features and the encoder
features, following the REPA paper.

Two concrete strategies are provided:

* **DINOv3-sat alignment** – for SAR2EO, SAR2IR, SAR2RGB tasks.  A frozen
  DINOv3-sat encoder extracts features from the input SAR image.
  Checkpoint: ``models/facebook/dinov3-vitl16-pretrain-sat493m``.

* **SARCLIP alignment** – for the RGB2IR task.  A frozen SARCLIP image
  encoder extracts features from the input RGB image.
  Checkpoint: ``models/BiliSakura/SARCLIP-ViT-L-14``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# Projector MLP (following REPA / vendor/REPA/models/sit.py)
# ---------------------------------------------------------------------------

def build_projector_mlp(
    input_dim: int,
    projector_dim: int,
    output_dim: int,
) -> nn.Sequential:
    """Build a 3-layer MLP projector following REPA.

    Architecture::

        Linear(input_dim → projector_dim) → SiLU →
        Linear(projector_dim → projector_dim) → SiLU →
        Linear(projector_dim → output_dim)
    """
    return nn.Sequential(
        nn.Linear(input_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, output_dim),
    )


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _normalize(
    images: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> torch.Tensor:
    """Channel-wise normalize images with given mean and std.

    Parameters
    ----------
    images : Tensor (B, 3, H, W) in ``[0, 1]``.
    """
    device = images.device
    dtype = images.dtype
    m = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
    return (images - m) / s


def _to_01(images: torch.Tensor) -> torch.Tensor:
    """Convert images from ``[-1, 1]`` to ``[0, 1]``."""
    return images * 0.5 + 0.5


def _ensure_3ch(images: torch.Tensor) -> torch.Tensor:
    """Expand 1-channel images to 3 channels by repeating."""
    if images.shape[1] == 1:
        return images.repeat(1, 3, 1, 1)
    return images


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RepresentationAlignmentBase(ABC, nn.Module):
    """Abstract base for representation alignment modules.

    Provides a shared :meth:`compute_alignment_loss` that follows the REPA
    projection loss: L2-normalize both feature sets and compute mean
    negative cosine similarity.  The learnable projector MLP bridges
    between the model's feature dimension and the encoder's feature
    dimension.

    Sub-classes must implement :meth:`extract_features` and should call
    :meth:`_load_encoder` to populate ``self.encoder``.
    """

    def __init__(
        self,
        model_path: str,
        encoder_dim: int = 1024,
        projector_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.encoder_dim = encoder_dim
        self.projector_dim = projector_dim
        # The projector is created lazily on first call to
        # compute_alignment_loss so that it adapts to the model feature dim.
        self._projector: Optional[nn.Sequential] = None

    # -- projector management ------------------------------------------------

    def get_projector(self, model_feature_dim: int) -> nn.Sequential:
        """Return (or lazily create) the projector MLP."""
        if self._projector is None:
            self._projector = build_projector_mlp(
                model_feature_dim, self.projector_dim, self.encoder_dim,
            )
            # Register as submodule so parameters are visible
            self.add_module("projector", self._projector)
        return self._projector

    # -- abstract -----------------------------------------------------------

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
            Typically ``(B, D)`` for global features.
        """

    # -- shared alignment loss -----------------------------------------------

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the REPA-style alignment loss.

        The loss is the mean negative cosine similarity between
        projected model features and frozen encoder features, following
        the projection loss in ``vendor/REPA/loss.py``.

        Parameters
        ----------
        model_features : Tensor
            Features from the translation model.  Accepted shapes:

            * ``(B, D_model)`` – global features
            * ``(B, N, D_model)`` – per-token features
            * ``(B, D_model, H, W)`` – spatial feature maps
        encoder_features : Tensor
            Features from the frozen pre-trained encoder.  Accepted shapes:

            * ``(B, D_enc)`` – global features
            * ``(B, N, D_enc)`` – per-token features

        Returns
        -------
        Tensor
            Scalar alignment loss.
        """
        # -- pool model features to (B, D_model) ----------------------------
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])
        elif model_features.ndim == 3:
            model_features = model_features.mean(dim=1)

        # -- pool encoder features to (B, D_enc) ----------------------------
        if encoder_features.ndim == 3:
            encoder_features = encoder_features.mean(dim=1)

        # -- project model features to encoder space -------------------------
        projector = self.get_projector(model_features.shape[-1])
        projected = projector(model_features)  # (B, D_enc)

        # -- L2-normalize and compute negative cosine similarity -------------
        projected = F.normalize(projected, dim=-1)
        encoder_features = F.normalize(encoder_features.detach(), dim=-1)

        loss = -(projected * encoder_features).sum(dim=-1).mean()
        return loss


# ---------------------------------------------------------------------------
# DINOv3-sat alignment (SAR tasks: SAR2EO, SAR2IR, SAR2RGB)
# ---------------------------------------------------------------------------

class DINOv3SatAlignment(RepresentationAlignmentBase):
    """Representation alignment using a pre-trained DINOv3-sat encoder.

    Used for SAR2EO, SAR2IR, and SAR2RGB tasks.  The DINOv3-sat encoder
    processes the input SAR image and produces a feature representation;
    an alignment loss encourages the translation model to preserve the
    satellite-image semantics captured by DINOv3-sat.

    The encoder is a DINOv2-family ViT-L/16 pre-trained on satellite
    imagery.  Features are extracted as the mean of patch tokens from
    the last layer (global average of spatial tokens).

    Parameters
    ----------
    model_path : str
        Path to the DINOv3-sat checkpoint directory
        (e.g. ``./models/facebook/dinov3-vitl16-pretrain-sat493m``).
    encoder_dim : int
        Feature dimension of the encoder (1024 for ViT-L).
    projector_dim : int
        Hidden dimension of the projector MLP.
    """

    def __init__(
        self,
        model_path: str,
        encoder_dim: int = 1024,
        projector_dim: int = 2048,
    ) -> None:
        super().__init__(model_path, encoder_dim, projector_dim)
        self.encoder: Optional[nn.Module] = None
        self._load_encoder(model_path)

    def _load_encoder(self, model_path: str) -> None:
        """Load a DINOv2-family ViT encoder and freeze it."""
        try:
            import timm
            encoder = timm.create_model(
                "vit_large_patch16_224", pretrained=False, num_classes=0,
            )
            ckpt_path = model_path
            # Support both directory (containing pytorch_model.bin) and file
            import os
            if os.path.isdir(ckpt_path):
                for name in ("pytorch_model.bin", "model.safetensors", "dinov3.pth"):
                    candidate = os.path.join(ckpt_path, name)
                    if os.path.isfile(candidate):
                        ckpt_path = candidate
                        break
            if os.path.isfile(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                encoder.load_state_dict(state_dict, strict=False)
                logger.info("Loaded DINOv3-sat weights from %s", ckpt_path)
            else:
                logger.warning(
                    "DINOv3-sat checkpoint not found at %s; "
                    "using randomly-initialised encoder.",
                    model_path,
                )
            self.encoder = encoder
        except Exception as exc:
            logger.warning(
                "Failed to load DINOv3-sat encoder from %s: %s. "
                "Using randomly-initialised ViT-L/16 encoder.",
                model_path, exc,
            )
            import timm
            self.encoder = timm.create_model(
                "vit_large_patch16_224", pretrained=False, num_classes=0,
            )
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINOv3-sat features from input images.

        Parameters
        ----------
        images : Tensor (B, C, H, W)
            Input images in ``[-1, 1]``.

        Returns
        -------
        Tensor (B, encoder_dim)
            Global feature vector (mean-pooled patch tokens).
        """
        x = _to_01(images)
        x = _ensure_3ch(x)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = _normalize(x, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.encoder.eval()
        features = self.encoder(x)  # (B, D) with num_classes=0
        return features


# ---------------------------------------------------------------------------
# SARCLIP alignment (RGB2IR task)
# ---------------------------------------------------------------------------

class SARCLIPAlignment(RepresentationAlignmentBase):
    """Representation alignment using a pre-trained SARCLIP image encoder.

    Used for the RGB2IR task.  The SARCLIP encoder processes the input
    RGB image and produces a feature representation; an alignment loss
    encourages the translation model to preserve the visual semantics
    captured by SARCLIP.

    The encoder is a CLIP ViT-L/14 model fine-tuned for SAR imagery.
    Features are the image embedding (CLS token / pooled output).

    Parameters
    ----------
    model_path : str
        Path to the SARCLIP checkpoint directory
        (e.g. ``./models/BiliSakura/SARCLIP-ViT-L-14``).
    encoder_dim : int
        Feature dimension of the encoder (1024 for ViT-L).
    projector_dim : int
        Hidden dimension of the projector MLP.
    """

    def __init__(
        self,
        model_path: str,
        encoder_dim: int = 1024,
        projector_dim: int = 2048,
    ) -> None:
        super().__init__(model_path, encoder_dim, projector_dim)
        self.encoder: Optional[nn.Module] = None
        self._load_encoder(model_path)

    def _load_encoder(self, model_path: str) -> None:
        """Load a CLIP ViT-L/14 image encoder and freeze it."""
        try:
            import timm
            encoder = timm.create_model(
                "vit_large_patch14_clip_224", pretrained=False, num_classes=0,
            )
            ckpt_path = model_path
            import os
            if os.path.isdir(ckpt_path):
                for name in ("pytorch_model.bin", "model.safetensors",
                              "open_clip_pytorch_model.bin", "sarclip.pth"):
                    candidate = os.path.join(ckpt_path, name)
                    if os.path.isfile(candidate):
                        ckpt_path = candidate
                        break
            if os.path.isfile(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                encoder.load_state_dict(state_dict, strict=False)
                logger.info("Loaded SARCLIP weights from %s", ckpt_path)
            else:
                logger.warning(
                    "SARCLIP checkpoint not found at %s; "
                    "using randomly-initialised encoder.",
                    model_path,
                )
            self.encoder = encoder
        except Exception as exc:
            logger.warning(
                "Failed to load SARCLIP encoder from %s: %s. "
                "Using randomly-initialised ViT-L/14 encoder.",
                model_path, exc,
            )
            import timm
            self.encoder = timm.create_model(
                "vit_large_patch14_clip_224", pretrained=False, num_classes=0,
            )
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract SARCLIP features from input images.

        Parameters
        ----------
        images : Tensor (B, C, H, W)
            Input images in ``[-1, 1]``.

        Returns
        -------
        Tensor (B, encoder_dim)
            Global feature vector (CLS / pooled output).
        """
        x = _to_01(images)
        x = _ensure_3ch(x)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = _normalize(x, CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)
        self.encoder.eval()
        features = self.encoder(x)  # (B, D) with num_classes=0
        return features
