"""Representation alignment via pre-trained image encoders.

This module implements representation alignment losses inspired by REPA
(REPresentation Alignment, see ``vendor/REPA``).  The technique is
architecture-agnostic: a frozen pre-trained encoder extracts features from
the **source** image, while a trainable projection head maps the
translation model's output features into the same embedding space.  A
negative-cosine-similarity loss encourages the model to preserve the
semantic content captured by the encoder.

Four concrete strategies are provided:

* **MaRS-RGB alignment** (default for RGB2IR) – A frozen MaRS-RGB SwinV2
  image encoder extracts features from the input RGB image.  Loaded via
  ``timm`` with ``swinv2_base_window8_256``.
  Checkpoint: ``models/WanderRainy/MaRS-RGB``.

* **MaRS-SAR alignment** (default for SAR2EO, SAR2IR, SAR2RGB) – A frozen
  MaRS-SAR SwinV2 image encoder extracts features from the input SAR
  image.  Loaded via ``timm`` with ``swinv2_base_window8_256``.
  Checkpoint: ``models/WanderRainy/MaRS-SAR``.

* **SARCLIP alignment** – for SAR2EO, SAR2IR, SAR2RGB tasks.  A frozen
  SARCLIP ViT-L/14 image encoder extracts features from the input SAR
  image.  Checkpoint: ``models/BiliSakura/SARCLIP-ViT-L-14``.

* **DINOv3-sat alignment** – for RGB2IR.  A frozen DINOv3-sat ViT-L
  encoder extracts features from the input RGB image.
  Checkpoint: ``models/facebook/dinov3-vitl16-pretrain-sat493m``.

The alignment loss follows REPA's formulation (negative cosine similarity
averaged over the batch) and uses a 3-layer MLP projection head identical
to the one in ``vendor/REPA/models/sit.py::build_mlp``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def build_projector(input_dim: int, projector_dim: int, output_dim: int) -> nn.Sequential:
    """Build a 3-layer MLP projection head."""
    return nn.Sequential(
        nn.Linear(input_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, output_dim),
    )


def normalize_to_01(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] range to [0, 1]."""
    return (images + 1.0) * 0.5


def adapt_channels(images: torch.Tensor) -> torch.Tensor:
    """Expand 1-channel images to 3 channels."""
    if images.shape[1] == 1:
        return images.repeat(1, 3, 1, 1)
    return images


class SARCLIPAlignment(nn.Module):
    """Representation alignment using SARCLIP encoder.
    
    Used for SAR2EO, SAR2IR, and SAR2RGB tasks.
    """

    def __init__(
        self,
        model_path: str = "./models/BiliSakura/SARCLIP-ViT-L-14",
        projector_dim: Optional[int] = None,
        encoder_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        from transformers import CLIPVisionModel, CLIPVisionModelWithProjection, CLIPImageProcessor
        
        # Load encoder (try with projection first)
        try:
            self.encoder = CLIPVisionModelWithProjection.from_pretrained(model_path, trust_remote_code=False)
        except Exception:
            self.encoder = CLIPVisionModel.from_pretrained(model_path, trust_remote_code=False)
        
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        
        # Load image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(model_path)
        
        # Get encoder_dim from model
        if encoder_dim is None:
            if hasattr(self.encoder.config, 'projection_dim'):
                self.encoder_dim = self.encoder.config.projection_dim
            elif hasattr(self.encoder, 'visual_projection'):
                self.encoder_dim = self.encoder.visual_projection.out_features
            else:
                self.encoder_dim = self.encoder.vision_model.config.hidden_size
        else:
            self.encoder_dim = encoder_dim
        
        self.projector_dim = projector_dim or (2 * self.encoder_dim)
        self.projector: Optional[nn.Module] = None
        
        logger.info("Loaded SARCLIP encoder from %s (encoder_dim=%d, projector_dim=%d)", 
                   model_path, self.encoder_dim, self.projector_dim)

    def build_projector(self, model_feature_dim: int) -> nn.Module:
        """Build the trainable projection head."""
        self.projector = build_projector(model_feature_dim, self.projector_dim, self.encoder_dim)
        return self.projector

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for SARCLIP."""
        import numpy as np
        from PIL import Image
        
        x = adapt_channels(normalize_to_01(images))
        pil_images = [Image.fromarray((x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) 
                      for i in range(x.shape[0])]
        return self.image_processor(pil_images, return_tensors="pt").pixel_values.to(images.device)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract SARCLIP features from images."""
        x = self._preprocess(images).to(next(self.encoder.parameters()).device)
        outputs = self.encoder(x)
        return outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs.last_hidden_state[:, 0]

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss using negative cosine similarity."""
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])  # global average pool
        if self.projector is not None:
            model_features = self.projector(model_features)
        z_model = F.normalize(model_features, dim=-1)
        z_enc = F.normalize(encoder_features.detach(), dim=-1)
        return -(z_model * z_enc).sum(dim=-1).mean()


class DINOv3SatAlignment(nn.Module):
    """Representation alignment using DINOv3-sat encoder.
    
    Used for RGB2IR task.
    """

    def __init__(
        self,
        model_path: str = "./models/facebook/dinov3-vitl16-pretrain-sat493m",
        projector_dim: Optional[int] = None,
        encoder_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor
        
        self.encoder = AutoModel.from_pretrained(model_path, trust_remote_code=False)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        self.encoder_dim = encoder_dim or self.encoder.config.hidden_size
        self.projector_dim = projector_dim or (2 * self.encoder_dim)
        self.projector: Optional[nn.Module] = None
        
        logger.info("Loaded DINOv3-sat encoder from %s (encoder_dim=%d, projector_dim=%d)", 
                   model_path, self.encoder_dim, self.projector_dim)

    def build_projector(self, model_feature_dim: int) -> nn.Module:
        """Build the trainable projection head."""
        self.projector = build_projector(model_feature_dim, self.projector_dim, self.encoder_dim)
        return self.projector

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for DINOv3."""
        import numpy as np
        from PIL import Image
        
        x = adapt_channels(normalize_to_01(images))
        pil_images = [Image.fromarray((x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) 
                      for i in range(x.shape[0])]
        inputs = self.image_processor(images=pil_images, return_tensors="pt")
        return inputs.pixel_values.to(images.device)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract DINOv3-sat features from images."""
        x = self._preprocess(images).to(next(self.encoder.parameters()).device)
        outputs = self.encoder(x)
        # Use pooler_output if available (as per official README), otherwise CLS token
        return outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss using negative cosine similarity."""
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])  # global average pool
        if self.projector is not None:
            model_features = self.projector(model_features)
        z_model = F.normalize(model_features, dim=-1)
        z_enc = F.normalize(encoder_features.detach(), dim=-1)
        return -(z_model * z_enc).sum(dim=-1).mean()


class MaRSRGBAlignment(nn.Module):
    """Representation alignment using MaRS-RGB encoder (SwinV2 backbone).

    Default encoder for the RGB2IR task.  Uses ``timm`` to load a
    ``swinv2_base_window8_256`` model with pre-trained MaRS-RGB weights.
    """

    def __init__(
        self,
        model_path: str = "./models/WanderRainy/MaRS-RGB",
        projector_dim: Optional[int] = None,
        encoder_dim: Optional[int] = None,
        timm_model_name: str = "swinv2_base_window8_256",
        img_size: int = 512,
    ) -> None:
        super().__init__()
        import timm as _timm

        self.encoder = _timm.create_model(
            timm_model_name,
            pretrained=False,
            features_only=True,
            in_chans=3,
            img_size=img_size,
            checkpoint_path=model_path,
        )
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        # SwinV2-Base last-stage feature dim: embed_dim * 2^3 = 128 * 8 = 1024
        self.encoder_dim = encoder_dim or 1024
        self.projector_dim = projector_dim or (2 * self.encoder_dim)
        self.projector: Optional[nn.Module] = None

        logger.info(
            "Loaded MaRS-RGB encoder from %s (encoder_dim=%d, projector_dim=%d)",
            model_path, self.encoder_dim, self.projector_dim,
        )

    def build_projector(self, model_feature_dim: int) -> nn.Module:
        """Build the trainable projection head."""
        self.projector = build_projector(model_feature_dim, self.projector_dim, self.encoder_dim)
        return self.projector

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract MaRS-RGB features (last-stage, global-average-pooled)."""
        x = adapt_channels(normalize_to_01(images))
        x = x.to(next(self.encoder.parameters()).device)
        feats = self.encoder(x)  # list of multi-scale feature maps
        last_feat = feats[-1]  # deepest stage: (B, C, H, W)
        return last_feat.mean(dim=[2, 3])  # GAP → (B, C)

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss using negative cosine similarity."""
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])
        if self.projector is not None:
            model_features = self.projector(model_features)
        z_model = F.normalize(model_features, dim=-1)
        z_enc = F.normalize(encoder_features.detach(), dim=-1)
        return -(z_model * z_enc).sum(dim=-1).mean()


class MaRSSARAlignment(nn.Module):
    """Representation alignment using MaRS-SAR encoder (SwinV2 backbone).

    Default encoder for SAR2EO, SAR2IR, and SAR2RGB tasks.  Uses ``timm``
    to load a ``swinv2_base_window8_256`` model with pre-trained MaRS-SAR
    weights.
    """

    def __init__(
        self,
        model_path: str = "./models/WanderRainy/MaRS-SAR",
        projector_dim: Optional[int] = None,
        encoder_dim: Optional[int] = None,
        timm_model_name: str = "swinv2_base_window8_256",
        img_size: int = 512,
    ) -> None:
        super().__init__()
        import timm as _timm

        self.encoder = _timm.create_model(
            timm_model_name,
            pretrained=False,
            features_only=True,
            in_chans=1,
            img_size=img_size,
            checkpoint_path=model_path,
        )
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        # SwinV2-Base last-stage feature dim: embed_dim * 2^3 = 128 * 8 = 1024
        self.encoder_dim = encoder_dim or 1024
        self.projector_dim = projector_dim or (2 * self.encoder_dim)
        self.projector: Optional[nn.Module] = None

        logger.info(
            "Loaded MaRS-SAR encoder from %s (encoder_dim=%d, projector_dim=%d)",
            model_path, self.encoder_dim, self.projector_dim,
        )

    def build_projector(self, model_feature_dim: int) -> nn.Module:
        """Build the trainable projection head."""
        self.projector = build_projector(model_feature_dim, self.projector_dim, self.encoder_dim)
        return self.projector

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract MaRS-SAR features (last-stage, global-average-pooled)."""
        x = normalize_to_01(images)
        # Keep 1-channel for SAR; do NOT expand to 3-ch
        if x.shape[1] == 3:
            logger.warning("MaRS-SAR encoder received 3-channel input; using first channel only")
            x = x[:, :1]  # take first channel if RGB passed by mistake
        x = x.to(next(self.encoder.parameters()).device)
        feats = self.encoder(x)  # list of multi-scale feature maps
        last_feat = feats[-1]  # deepest stage: (B, C, H, W)
        return last_feat.mean(dim=[2, 3])  # GAP → (B, C)

    def compute_alignment_loss(
        self,
        model_features: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss using negative cosine similarity."""
        if model_features.ndim == 4:
            model_features = model_features.mean(dim=[2, 3])
        if self.projector is not None:
            model_features = self.projector(model_features)
        z_model = F.normalize(model_features, dim=-1)
        z_enc = F.normalize(encoder_features.detach(), dim=-1)
        return -(z_model * z_enc).sum(dim=-1).mean()
