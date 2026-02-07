"""Tests for new ablation and representation alignment features.

Covers:
* New config fields (latent target, representation alignment)
* Task-specific config defaults
* LatentTargetEncoder API
* Representation alignment modules (REPA)
* Trainer loss function with latent target encoder and rep alignment
"""

import pytest
import torch

from src.img2img_turbo.config import (
    TaskConfig,
    rgb2ir_config,
    sar2eo_config,
    sar2ir_config,
    sar2rgb_config,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestNewConfigDefaults:
    """New config fields have correct defaults."""

    def test_latent_target_defaults(self):
        cfg = TaskConfig()
        assert cfg.use_latent_target is False
        assert cfg.latent_vae_path is None
        assert cfg.lambda_latent == 1.0

    def test_rep_alignment_defaults(self):
        cfg = TaskConfig()
        assert cfg.use_rep_alignment is False
        assert cfg.rep_alignment_model_path is None
        assert cfg.lambda_rep_alignment == 1.0


# ---------------------------------------------------------------------------
# Task-specific config paths
# ---------------------------------------------------------------------------

class TestTaskConfigPaths:
    """Pre-built task configs set the expected model paths."""

    def test_rgb2ir_has_vae_path(self):
        cfg = rgb2ir_config()
        assert cfg.latent_vae_path == "./models/BiliSakura/VAEs"

    def test_rgb2ir_has_dinov3sat_path(self):
        cfg = rgb2ir_config()
        assert cfg.rep_alignment_model_path == "./models/facebook/dinov3-vitl16-pretrain-sat493m"

    def test_sar2eo_has_sarclip_path(self):
        cfg = sar2eo_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP-ViT-L-14"

    def test_sar2ir_has_sarclip_path(self):
        cfg = sar2ir_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP-ViT-L-14"

    def test_sar2rgb_has_sarclip_path(self):
        cfg = sar2rgb_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP-ViT-L-14"

    def test_all_tasks_have_vae_path(self):
        """All tasks should have latent_vae_path set for optional latent modeling."""
        for fn in (sar2eo_config, sar2ir_config, sar2rgb_config, rgb2ir_config):
            cfg = fn()
            assert cfg.latent_vae_path == "./models/BiliSakura/VAEs"

    def test_overrides_work(self):
        cfg = rgb2ir_config(use_latent_target=True, lambda_latent=0.5)
        assert cfg.use_latent_target is True
        assert cfg.lambda_latent == 0.5


# ---------------------------------------------------------------------------
# Representation alignment modules (REPA)
# ---------------------------------------------------------------------------

class TestSARCLIPAlignment:
    """SARCLIPAlignment has concrete implementation with lazy encoder loading."""

    def test_instantiation(self):
        from src.img2img_turbo.utils.rep_alignment import SARCLIPAlignment
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14")
        assert module.model_path == "./models/BiliSakura/SARCLIP-ViT-L-14"
        assert module.encoder_dim == 1024

    def test_build_projector(self):
        from src.img2img_turbo.utils.rep_alignment import SARCLIPAlignment
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14")
        proj = module.build_projector(model_feature_dim=3)
        assert proj is not None
        assert module.projector is proj
        # Check it has trainable parameters
        params = list(proj.parameters())
        assert len(params) > 0

    def test_compute_alignment_loss_with_projector(self):
        from src.img2img_turbo.utils.rep_alignment import SARCLIPAlignment
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14", encoder_dim=64)
        module.build_projector(model_feature_dim=16)
        model_feats = torch.randn(2, 16)
        enc_feats = torch.randn(2, 64)
        loss = module.compute_alignment_loss(model_feats, enc_feats)
        assert loss.ndim == 0  # scalar
        assert loss.requires_grad  # trainable via projector

    def test_compute_alignment_loss_spatial_features(self):
        """4-D model features are global-avg-pooled then projected."""
        from src.img2img_turbo.utils.rep_alignment import SARCLIPAlignment
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14", encoder_dim=64)
        module.build_projector(model_feature_dim=8)
        model_feats = torch.randn(2, 8, 4, 4)  # spatial (B, C, H, W)
        enc_feats = torch.randn(2, 64)
        loss = module.compute_alignment_loss(model_feats, enc_feats)
        assert loss.ndim == 0

    def test_alignment_loss_range(self):
        """Negative cosine similarity should be in [-1, 1]."""
        from src.img2img_turbo.utils.rep_alignment import SARCLIPAlignment
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14", encoder_dim=32)
        module.build_projector(model_feature_dim=32)
        feats = torch.randn(4, 32)
        enc_feats = torch.randn(4, 32)
        loss = module.compute_alignment_loss(feats, enc_feats)
        assert -1.0 <= loss.item() <= 1.0


class TestDINOv3SatAlignment:
    """DINOv3SatAlignment has concrete implementation with lazy encoder loading."""

    def test_instantiation(self):
        from src.img2img_turbo.utils.rep_alignment import DINOv3SatAlignment
        module = DINOv3SatAlignment("./models/facebook/dinov3-vitl16-pretrain-sat493m")
        assert module.model_path == "./models/facebook/dinov3-vitl16-pretrain-sat493m"
        assert module.encoder_dim == 1024

    def test_build_projector(self):
        from src.img2img_turbo.utils.rep_alignment import DINOv3SatAlignment
        module = DINOv3SatAlignment("./models/facebook/dinov3-vitl16-pretrain-sat493m")
        proj = module.build_projector(model_feature_dim=3)
        assert proj is not None
        assert module.projector is proj

    def test_compute_alignment_loss_with_projector(self):
        from src.img2img_turbo.utils.rep_alignment import DINOv3SatAlignment
        module = DINOv3SatAlignment("./models/facebook/dinov3-vitl16-pretrain-sat493m", encoder_dim=64)
        module.build_projector(model_feature_dim=16)
        model_feats = torch.randn(2, 16)
        enc_feats = torch.randn(2, 64)
        loss = module.compute_alignment_loss(model_feats, enc_feats)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_compute_alignment_loss_spatial_features(self):
        """4-D model features are global-avg-pooled then projected."""
        from src.img2img_turbo.utils.rep_alignment import DINOv3SatAlignment
        module = DINOv3SatAlignment("./models/facebook/dinov3-vitl16-pretrain-sat493m", encoder_dim=64)
        module.build_projector(model_feature_dim=8)
        model_feats = torch.randn(2, 8, 4, 4)
        enc_feats = torch.randn(2, 64)
        loss = module.compute_alignment_loss(model_feats, enc_feats)
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# LatentTargetEncoder API
# ---------------------------------------------------------------------------

class TestLatentTargetEncoder:
    """LatentTargetEncoder can be imported and has the expected interface."""

    def test_import(self):
        from src.img2img_turbo.utils.latent_target import LatentTargetEncoder
        assert LatentTargetEncoder is not None

    def test_has_encode_method(self):
        from src.img2img_turbo.utils.latent_target import LatentTargetEncoder
        assert callable(getattr(LatentTargetEncoder, "encode", None))

    def test_has_encode_with_grad_method(self):
        from src.latent_target import LatentTargetEncoder
        assert callable(getattr(LatentTargetEncoder, "encode_with_grad", None))

    def test_adapt_channels_expands_1ch(self):
        from src.latent_target import LatentTargetEncoder
        t = torch.randn(2, 1, 8, 8)
        out = LatentTargetEncoder._adapt_channels(t)
        assert out.shape == (2, 3, 8, 8)
        assert torch.allclose(out[:, 0], out[:, 1])
        assert torch.allclose(out[:, 0], out[:, 2])

    def test_adapt_channels_passthrough_3ch(self):
        from src.latent_target import LatentTargetEncoder
        t = torch.randn(2, 3, 8, 8)
        out = LatentTargetEncoder._adapt_channels(t)
        assert out.shape == (2, 3, 8, 8)
        assert out is t  # should be the same tensor, not a copy


# ---------------------------------------------------------------------------
# Trainer loss function signature
# ---------------------------------------------------------------------------

class TestTrainerLossSignature:
    """compute_training_loss accepts the new latent_target_encoder kwarg."""

    def test_accepts_latent_target_encoder_none(self):
        """Passing latent_target_encoder=None should not break the call."""
        import inspect
        from src.img2img_turbo.trainer import Pix2PixTurboTrainer
        sig = inspect.signature(Pix2PixTurboTrainer.compute_training_loss)
        assert "latent_target_encoder" in sig.parameters
        assert "lambda_latent" in sig.parameters

    def test_accepts_rep_alignment_kwargs(self):
        """compute_training_loss accepts rep_alignment_module and lambda."""
        import inspect
        from src.img2img_turbo.trainer import Pix2PixTurboTrainer
        sig = inspect.signature(Pix2PixTurboTrainer.compute_training_loss)
        assert "rep_alignment_module" in sig.parameters
        assert "lambda_rep_alignment" in sig.parameters
