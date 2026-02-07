"""Tests for new ablation and representation alignment features.

Covers:
* New config fields (latent target, representation alignment)
* Task-specific config defaults
* LatentTargetEncoder API
* Representation alignment placeholder modules
* Trainer loss function with latent target encoder
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

    def test_rgb2ir_has_sarclip_path(self):
        cfg = rgb2ir_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP-ViT-L-14"

    def test_sar2eo_has_dinov3sat_path(self):
        cfg = sar2eo_config()
        assert cfg.rep_alignment_model_path == "./models/facebook/dinov3-vitl16-pretrain-sat493m"

    def test_sar2ir_has_dinov3sat_path(self):
        cfg = sar2ir_config()
        assert cfg.rep_alignment_model_path == "./models/facebook/dinov3-vitl16-pretrain-sat493m"

    def test_sar2rgb_has_dinov3sat_path(self):
        cfg = sar2rgb_config()
        assert cfg.rep_alignment_model_path == "./models/facebook/dinov3-vitl16-pretrain-sat493m"

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
# Representation alignment placeholders
# ---------------------------------------------------------------------------

class TestSARCLIPImplementation:
    """SARCLIPAlignment now has working implementations."""

    def test_extract_features_returns_tensor(self):
        from src.img2img_turbo.utils.rep_alignment import SARCLIPAlignment
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14")
        features = module.extract_features(torch.randn(1, 3, 64, 64))
        assert isinstance(features, torch.Tensor)
        assert features.ndim == 2
        assert features.shape[0] == 1

    def test_compute_alignment_loss_returns_scalar(self):
        from src.img2img_turbo.utils.rep_alignment import SARCLIPAlignment
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14")
        model_feat = torch.randn(1, 128)
        enc_feat = torch.randn(1, module.encoder_dim)
        loss = module.compute_alignment_loss(model_feat, enc_feat)
        assert loss.ndim == 0


class TestDINOv3SatImplementation:
    """DINOv3SatAlignment now has working implementations."""

    def test_extract_features_returns_tensor(self):
        from src.img2img_turbo.utils.rep_alignment import DINOv3SatAlignment
        module = DINOv3SatAlignment("./models/facebook/dinov3-vitl16-pretrain-sat493m")
        features = module.extract_features(torch.randn(1, 3, 64, 64))
        assert isinstance(features, torch.Tensor)
        assert features.ndim == 2
        assert features.shape[0] == 1

    def test_compute_alignment_loss_returns_scalar(self):
        from src.img2img_turbo.utils.rep_alignment import DINOv3SatAlignment
        module = DINOv3SatAlignment("./models/facebook/dinov3-vitl16-pretrain-sat493m")
        model_feat = torch.randn(1, 128)
        enc_feat = torch.randn(1, module.encoder_dim)
        loss = module.compute_alignment_loss(model_feat, enc_feat)
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
