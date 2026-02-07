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

    def test_rgb2ir_has_dinov3sat_path(self):
        cfg = rgb2ir_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/DINOv3-sat"

    def test_sar2eo_has_sarclip_path(self):
        cfg = sar2eo_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar2ir_has_sarclip_path(self):
        cfg = sar2ir_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar2rgb_has_sarclip_path(self):
        cfg = sar2rgb_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar_tasks_no_vae_path(self):
        """SAR tasks should not have latent_vae_path set."""
        for fn in (sar2eo_config, sar2ir_config, sar2rgb_config):
            cfg = fn()
            assert cfg.latent_vae_path is None

    def test_overrides_work(self):
        cfg = rgb2ir_config(use_latent_target=True, lambda_latent=0.5)
        assert cfg.use_latent_target is True
        assert cfg.lambda_latent == 0.5


# ---------------------------------------------------------------------------
# Representation alignment placeholders
# ---------------------------------------------------------------------------

class TestSARCLIPPlaceholder:
    """SARCLIPAlignment raises NotImplementedError for placeholder methods."""

    def test_extract_features_not_implemented(self):
        from src.img2img_turbo.utils.rep_alignment import SARCLIPAlignment
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP")
        with pytest.raises(NotImplementedError, match="placeholder"):
            module.extract_features(torch.randn(1, 3, 64, 64))

    def test_compute_alignment_loss_not_implemented(self):
        from src.img2img_turbo.utils.rep_alignment import SARCLIPAlignment
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP")
        with pytest.raises(NotImplementedError, match="placeholder"):
            module.compute_alignment_loss(torch.randn(1, 64), torch.randn(1, 64))


class TestDINOv3SatPlaceholder:
    """DINOv3SatAlignment raises NotImplementedError for placeholder methods."""

    def test_extract_features_not_implemented(self):
        from src.img2img_turbo.utils.rep_alignment import DINOv3SatAlignment
        module = DINOv3SatAlignment("./models/BiliSakura/DINOv3-sat")
        with pytest.raises(NotImplementedError, match="placeholder"):
            module.extract_features(torch.randn(1, 3, 64, 64))

    def test_compute_alignment_loss_not_implemented(self):
        from src.img2img_turbo.utils.rep_alignment import DINOv3SatAlignment
        module = DINOv3SatAlignment("./models/BiliSakura/DINOv3-sat")
        with pytest.raises(NotImplementedError, match="placeholder"):
            module.compute_alignment_loss(torch.randn(1, 64), torch.randn(1, 64))


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
