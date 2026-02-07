"""Tests for cross-baseline latent target and representation alignment support.

Covers:
* New config fields in DDBM and CUT baselines (latent target, rep alignment)
* Task-specific config defaults for all baselines
* Trainer loss function signatures accept latent_target_encoder kwarg
"""

import inspect

import pytest

from src.ddbm_baseline.config import TaskConfig as DdbmConfig
from src.ddbm_baseline.config import (
    rgb2ir_config as ddbm_rgb2ir_config,
    sar2eo_config as ddbm_sar2eo_config,
    sar2ir_config as ddbm_sar2ir_config,
    sar2rgb_config as ddbm_sar2rgb_config,
)
from src.cut_baseline.config import TaskConfig as CutConfig
from src.cut_baseline.config import (
    rgb2ir_config as cut_rgb2ir_config,
    sar2eo_config as cut_sar2eo_config,
    sar2ir_config as cut_sar2ir_config,
    sar2rgb_config as cut_sar2rgb_config,
)


# ---------------------------------------------------------------------------
# Config defaults – DDBM
# ---------------------------------------------------------------------------

class TestDDBMConfigDefaults:
    """DDBM TaskConfig has latent target and rep alignment fields."""

    def test_latent_target_defaults(self):
        cfg = DdbmConfig()
        assert cfg.use_latent_target is False
        assert cfg.latent_vae_path is None
        assert cfg.lambda_latent == 1.0

    def test_rep_alignment_defaults(self):
        cfg = DdbmConfig()
        assert cfg.use_rep_alignment is False
        assert cfg.rep_alignment_model_path is None
        assert cfg.lambda_rep_alignment == 1.0


# ---------------------------------------------------------------------------
# Config defaults – CUT
# ---------------------------------------------------------------------------

class TestCUTConfigDefaults:
    """CUT TaskConfig has latent target and rep alignment fields."""

    def test_latent_target_defaults(self):
        cfg = CutConfig()
        assert cfg.use_latent_target is False
        assert cfg.latent_vae_path is None
        assert cfg.lambda_latent == 1.0

    def test_rep_alignment_defaults(self):
        cfg = CutConfig()
        assert cfg.use_rep_alignment is False
        assert cfg.rep_alignment_model_path is None
        assert cfg.lambda_rep_alignment == 1.0


# ---------------------------------------------------------------------------
# Task-specific config paths – DDBM
# ---------------------------------------------------------------------------

class TestDDBMTaskConfigPaths:
    """DDBM pre-built task configs set the expected model paths."""

    def test_rgb2ir_has_vae_path(self):
        cfg = ddbm_rgb2ir_config()
        assert cfg.latent_vae_path == "./models/BiliSakura/VAEs"

    def test_rgb2ir_has_dinov3sat_path(self):
        cfg = ddbm_rgb2ir_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/DINOv3-sat"

    def test_sar2eo_has_sarclip_path(self):
        cfg = ddbm_sar2eo_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar2ir_has_sarclip_path(self):
        cfg = ddbm_sar2ir_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar2rgb_has_sarclip_path(self):
        cfg = ddbm_sar2rgb_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar_tasks_no_vae_path(self):
        for fn in (ddbm_sar2eo_config, ddbm_sar2ir_config, ddbm_sar2rgb_config):
            cfg = fn()
            assert cfg.latent_vae_path is None

    def test_overrides_work(self):
        cfg = ddbm_rgb2ir_config(use_latent_target=True, lambda_latent=0.5)
        assert cfg.use_latent_target is True
        assert cfg.lambda_latent == 0.5


# ---------------------------------------------------------------------------
# Task-specific config paths – CUT
# ---------------------------------------------------------------------------

class TestCUTTaskConfigPaths:
    """CUT pre-built task configs set the expected model paths."""

    def test_rgb2ir_has_vae_path(self):
        cfg = cut_rgb2ir_config()
        assert cfg.latent_vae_path == "./models/BiliSakura/VAEs"

    def test_rgb2ir_has_dinov3sat_path(self):
        cfg = cut_rgb2ir_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/DINOv3-sat"

    def test_sar2eo_has_sarclip_path(self):
        cfg = cut_sar2eo_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar2ir_has_sarclip_path(self):
        cfg = cut_sar2ir_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar2rgb_has_sarclip_path(self):
        cfg = cut_sar2rgb_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar_tasks_no_vae_path(self):
        for fn in (cut_sar2eo_config, cut_sar2ir_config, cut_sar2rgb_config):
            cfg = fn()
            assert cfg.latent_vae_path is None

    def test_overrides_work(self):
        cfg = cut_rgb2ir_config(use_latent_target=True, lambda_latent=0.5)
        assert cfg.use_latent_target is True
        assert cfg.lambda_latent == 0.5


# ---------------------------------------------------------------------------
# Trainer loss function signatures
# ---------------------------------------------------------------------------

class TestDDBMTrainerLossSignature:
    """DDBM compute_training_loss accepts the new latent_target_encoder kwarg."""

    def test_accepts_latent_target_encoder(self):
        from src.ddbm_baseline.trainer import DDBMTrainer
        sig = inspect.signature(DDBMTrainer.compute_training_loss)
        assert "latent_target_encoder" in sig.parameters
        assert "lambda_latent" in sig.parameters


class TestCUTTrainerLossSignature:
    """CUT compute_G_loss accepts the new latent_target_encoder kwarg."""

    def test_accepts_latent_target_encoder(self):
        from src.cut_baseline.trainer import CUTTrainer
        sig = inspect.signature(CUTTrainer.compute_G_loss)
        assert "latent_target_encoder" in sig.parameters
        assert "lambda_latent" in sig.parameters
