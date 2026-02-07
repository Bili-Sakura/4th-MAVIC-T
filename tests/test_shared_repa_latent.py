"""Tests for shared REPA and latent target support across all baselines.

Covers:
* Shared module imports (src.rep_alignment, src.latent_target)
* New config fields in CUT and DDBM baselines
* Task-specific config defaults for CUT and DDBM
* CUT and DDBM trainer loss signatures accept new kwargs
"""

import inspect

import pytest
import torch

from src.rep_alignment import (
    RepresentationAlignmentBase,
    SARCLIPAlignment,
    DINOv3SatAlignment,
)
from src.latent_target import LatentTargetEncoder

from src.cut_baseline.config import (
    TaskConfig as CutConfig,
    sar2eo_config as cut_sar2eo,
    rgb2ir_config as cut_rgb2ir,
    sar2ir_config as cut_sar2ir,
    sar2rgb_config as cut_sar2rgb,
)
from src.ddbm_baseline.config import (
    TaskConfig as DdbmConfig,
    sar2eo_config as ddbm_sar2eo,
    rgb2ir_config as ddbm_rgb2ir,
    sar2ir_config as ddbm_sar2ir,
    sar2rgb_config as ddbm_sar2rgb,
)


# ---------------------------------------------------------------------------
# Shared module imports
# ---------------------------------------------------------------------------

class TestSharedImports:
    """REPA and latent modules are importable from the shared src level."""

    def test_rep_alignment_from_shared(self):
        assert RepresentationAlignmentBase is not None
        assert SARCLIPAlignment is not None
        assert DINOv3SatAlignment is not None

    def test_latent_target_from_shared(self):
        assert LatentTargetEncoder is not None
        assert callable(getattr(LatentTargetEncoder, "encode", None))

    def test_backward_compat_turbo_utils(self):
        """Old import path still works via re-export."""
        from src.img2img_turbo.utils.rep_alignment import SARCLIPAlignment as SA
        from src.img2img_turbo.utils.latent_target import LatentTargetEncoder as LTE
        assert SA is SARCLIPAlignment
        assert LTE is LatentTargetEncoder


# ---------------------------------------------------------------------------
# CUT config defaults
# ---------------------------------------------------------------------------

class TestCUTConfigDefaults:
    """CUT config has latent and REPA fields with correct defaults."""

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

    def test_overrides_work(self):
        cfg = cut_rgb2ir(use_latent_target=True, lambda_latent=0.5)
        assert cfg.use_latent_target is True
        assert cfg.lambda_latent == 0.5


# ---------------------------------------------------------------------------
# DDBM config defaults
# ---------------------------------------------------------------------------

class TestDDBMConfigDefaults:
    """DDBM config has latent and REPA fields with correct defaults."""

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

    def test_overrides_work(self):
        cfg = ddbm_rgb2ir(use_latent_target=True, lambda_latent=0.5)
        assert cfg.use_latent_target is True
        assert cfg.lambda_latent == 0.5


# ---------------------------------------------------------------------------
# CUT task-specific config paths
# ---------------------------------------------------------------------------

class TestCUTTaskConfigPaths:
    """CUT pre-built task configs set the expected model paths."""

    def test_rgb2ir_has_vae_path(self):
        cfg = cut_rgb2ir()
        assert cfg.latent_vae_path == "./models/BiliSakura/VAEs"

    def test_rgb2ir_has_dinov3sat_path(self):
        cfg = cut_rgb2ir()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/DINOv3-sat"

    def test_sar2eo_has_sarclip_path(self):
        cfg = cut_sar2eo()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar2ir_has_sarclip_path(self):
        cfg = cut_sar2ir()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar2rgb_has_sarclip_path(self):
        cfg = cut_sar2rgb()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar_tasks_no_vae_path(self):
        for fn in (cut_sar2eo, cut_sar2ir, cut_sar2rgb):
            cfg = fn()
            assert cfg.latent_vae_path is None, f"{fn.__name__} should not set latent_vae_path"


# ---------------------------------------------------------------------------
# DDBM task-specific config paths
# ---------------------------------------------------------------------------

class TestDDBMTaskConfigPaths:
    """DDBM pre-built task configs set the expected model paths."""

    def test_rgb2ir_has_vae_path(self):
        cfg = ddbm_rgb2ir()
        assert cfg.latent_vae_path == "./models/BiliSakura/VAEs"

    def test_rgb2ir_has_dinov3sat_path(self):
        cfg = ddbm_rgb2ir()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/DINOv3-sat"

    def test_sar2eo_has_sarclip_path(self):
        cfg = ddbm_sar2eo()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar2ir_has_sarclip_path(self):
        cfg = ddbm_sar2ir()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar2rgb_has_sarclip_path(self):
        cfg = ddbm_sar2rgb()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/SARCLIP"

    def test_sar_tasks_no_vae_path(self):
        for fn in (ddbm_sar2eo, ddbm_sar2ir, ddbm_sar2rgb):
            cfg = fn()
            assert cfg.latent_vae_path is None, f"{fn.__name__} should not set latent_vae_path"


# ---------------------------------------------------------------------------
# REPA placeholders work from shared import
# ---------------------------------------------------------------------------

class TestSharedREPAPlaceholders:
    """REPA placeholders raise NotImplementedError when imported from src."""

    def test_sarclip_extract_features(self):
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP")
        with pytest.raises(NotImplementedError, match="placeholder"):
            module.extract_features(torch.randn(1, 3, 64, 64))

    def test_dinov3sat_compute_alignment_loss(self):
        module = DINOv3SatAlignment("./models/BiliSakura/DINOv3-sat")
        with pytest.raises(NotImplementedError, match="placeholder"):
            module.compute_alignment_loss(torch.randn(1, 64), torch.randn(1, 64))


# ---------------------------------------------------------------------------
# CUT trainer loss signature
# ---------------------------------------------------------------------------

class TestCUTTrainerLossSignature:
    """CUT compute_G_loss accepts the new latent_target_encoder kwarg."""

    def test_accepts_latent_kwargs(self):
        from src.cut_baseline.trainer import CUTTrainer
        sig = inspect.signature(CUTTrainer.compute_G_loss)
        assert "latent_target_encoder" in sig.parameters
        assert "lambda_latent" in sig.parameters


# ---------------------------------------------------------------------------
# DDBM trainer loss signature
# ---------------------------------------------------------------------------

class TestDDBMTrainerLossSignature:
    """DDBM compute_training_loss accepts the new latent_target_encoder kwarg."""

    def test_accepts_latent_kwargs(self):
        from src.ddbm_baseline.trainer import DDBMTrainer
        sig = inspect.signature(DDBMTrainer.compute_training_loss)
        assert "latent_target_encoder" in sig.parameters
        assert "lambda_latent" in sig.parameters
