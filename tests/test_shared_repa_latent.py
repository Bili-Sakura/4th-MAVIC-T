"""Tests for shared REPA and latent target support across all baselines.

Covers:
* Shared module imports (src.rep_alignment, src.latent_target)
* New config fields in CUT and DDBM baselines
* Task-specific config defaults for CUT and DDBM
* Representation alignment modules (concrete implementations)
* CUT and DDBM trainer loss signatures accept new kwargs
"""

import inspect

import pytest
import torch

from src.utils.rep_alignment import (
    SARCLIPAlignment,
    DINOv3SatAlignment,
    MaRSRGBAlignment,
    MaRSSARAlignment,
)
from src.utils.latent_target import LatentTargetEncoder

from examples.cut.config import (
    TaskConfig as CutConfig,
    sar2eo_config as cut_sar2eo,
    rgb2ir_config as cut_rgb2ir,
    sar2ir_config as cut_sar2ir,
    sar2rgb_config as cut_sar2rgb,
)
from examples.ddbm.config import (
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
        assert SARCLIPAlignment is not None
        assert DINOv3SatAlignment is not None
        assert MaRSRGBAlignment is not None
        assert MaRSSARAlignment is not None

    def test_latent_target_from_shared(self):
        assert LatentTargetEncoder is not None
        assert callable(getattr(LatentTargetEncoder, "encode", None))

    def test_latent_target_encode_with_grad(self):
        assert callable(getattr(LatentTargetEncoder, "encode_with_grad", None))

    def test_latent_target_adapt_channels(self):
        """1-channel images are expanded to 3 channels for the VAE."""
        t = torch.randn(2, 1, 8, 8)
        out = LatentTargetEncoder._adapt_channels(t)
        assert out.shape == (2, 3, 8, 8)
        assert torch.allclose(out[:, 0], out[:, 1])

    def test_latent_target_adapt_channels_3ch_passthrough(self):
        """3-channel images pass through without modification."""
        t = torch.randn(2, 3, 8, 8)
        out = LatentTargetEncoder._adapt_channels(t)
        assert out is t

    def test_backward_compat_turbo_utils(self):
        """Old import path still works via re-export."""
        from src.utils.rep_alignment import SARCLIPAlignment as SA
        from src.utils.latent_target import LatentTargetEncoder as LTE
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

    def test_rgb2ir_has_mars_rgb_path(self):
        cfg = cut_rgb2ir()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-RGB"

    def test_sar2eo_has_mars_sar_path(self):
        cfg = cut_sar2eo()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-SAR"

    def test_sar2ir_has_mars_sar_path(self):
        cfg = cut_sar2ir()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-SAR"

    def test_sar2rgb_has_mars_sar_path(self):
        cfg = cut_sar2rgb()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-SAR"

    def test_all_tasks_have_vae_path(self):
        """All CUT tasks should have latent_vae_path set for optional latent modeling."""
        for fn in (cut_sar2eo, cut_sar2ir, cut_sar2rgb, cut_rgb2ir):
            cfg = fn()
            assert cfg.latent_vae_path == "./models/BiliSakura/VAEs", (
                f"{fn.__name__} should set latent_vae_path"
            )


# ---------------------------------------------------------------------------
# DDBM task-specific config paths
# ---------------------------------------------------------------------------

class TestDDBMTaskConfigPaths:
    """DDBM pre-built task configs set the expected model paths."""

    def test_rgb2ir_has_vae_path(self):
        cfg = ddbm_rgb2ir()
        assert cfg.latent_vae_path == "./models/BiliSakura/VAEs"

    def test_rgb2ir_has_mars_rgb_path(self):
        cfg = ddbm_rgb2ir()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-RGB"

    def test_sar2eo_has_mars_sar_path(self):
        cfg = ddbm_sar2eo()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-SAR"

    def test_sar2ir_has_mars_sar_path(self):
        cfg = ddbm_sar2ir()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-SAR"

    def test_sar2rgb_has_mars_sar_path(self):
        cfg = ddbm_sar2rgb()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-SAR"

    def test_all_tasks_have_vae_path(self):
        """All DDBM tasks should have latent_vae_path set for optional latent modeling."""
        for fn in (ddbm_sar2eo, ddbm_sar2ir, ddbm_sar2rgb, ddbm_rgb2ir):
            cfg = fn()
            assert cfg.latent_vae_path == "./models/BiliSakura/VAEs", (
                f"{fn.__name__} should set latent_vae_path"
            )


# ---------------------------------------------------------------------------
# REPA concrete implementations from shared import
# ---------------------------------------------------------------------------

class TestSharedREPAImplementation:
    """REPA modules are concrete and compute alignment losses."""

    def test_sarclip_instantiation(self):
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14")
        assert module.model_path == "./models/BiliSakura/SARCLIP-ViT-L-14"
        assert module.encoder_dim == 1024

    def test_sarclip_build_projector(self):
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14", encoder_dim=64)
        proj = module.build_projector(model_feature_dim=16)
        assert proj is not None
        assert len(list(proj.parameters())) > 0

    def test_sarclip_alignment_loss(self):
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14", encoder_dim=32)
        module.build_projector(model_feature_dim=16)
        model_feats = torch.randn(4, 16)
        enc_feats = torch.randn(4, 32)
        loss = module.compute_alignment_loss(model_feats, enc_feats)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_dinov3sat_instantiation(self):
        module = DINOv3SatAlignment("./models/facebook/dinov3-vitl16-pretrain-sat493m")
        assert module.model_path == "./models/facebook/dinov3-vitl16-pretrain-sat493m"
        assert module.encoder_dim == 1024

    def test_dinov3sat_build_projector(self):
        module = DINOv3SatAlignment("./models/facebook/dinov3-vitl16-pretrain-sat493m", encoder_dim=64)
        proj = module.build_projector(model_feature_dim=16)
        assert proj is not None
        assert len(list(proj.parameters())) > 0

    def test_dinov3sat_alignment_loss(self):
        module = DINOv3SatAlignment("./models/facebook/dinov3-vitl16-pretrain-sat493m", encoder_dim=32)
        module.build_projector(model_feature_dim=16)
        model_feats = torch.randn(4, 16)
        enc_feats = torch.randn(4, 32)
        loss = module.compute_alignment_loss(model_feats, enc_feats)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_alignment_loss_spatial_input(self):
        """4-D spatial model features are pooled before projection."""
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14", encoder_dim=32)
        module.build_projector(model_feature_dim=8)
        model_feats = torch.randn(2, 8, 4, 4)
        enc_feats = torch.randn(2, 32)
        loss = module.compute_alignment_loss(model_feats, enc_feats)
        assert loss.ndim == 0

    def test_alignment_loss_identical_features(self):
        """Identical normalised features should yield loss close to -1."""
        module = SARCLIPAlignment("./models/BiliSakura/SARCLIP-ViT-L-14", encoder_dim=16)
        # No projector – direct comparison
        feats = torch.randn(4, 16)
        loss = module.compute_alignment_loss(feats.clone(), feats.clone())
        assert loss.item() < -0.9


# ---------------------------------------------------------------------------
# CUT trainer loss signature
# ---------------------------------------------------------------------------

class TestCUTTrainerLossSignature:
    """CUT compute_G_loss accepts the new latent_target_encoder and rep alignment kwargs."""

    def test_accepts_latent_kwargs(self):
        from examples.cut.trainer import CUTTrainer
        sig = inspect.signature(CUTTrainer.compute_G_loss)
        assert "latent_target_encoder" in sig.parameters
        assert "lambda_latent" in sig.parameters

    def test_accepts_rep_alignment_kwargs(self):
        from examples.cut.trainer import CUTTrainer
        sig = inspect.signature(CUTTrainer.compute_G_loss)
        assert "rep_alignment_module" in sig.parameters
        assert "lambda_rep_alignment" in sig.parameters


# ---------------------------------------------------------------------------
# DDBM trainer loss signature
# ---------------------------------------------------------------------------

class TestDDBMTrainerLossSignature:
    """DDBM compute_training_loss accepts the new latent_target_encoder and rep alignment kwargs."""

    def test_accepts_latent_kwargs(self):
        from examples.ddbm.trainer import DDBMTrainer
        sig = inspect.signature(DDBMTrainer.compute_training_loss)
        assert "latent_target_encoder" in sig.parameters
        assert "lambda_latent" in sig.parameters

    def test_accepts_rep_alignment_kwargs(self):
        from examples.ddbm.trainer import DDBMTrainer
        sig = inspect.signature(DDBMTrainer.compute_training_loss)
        assert "rep_alignment_module" in sig.parameters
        assert "lambda_rep_alignment" in sig.parameters


# ---------------------------------------------------------------------------
# DDIB config defaults
# ---------------------------------------------------------------------------

class TestDDIBConfigDefaults:
    """DDIB config has REPA fields with correct defaults."""

    def test_rep_alignment_defaults(self):
        from examples.ddib.config import TaskConfig as DdibConfig
        cfg = DdibConfig()
        assert cfg.use_rep_alignment is False
        assert cfg.rep_alignment_model_path is None
        assert cfg.lambda_rep_alignment == 1.0

    def test_overrides_work(self):
        from examples.ddib.config import rgb2ir_config as ddib_rgb2ir
        cfg = ddib_rgb2ir(use_rep_alignment=True, lambda_rep_alignment=0.5)
        assert cfg.use_rep_alignment is True
        assert cfg.lambda_rep_alignment == 0.5


# ---------------------------------------------------------------------------
# DDIB task-specific config paths
# ---------------------------------------------------------------------------

class TestDDIBTaskConfigPaths:
    """DDIB pre-built task configs set the expected model paths."""

    def test_rgb2ir_has_mars_rgb_path(self):
        from examples.ddib.config import rgb2ir_config as ddib_rgb2ir
        cfg = ddib_rgb2ir()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-RGB"

    def test_sar2eo_has_mars_sar_path(self):
        from examples.ddib.config import sar2eo_config as ddib_sar2eo
        cfg = ddib_sar2eo()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-SAR"

    def test_sar2ir_has_mars_sar_path(self):
        from examples.ddib.config import sar2ir_config as ddib_sar2ir
        cfg = ddib_sar2ir()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-SAR"

    def test_sar2rgb_has_mars_sar_path(self):
        from examples.ddib.config import sar2rgb_config as ddib_sar2rgb
        cfg = ddib_sar2rgb()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-B-SAR"


# ---------------------------------------------------------------------------
# DDIB trainer _train_single_domain signature
# ---------------------------------------------------------------------------

class TestDDIBTrainerSignature:
    """DDIB _train_single_domain accepts rep_alignment kwargs."""

    def test_accepts_rep_alignment_kwargs(self):
        from examples.ddib.trainer import DDIBTrainer
        sig = inspect.signature(DDIBTrainer._train_single_domain)
        assert "rep_alignment_module" in sig.parameters
        assert "lambda_rep_alignment" in sig.parameters


# ---------------------------------------------------------------------------
# DDIB scheduler return_pred_xstart
# ---------------------------------------------------------------------------

class TestDDIBSchedulerReturnPredXstart:
    """DDIB scheduler compute_training_loss supports return_pred_xstart."""

    def test_returns_tuple_when_requested(self):
        from src.schedulers import DDIBScheduler
        sig = inspect.signature(DDIBScheduler.compute_training_loss)
        assert "return_pred_xstart" in sig.parameters
