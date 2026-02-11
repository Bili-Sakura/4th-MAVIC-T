"""Tests for the BiBBDM baseline integration.

Covers:
* Module imports and public API
* Config dataclass fields and factory functions
* Scheduler (Brownian Bridge schedule, q-sample, objective, reconstruction)
* UNet model creation and forward pass
* Pipeline output format
* Trainer loss signature
"""

import inspect

import pytest
import torch

from src.schedulers import BiBBDMScheduler, BiBBDMSchedulerOutput
from src.pipelines.bibbdm import BiBBDMPipeline, BiBBDMPipelineOutput
from src.models.unet_bibbdm import BiBBDMUNet, create_model
from examples.bibbdm.config import (
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from src.schedulers.scheduling_bibbdm import _extract
from src.models.unet_bibbdm import _out_channels_for_objective


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestImports:
    """All public symbols are importable from the package."""

    def test_scheduler_classes(self):
        assert BiBBDMScheduler is not None
        assert BiBBDMSchedulerOutput is not None

    def test_pipeline_classes(self):
        assert BiBBDMPipeline is not None
        assert BiBBDMPipelineOutput is not None

    def test_model_classes(self):
        assert BiBBDMUNet is not None
        assert create_model is not None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    """TaskConfig has all expected fields with correct defaults."""

    def test_bibbdm_specific_defaults(self):
        cfg = TaskConfig()
        assert cfg.num_timesteps == 1000
        assert cfg.mt_type == "linear"
        assert cfg.m0 == 0.001
        assert cfg.mT == 0.999
        assert cfg.eta == 1.0
        assert cfg.var_scale == 2.0
        assert cfg.objective == "dlns"
        assert cfg.loss_type == "l1"
        assert cfg.weight_obj == 1.0
        assert cfg.weight_a_recon == 0.0
        assert cfg.weight_b_recon == 0.0
        assert cfg.skip_sample is True
        assert cfg.sample_step == 100
        assert cfg.sample_step_type == "linear"

    def test_latent_target_defaults(self):
        cfg = TaskConfig()
        assert cfg.use_latent_target is False
        assert cfg.latent_vae_path is None
        assert cfg.lambda_latent == 1.0

    def test_rep_alignment_defaults(self):
        cfg = TaskConfig()
        assert cfg.use_rep_alignment is False
        assert cfg.rep_alignment_model_path is None
        assert cfg.lambda_rep_alignment == 0.1

    def test_overrides_work(self):
        cfg = rgb2ir_config(use_latent_target=True, lambda_latent=0.5)
        assert cfg.use_latent_target is True
        assert cfg.lambda_latent == 0.5


class TestTaskConfigPaths:
    """Pre-built task configs set the expected model paths."""

    def test_rgb2ir_has_vae_path(self):
        cfg = rgb2ir_config()
        assert cfg.latent_vae_path == "./models/BiliSakura/VAEs/FLUX2-VAE"

    def test_rgb2ir_no_repa_path(self):
        """RGB2IR has no REPA encoder (no pre-trained IR encoder)."""
        cfg = rgb2ir_config()
        assert cfg.rep_alignment_model_path is None

    def test_sar2eo_no_repa_path(self):
        """SAR2EO has no REPA encoder (no pre-trained EO encoder)."""
        cfg = sar2eo_config()
        assert cfg.rep_alignment_model_path is None

    def test_sar2ir_no_repa_path(self):
        """SAR2IR has no REPA encoder (no pre-trained IR encoder)."""
        cfg = sar2ir_config()
        assert cfg.rep_alignment_model_path is None

    def test_sar2rgb_has_mars_rgb_path(self):
        """SAR2RGB uses MaRS-Base-RGB to encode the RGB target."""
        cfg = sar2rgb_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-Base-RGB"

    def test_all_tasks_have_vae_path(self):
        for fn in (sar2eo_config, sar2ir_config, sar2rgb_config, rgb2ir_config):
            cfg = fn()
            assert cfg.latent_vae_path is not None

    def test_task_names(self):
        assert sar2eo_config().task_name == "sar2eo"
        assert rgb2ir_config().task_name == "rgb2ir"
        assert sar2ir_config().task_name == "sar2ir"
        assert sar2rgb_config().task_name == "sar2rgb"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TestScheduler:
    """BiBBDMScheduler correctly implements the Brownian Bridge schedule."""

    def test_schedule_registration(self):
        sched = BiBBDMScheduler(num_timesteps=100, mt_type="linear")
        assert sched.m_t is not None
        assert sched.variance_t is not None
        assert sched.m_t.shape == (100,)
        assert sched.variance_t.shape == (100,)

    def test_m_t_bounds(self):
        sched = BiBBDMScheduler(num_timesteps=1000, mt_type="linear", m0=0.001, mT=0.999)
        assert sched.m_t[0].item() == pytest.approx(0.001, abs=1e-6)
        assert sched.m_t[-1].item() == pytest.approx(0.999, abs=1e-6)

    def test_variance_t_non_negative(self):
        sched = BiBBDMScheduler(num_timesteps=1000, mt_type="linear")
        assert (sched.variance_t >= 0).all()

    def test_sin_schedule(self):
        sched = BiBBDMScheduler(num_timesteps=100, mt_type="sin")
        assert sched.m_t.shape == (100,)

    def test_add_noise_shape(self):
        sched = BiBBDMScheduler(num_timesteps=100)
        target = torch.randn(2, 3, 32, 32)
        source = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 100, (2,))
        x_t = sched.add_noise(target, source, t)
        assert x_t.shape == target.shape

    def test_add_noise_at_t0_close_to_target(self):
        sched = BiBBDMScheduler(num_timesteps=100, mt_type="linear", m0=0.001, mT=0.999)
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.zeros(2, dtype=torch.long)
        noise = torch.zeros_like(target)  # zero noise
        x_t = sched.add_noise(target, source, t, noise=noise)
        # At t=0, m_t ≈ 0.001 so x_t ≈ 0.999 * target + 0.001 * source
        expected = (1 - 0.001) * target + 0.001 * source
        assert torch.allclose(x_t, expected, atol=1e-4)

    def test_get_objective_dlns(self):
        sched = BiBBDMScheduler(num_timesteps=100, objective="dlns")
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 100, (2,))
        noise = torch.randn_like(target)
        obj = sched.get_objective(target, source, t, noise)
        # dlns concatenates (b-a, noise) → 6 channels
        assert obj.shape == (2, 6, 16, 16)

    def test_get_objective_noise(self):
        sched = BiBBDMScheduler(num_timesteps=100, objective="noise")
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 100, (2,))
        noise = torch.randn_like(target)
        obj = sched.get_objective(target, source, t, noise)
        assert torch.allclose(obj, noise)

    def test_steps_built(self):
        sched = BiBBDMScheduler(num_timesteps=100, skip_sample=True, sample_step=20)
        assert sched.steps is not None
        assert len(sched.steps) > 0

    def test_set_timesteps(self):
        sched = BiBBDMScheduler(num_timesteps=100, skip_sample=True, sample_step=20)
        old_len = len(sched.steps)
        sched.set_timesteps(50)
        assert sched.sample_step == 50

    def test_predict_target_roundtrip_noise(self):
        """With objective='noise', predicting target from perfect noise pred recovers target."""
        sched = BiBBDMScheduler(num_timesteps=100, mt_type="linear", m0=0.01, mT=0.99, objective="noise")
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.tensor([50, 50], dtype=torch.long)
        noise = torch.randn_like(target)
        # Perfect prediction: objective_recon == noise
        target_pred = sched.predict_target_from_objective(
            sched.add_noise(target, source, t, noise), source, t, noise
        )
        assert torch.allclose(target_pred, target, atol=1e-4)


# ---------------------------------------------------------------------------
# UNet model
# ---------------------------------------------------------------------------

class TestUNetModel:
    """BiBBDMUNet creation and forward pass."""

    def test_create_model_default(self):
        model = create_model(image_size=32, in_channels=3, num_channels=32)
        assert isinstance(model, BiBBDMUNet)

    def test_out_channels_for_objective(self):
        assert _out_channels_for_objective("noise", 3) == 3
        assert _out_channels_for_objective("dlns", 3) == 6
        assert _out_channels_for_objective("dlab", 1) == 2
        assert _out_channels_for_objective("dlgab", 3) == 6
        assert _out_channels_for_objective("a", 3) == 3

    def test_forward_shape_standard(self):
        model = create_model(image_size=32, in_channels=3, num_channels=32, objective="noise")
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([0, 1])
        ctx = torch.randn(2, 3, 32, 32)
        out = model(x, t, context=ctx)
        assert out.shape == (2, 3, 32, 32)

    def test_forward_shape_dlns(self):
        model = create_model(image_size=32, in_channels=3, num_channels=32, objective="dlns")
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([0, 1])
        ctx = torch.randn(2, 3, 32, 32)
        out = model(x, t, context=ctx)
        assert out.shape == (2, 6, 32, 32)

    def test_forward_no_context(self):
        model = create_model(
            image_size=32, in_channels=3, num_channels=32,
            condition_mode=None, objective="noise",
        )
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([0, 1])
        out = model(x, t)
        assert out.shape == (2, 3, 32, 32)


# ---------------------------------------------------------------------------
# Trainer loss signature
# ---------------------------------------------------------------------------

class TestTrainerSignature:
    """BiBBDM trainer accepts the expected keyword arguments."""

    def test_compute_training_loss_accepts_kwargs(self):
        from examples.bibbdm.trainer import BiBBDMTrainer
        sig = inspect.signature(BiBBDMTrainer.compute_training_loss)
        expected_params = [
            "model", "scheduler", "target", "source",
            "objective", "loss_type",
            "weight_obj", "weight_a_recon", "weight_b_recon",
            "mavic_criterion", "mavic_loss_weight",
            "latent_target_encoder", "lambda_latent",
            "rep_alignment_module", "lambda_rep_alignment",
        ]
        for param_name in expected_params:
            assert param_name in sig.parameters, f"Missing param: {param_name}"


# ---------------------------------------------------------------------------
# Extract helper
# ---------------------------------------------------------------------------

class TestExtract:
    """The _extract helper works correctly."""

    def test_extract_shape(self):
        a = torch.arange(10, dtype=torch.float32)
        t = torch.tensor([3, 7])
        out = _extract(a, t, (2, 3, 4, 4))
        assert out.shape == (2, 1, 1, 1)
        assert out[0].item() == pytest.approx(3.0)
        assert out[1].item() == pytest.approx(7.0)
