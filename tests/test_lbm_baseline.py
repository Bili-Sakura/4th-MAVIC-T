# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for the LBM baseline module.

Validates config builders, model forward pass, scheduler operations,
and pipeline end-to-end (with synthetic tensors — no real checkpoints needed).
"""

import pytest
import torch

from examples.lbm.config import (
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from src.models.unet import I2SBUNet
from src.models.unet.unet_2d import create_model
from src.schedulers import LBMScheduler, LBMSchedulerOutput
from src.pipelines.lbm import LBMPipeline, LBMPipelineOutput


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self):
        cfg = TaskConfig()
        assert cfg.num_train_timesteps == 1000
        assert cfg.bridge_noise_sigma == 0.001
        assert cfg.timestep_sampling == "uniform"
        assert cfg.condition_mode == "concat"
        assert cfg.use_ema is True
        assert cfg.push_to_hub is True
        assert cfg.num_inference_steps == 1  # LBM default: single-step

    @pytest.mark.parametrize("builder,task_name", [
        (sar2eo_config, "sar2eo"),
        (rgb2ir_config, "rgb2ir"),
        (sar2ir_config, "sar2ir"),
        (sar2rgb_config, "sar2rgb"),
    ])
    def test_task_builders(self, builder, task_name):
        cfg = builder()
        assert cfg.task_name == task_name

    def test_override(self):
        cfg = sar2eo_config(train_batch_size=64)
        assert cfg.train_batch_size == 64

    @pytest.mark.parametrize("builder", [
        sar2eo_config, rgb2ir_config, sar2ir_config, sar2rgb_config,
    ])
    def test_latent_paths(self, builder):
        cfg = builder()
        assert cfg.latent_vae_path is not None

    def test_sar2rgb_has_repa_path(self):
        """Only SAR2RGB has a REPA model path (MaRS-Base-RGB for target encoding)."""
        cfg = sar2rgb_config()
        assert cfg.rep_alignment_model_path == "./models/BiliSakura/MaRS-Base-RGB"

    @pytest.mark.parametrize("builder", [sar2eo_config, rgb2ir_config, sar2ir_config])
    def test_no_repa_path_for_unsupported_tasks(self, builder):
        """SAR2EO, RGB2IR, SAR2IR have no REPA model path."""
        cfg = builder()
        assert cfg.rep_alignment_model_path is None

    def test_custom_timesteps_config(self):
        """Verify custom_timesteps config can be set."""
        cfg = TaskConfig(
            timestep_sampling="custom_timesteps",
            selected_timesteps=[250, 500, 750, 1000],
            prob=[0.25, 0.25, 0.25, 0.25],
        )
        assert cfg.timestep_sampling == "custom_timesteps"
        assert len(cfg.selected_timesteps) == 4
        assert sum(cfg.prob) == 1.0


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestModel:
    def test_create_small_model(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode="concat",
        )
        assert isinstance(model, I2SBUNet)

    def test_forward_unconditional(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode=None,
        )
        x = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0.5, 0.8])
        out = model(x, t)
        assert out.shape == x.shape

    def test_forward_conditional(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode="concat",
        )
        x = torch.randn(2, 1, 32, 32)
        cond = torch.randn(2, 1, 32, 32)
        t = torch.tensor([0.5, 0.8])
        out = model(x, t, cond=cond)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------

class TestScheduler:
    @pytest.fixture
    def scheduler(self):
        return LBMScheduler(num_train_timesteps=100, bridge_noise_sigma=0.005)

    def test_init(self, scheduler):
        assert scheduler.num_train_timesteps == 100
        assert scheduler.bridge_noise_sigma == 0.005
        assert len(scheduler.sigmas) == 101  # num_train_timesteps + 1
        assert len(scheduler.timesteps) == 100

    def test_sample_timesteps_uniform(self, scheduler):
        ts = scheduler.sample_timesteps(4)
        assert ts.shape == (4,)
        assert (ts >= 0).all()
        assert (ts < 100).all()

    def test_sample_timesteps_log_normal(self):
        sched = LBMScheduler(num_train_timesteps=100, timestep_sampling="log_normal")
        ts = sched.sample_timesteps(4)
        assert ts.shape == (4,)

    def test_sample_timesteps_custom(self):
        sched = LBMScheduler(
            num_train_timesteps=100,
            timestep_sampling="custom_timesteps",
            selected_timesteps=[25, 50, 75],
            prob=[0.3, 0.4, 0.3],
        )
        ts = sched.sample_timesteps(8)
        assert ts.shape == (8,)
        for t in ts:
            assert t.item() in [25, 50, 75]

    def test_get_sigmas(self, scheduler):
        ts = torch.tensor([0, 50, 99])
        sigmas = scheduler.get_sigmas(ts, n_dim=4)
        assert sigmas.shape == (3, 1, 1, 1)
        # sigma at t=0 should be highest (close to 1.0)
        assert sigmas[0].item() > sigmas[1].item()
        assert sigmas[1].item() > sigmas[2].item()

    def test_add_noise(self, scheduler):
        x_target = torch.randn(2, 1, 8, 8)
        x_source = torch.randn(2, 1, 8, 8)
        ts = torch.tensor([10, 50])
        noisy = scheduler.add_noise(x_target, x_source, ts)
        assert noisy.shape == x_target.shape

    def test_compute_target(self, scheduler):
        x_source = torch.randn(2, 1, 8, 8)
        x_target = torch.randn(2, 1, 8, 8)
        target = scheduler.compute_target(x_source, x_target)
        assert target.shape == x_source.shape
        assert torch.allclose(target, x_source - x_target)

    def test_compute_pred_x0(self, scheduler):
        noisy = torch.randn(2, 1, 8, 8)
        model_output = torch.randn(2, 1, 8, 8)
        sigmas = torch.tensor([0.5, 0.3]).view(2, 1, 1, 1)
        pred_x0 = scheduler.compute_pred_x0(noisy, model_output, sigmas)
        assert pred_x0.shape == noisy.shape
        expected = noisy - model_output * sigmas
        assert torch.allclose(pred_x0, expected)

    def test_set_timesteps(self, scheduler):
        scheduler.set_timesteps(num_inference_steps=5)
        assert scheduler.num_inference_steps == 5
        assert len(scheduler.sigmas) == 5
        assert len(scheduler.timesteps) == 5

    def test_step(self, scheduler):
        scheduler.set_timesteps(num_inference_steps=5)
        sample = torch.randn(2, 1, 8, 8)
        model_output = torch.randn(2, 1, 8, 8)
        t = scheduler.timesteps[0]
        result = scheduler.step(model_output, t, sample)
        assert isinstance(result, LBMSchedulerOutput)
        assert result.prev_sample.shape == sample.shape
        assert result.pred_original_sample.shape == sample.shape

    def test_step_return_tuple(self, scheduler):
        scheduler.set_timesteps(num_inference_steps=5)
        sample = torch.randn(2, 1, 8, 8)
        model_output = torch.randn(2, 1, 8, 8)
        t = scheduler.timesteps[0]
        result = scheduler.step(model_output, t, sample, return_dict=False)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPipeline:
    @pytest.fixture
    def pipeline(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode="concat",
        )
        scheduler = LBMScheduler(num_train_timesteps=100, bridge_noise_sigma=0.005)
        return LBMPipeline(unet=model, scheduler=scheduler)

    def test_pipeline_pt_output(self, pipeline):
        source = torch.randn(2, 1, 32, 32)
        result = pipeline(source, num_inference_steps=1, output_type="pt")
        assert isinstance(result, LBMPipelineOutput)
        assert result.images.shape == source.shape
        assert result.nfe == 1

    def test_pipeline_pil_output(self, pipeline):
        source = torch.randn(1, 1, 32, 32)
        result = pipeline(source, num_inference_steps=1, output_type="pil")
        assert isinstance(result.images, list)
        assert len(result.images) == 1

    def test_pipeline_np_output(self, pipeline):
        source = torch.randn(1, 1, 32, 32)
        result = pipeline(source, num_inference_steps=1, output_type="np")
        import numpy as np
        assert isinstance(result.images, np.ndarray)

    def test_pipeline_multi_step(self, pipeline):
        """LBM supports multi-step inference (not just single-step)."""
        source = torch.randn(1, 1, 32, 32)
        result = pipeline(source, num_inference_steps=4, output_type="pt")
        assert result.images.shape == source.shape
        assert result.nfe == 4


# ---------------------------------------------------------------------------
# Trainer signature tests
# ---------------------------------------------------------------------------

class TestTrainerSignature:
    """Test that LBMTrainer methods exist with correct signatures."""

    @pytest.fixture(autouse=True)
    def _require_datasets(self):
        pytest.importorskip("datasets", reason="datasets package required")

    def test_trainer_instantiation(self):
        from examples.lbm.trainer import LBMTrainer
        cfg = sar2eo_config()
        trainer = LBMTrainer(cfg)
        assert trainer.cfg is cfg

    def test_build_model(self):
        from examples.lbm.trainer import LBMTrainer
        cfg = sar2eo_config(resolution=32, num_channels=32, attention_resolutions="")
        trainer = LBMTrainer(cfg)
        model = trainer.build_model()
        assert isinstance(model, I2SBUNet)

    def test_build_scheduler(self):
        from examples.lbm.trainer import LBMTrainer
        cfg = sar2eo_config()
        trainer = LBMTrainer(cfg)
        scheduler = trainer.build_scheduler()
        assert isinstance(scheduler, LBMScheduler)

    def test_compute_training_loss(self):
        from examples.lbm.trainer import LBMTrainer
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            condition_mode="concat",
        )
        scheduler = LBMScheduler(num_train_timesteps=100, bridge_noise_sigma=0.005)
        x0 = torch.randn(2, 1, 32, 32)
        x_T = torch.randn(2, 1, 32, 32)
        loss, extras = LBMTrainer.compute_training_loss(model, scheduler, x0, x_T)
        assert loss.ndim == 0  # scalar loss
        assert loss.requires_grad
        assert "loss_mavic" in extras
        assert "loss_latent" in extras
        assert "loss_rep_alignment" in extras
