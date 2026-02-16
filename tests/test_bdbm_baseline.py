"""Tests for BDBM baseline components."""

import pytest
import torch

from src.models.unet_bdbm import BDBMUNet, create_model
from src.pipelines.bdbm import BDBMPipeline, BDBMPipelineOutput
from src.schedulers import BDBMScheduler, BDBMSchedulerOutput


class TestBDBMScheduler:
    def test_default_config(self):
        sched = BDBMScheduler()
        assert sched.config.num_timesteps == 1000
        assert sched.config.mt_type == "linear"
        assert sched.config.objective == "noise"
        assert sched.steps is not None
        assert sched.asc_steps is not None

    def test_add_noise_shape(self):
        sched = BDBMScheduler(num_timesteps=100)
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 100, (2,))
        x_t = sched.add_noise(target, source, t)
        assert x_t.shape == target.shape

    def test_objective_both_shape(self):
        sched = BDBMScheduler(num_timesteps=100, objective="both")
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.randint(0, 100, (2,))
        noise = torch.randn_like(target)
        obj = sched.get_objective(target, source, t, noise)
        assert obj.shape == (2, 6, 16, 16)

    def test_predict_target_roundtrip_noise(self):
        sched = BDBMScheduler(num_timesteps=100, objective="noise")
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.tensor([50, 50], dtype=torch.long)
        noise = torch.randn_like(target)
        x_t = sched.add_noise(target, source, t, noise)
        pred = sched.predict_target_from_objective(x_t, source, t, noise)
        assert torch.allclose(pred, target, atol=1e-4)

    def test_step_outputs(self):
        sched = BDBMScheduler(num_timesteps=100, sample_step=12, objective="noise")
        x = torch.randn(1, 3, 16, 16)
        other = torch.randn(1, 3, 16, 16)
        model_output = torch.randn_like(x)
        out_b2a = sched.step_b2a(model_output, 0, x, other)
        out_a2b = sched.step_a2b(model_output, 0, x, other)
        assert isinstance(out_b2a, BDBMSchedulerOutput)
        assert isinstance(out_a2b, BDBMSchedulerOutput)
        assert out_b2a.prev_sample.shape == x.shape
        assert out_a2b.prev_sample.shape == x.shape


class TestBDBMUNet:
    def test_create_model(self):
        model = create_model(
            image_size=32,
            in_channels=3,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            condition_mode="dual",
            objective="both",
        )
        assert isinstance(model, BDBMUNet)

    def test_forward_dual_context(self):
        model = create_model(
            image_size=32,
            in_channels=3,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            condition_mode="dual",
            objective="both",
        )
        x_t = torch.randn(2, 3, 32, 32)
        t = torch.tensor([1, 2], dtype=torch.long)
        context = torch.randn(2, 6, 32, 32)
        out = model(x_t, t, context=context)
        assert out.shape == (2, 6, 32, 32)

    def test_dual_requires_context(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            condition_mode="dual",
            objective="noise",
        )
        x_t = torch.randn(1, 1, 32, 32)
        t = torch.tensor([1], dtype=torch.long)
        with pytest.raises(ValueError):
            _ = model(x_t, t)


class TestBDBMPipeline:
    def _make_pipeline(self) -> BDBMPipeline:
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            condition_mode="dual",
            objective="noise",
        )
        scheduler = BDBMScheduler(num_timesteps=100, sample_step=10, objective="noise")
        return BDBMPipeline(unet=model, scheduler=scheduler)

    def test_b2a_pt_output(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)
        out = pipe(source_image=source, direction="b2a", num_inference_steps=8, output_type="pt")
        assert isinstance(out, BDBMPipelineOutput)
        assert out.images.shape == source.shape

    def test_a2b_pt_output(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)
        out = pipe(source_image=source, direction="a2b", num_inference_steps=8, output_type="pt")
        assert out.images.shape == source.shape
