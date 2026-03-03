# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for DAB baseline components."""

import pytest
import torch

from src.models.unet_dab import DABUNet, create_model
from src.pipelines.dab import DABPipeline, DABPipelineOutput
from src.schedulers import DABScheduler, DABSchedulerOutput


class TestDABScheduler:
    def test_default_config(self):
        sched = DABScheduler()
        assert sched.config.num_timesteps == 1000
        assert sched.config.objective == "grad"
        assert sched.config.loss_type == "l1"
        assert sched.steps is not None

    def test_add_noise_shape(self):
        sched = DABScheduler(num_timesteps=100)
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.randint(1, 100, (2,))
        x_t = sched.add_noise(target, source, t)
        assert x_t.shape == target.shape

    def test_objective_grad(self):
        sched = DABScheduler(num_timesteps=100, objective="grad")
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.randint(1, 100, (2,))
        noise = torch.randn_like(target)
        x_t = sched.add_noise(target, source, t, noise)
        obj = sched.get_objective(target, source, t, noise, x_t=x_t)
        expected = x_t - target
        assert torch.allclose(obj, expected, atol=1e-5)

    def test_objective_noise(self):
        sched = DABScheduler(num_timesteps=100, objective="noise")
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.randint(1, 100, (2,))
        noise = torch.randn_like(target)
        obj = sched.get_objective(target, source, t, noise)
        assert torch.allclose(obj, noise)

    def test_objective_ysubx(self):
        sched = DABScheduler(num_timesteps=100, objective="ysubx")
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.randint(1, 100, (2,))
        noise = torch.randn_like(target)
        obj = sched.get_objective(target, source, t, noise)
        assert torch.allclose(obj, source - target)

    def test_predict_target_roundtrip_grad(self):
        sched = DABScheduler(num_timesteps=100, objective="grad")
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.tensor([50, 50], dtype=torch.long)
        noise = torch.randn_like(target)
        x_t = sched.add_noise(target, source, t, noise)
        obj = sched.get_objective(target, source, t, noise, x_t=x_t)
        pred = sched.predict_target_from_objective(x_t, source, t, obj)
        assert torch.allclose(pred, target, atol=1e-4)

    def test_predict_target_roundtrip_noise(self):
        sched = DABScheduler(num_timesteps=100, objective="noise")
        target = torch.randn(2, 3, 16, 16)
        source = torch.randn(2, 3, 16, 16)
        t = torch.tensor([50, 50], dtype=torch.long)
        noise = torch.randn_like(target)
        x_t = sched.add_noise(target, source, t, noise)
        pred = sched.predict_target_from_objective(x_t, source, t, noise)
        assert torch.allclose(pred, target, atol=1e-4)

    def test_step_output(self):
        sched = DABScheduler(num_timesteps=100, objective="grad")
        sched.set_timesteps(10)
        x = torch.randn(1, 3, 16, 16)
        source = torch.randn(1, 3, 16, 16)
        model_output = torch.randn_like(x)
        out = sched.step(model_output, 0, x, source)
        assert isinstance(out, DABSchedulerOutput)
        assert out.prev_sample.shape == x.shape

    def test_set_timesteps(self):
        sched = DABScheduler(num_timesteps=1000, skip_sample=False)
        assert len(sched.steps) == 999
        sched.set_timesteps(50)
        assert len(sched.steps) == 50


class TestDABUNet:
    def test_create_model(self):
        model = create_model(
            image_size=32,
            in_channels=3,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            condition_mode="concat",
            objective="grad",
        )
        assert isinstance(model, DABUNet)

    def test_forward_concat_context(self):
        model = create_model(
            image_size=32,
            in_channels=3,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            condition_mode="concat",
            objective="grad",
        )
        x_t = torch.randn(2, 3, 32, 32)
        t = torch.tensor([1, 2], dtype=torch.long)
        context = torch.randn(2, 3, 32, 32)
        out = model(x_t, t, context=context)
        assert out.shape == (2, 3, 32, 32)

    def test_concat_requires_context(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            condition_mode="concat",
            objective="grad",
        )
        x_t = torch.randn(1, 1, 32, 32)
        t = torch.tensor([1], dtype=torch.long)
        with pytest.raises(ValueError):
            _ = model(x_t, t)


class TestDABPipeline:
    def _make_pipeline(self) -> DABPipeline:
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            condition_mode="concat",
            objective="grad",
        )
        scheduler = DABScheduler(num_timesteps=100, objective="grad")
        return DABPipeline(unet=model, scheduler=scheduler)

    def test_pt_output(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)
        out = pipe(source_image=source, num_inference_steps=8, output_type="pt")
        assert isinstance(out, DABPipelineOutput)
        assert out.images.shape == source.shape

    def test_noise_objective_pipeline(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            condition_mode="concat",
            objective="noise",
        )
        scheduler = DABScheduler(num_timesteps=100, objective="noise")
        pipe = DABPipeline(unet=model, scheduler=scheduler)
        source = torch.randn(1, 1, 32, 32)
        out = pipe(source_image=source, num_inference_steps=8, output_type="pt")
        assert out.images.shape == source.shape
