# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for CDTSDE baseline integration."""

import torch

from examples.cdtsde.config import (
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from src.models import CDTSDEUNet
from examples.cdtsde.model import create_cdtsde_model as create_model
from src.schedulers import CDTSDEScheduler, CDTSDESchedulerOutput
from src.pipelines.cdtsde import CDTSDEPipeline, CDTSDEPipelineOutput


class TestConfig:
    def test_defaults(self):
        cfg = TaskConfig()
        assert cfg.cdtsde_num_train_timesteps == 1000
        assert cfg.beta_schedule == "linear"
        assert cfg.eta_schedule == "trunc_12"
        assert cfg.num_inference_steps == 50
        assert cfg.stochastic is True
        assert cfg.apply_domain_shift is True

    def test_task_builders(self):
        assert sar2eo_config().task_name == "sar2eo"
        assert rgb2ir_config().task_name == "rgb2ir"
        assert sar2ir_config().task_name == "sar2ir"
        assert sar2rgb_config().task_name == "sar2rgb"


class TestScheduler:
    def test_set_timesteps(self):
        sched = CDTSDEScheduler(num_train_timesteps=100)
        sched.set_timesteps(8)
        assert sched.timesteps is not None
        assert len(sched.timesteps) == 9
        assert sched.sigmas is not None
        assert len(sched.sigmas) == 9

    def test_add_noise_and_roundtrip(self):
        sched = CDTSDEScheduler(num_train_timesteps=100)
        x0 = torch.randn(2, 1, 16, 16)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 100, (2,), dtype=torch.long)
        x_t = sched.add_noise(x0, noise, t)
        x0_recon = sched.predict_start_from_noise(
            sample=x_t,
            timesteps=t,
            noise=noise,
            use_inference_schedule=False,
        )
        assert torch.allclose(x0_recon, x0, atol=1e-4)

    def test_step_output_shape(self):
        sched = CDTSDEScheduler(num_train_timesteps=100)
        sched.set_timesteps(4)
        sample = torch.randn(1, 1, 16, 16)
        pred = torch.randn_like(sample)
        ref = torch.randn_like(sample)
        out = sched.step(
            pred_original_sample=pred,
            step_index=4,
            sample=sample,
            reference_sample=ref,
            stochastic=False,
        )
        assert isinstance(out, CDTSDESchedulerOutput)
        assert out.prev_sample.shape == sample.shape


class TestModel:
    def test_create_model(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            lambda_hidden_channels=8,
        )
        assert isinstance(model, CDTSDEUNet)

    def test_forward_and_lambda_field(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            lambda_hidden_channels=8,
        )
        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 100, (2,), dtype=torch.long)
        source = torch.randn_like(x)
        out = model(x, t, xT=source)
        assert out.shape == x.shape

        lam = model.predict_lambda(torch.tensor([0.3, 0.7]), x.shape)
        assert lam.shape == x.shape
        assert (lam >= 0).all() and (lam <= 1).all()


class TestPipeline:
    def _make_pipeline(self) -> CDTSDEPipeline:
        model = create_model(
            image_size=32,
            in_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            lambda_hidden_channels=8,
        )
        scheduler = CDTSDEScheduler(num_train_timesteps=100)
        return CDTSDEPipeline(unet=model, scheduler=scheduler)

    def test_pt_output(self):
        pipe = self._make_pipeline()
        source = torch.randn(1, 1, 32, 32)
        out = pipe(
            source_image=source,
            num_inference_steps=4,
            stochastic=False,
            output_type="pt",
        )
        assert isinstance(out, CDTSDEPipelineOutput)
        assert out.images.shape == source.shape
        assert out.nfe > 0

