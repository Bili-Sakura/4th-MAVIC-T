# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for standalone SID2 baseline integration."""

import torch

from examples.sid2.config import (
    TaskConfig,
    rgb2ir_config,
    sar2eo_config,
    sar2ir_config,
    sar2rgb_config,
)
from examples.sid2.trainer import SID2Trainer
from src.models.unet.unet_sid import SiDUNet, create_sid_model
from src.pipelines.sid2 import SID2Pipeline, SID2PipelineOutput
from src.schedulers import SiD2Scheduler, SiD2SchedulerOutput


class TestConfig:
    def test_defaults(self):
        cfg = TaskConfig()
        cfg = sar2eo_config()
        assert cfg.unet_type == "sid"
        assert cfg.prediction_type == "v"
        assert cfg.schedule_type == "cosine_interpolated"
        assert cfg.sid2_include_dlogsnr is True
        assert cfg.sid2_sigmoid_bias == -1.0  # 256² default

    def test_task_builders(self):
        assert sar2eo_config().task_name == "sar2eo"
        assert rgb2ir_config().task_name == "rgb2ir"
        assert sar2ir_config().task_name == "sar2ir"
        assert sar2rgb_config().task_name == "sar2rgb"

    def test_resolution_bias_defaults(self):
        assert sar2eo_config(resolution=128).sid2_sigmoid_bias == 0.0
        assert sar2eo_config(resolution=256).sid2_sigmoid_bias == -1.0
        assert sar2eo_config(resolution=512).sid2_sigmoid_bias == -3.0
        assert sar2eo_config(resolution=1024).sid2_sigmoid_bias == -4.0


class TestModelAndScheduler:
    def test_create_small_model(self):
        model = create_sid_model(
            image_size=32,
            in_channels=1,
            out_channels=1,
            condition_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
        )
        assert isinstance(model, SiDUNet)

    def test_scheduler_step_shape(self):
        sched = SiD2Scheduler(image_d=32, interpolated_noise_d_low=2.0, interpolated_noise_d_high=32.0)
        sched.set_timesteps(4)
        sample = torch.randn(2, 1, 32, 32)
        pred = torch.randn_like(sample)
        out = sched.step(pred, 0, sample)
        assert isinstance(out, SiD2SchedulerOutput)
        assert out.prev_sample.shape == sample.shape


class TestPipeline:
    def test_pt_output(self):
        model = create_sid_model(
            image_size=32,
            in_channels=1,
            out_channels=1,
            condition_channels=1,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
        )
        scheduler = SiD2Scheduler(
            image_d=32,
            interpolated_noise_d_low=2.0,
            interpolated_noise_d_high=32.0,
        )
        pipe = SID2Pipeline(unet=model, scheduler=scheduler)
        source = torch.randn(1, 1, 32, 32)
        out = pipe(source_image=source, num_inference_steps=4, output_type="pt")
        assert isinstance(out, SID2PipelineOutput)
        assert out.images.shape == source.shape
        assert out.nfe > 0


class TestTrainerLoss:
    def test_compute_training_loss_scalar(self):
        cfg = sar2eo_config(
            resolution=32,
            num_channels=32,
            num_res_blocks=1,
            attention_resolutions="",
            channel_mult="1",
            source_channels=1,
            target_channels=1,
        )
        trainer = SID2Trainer(cfg)
        model = trainer.build_model(image_size=32)
        scheduler = trainer.build_scheduler()
        x0 = torch.randn(2, 1, 32, 32)
        x_t = torch.randn(2, 1, 32, 32)
        loss, extras = trainer.compute_training_loss(model, scheduler, x0, x_t)
        assert loss.ndim == 0
        assert loss.requires_grad
        assert isinstance(extras, dict)
