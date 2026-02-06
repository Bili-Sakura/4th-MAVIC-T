# Copyright 2024 The DDBM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Tests for the diffusers-compatible DDBM components.

These tests verify that the DDBM scheduler and pipeline work correctly
with the Hugging Face diffusers library conventions.
"""

import pytest
import torch
import numpy as np
import tempfile
import os


class TestDDBMScheduler:
    """Tests for the DDBMScheduler class."""

    def test_scheduler_import(self):
        """Test that the scheduler can be imported."""
        from ddbm.schedulers import DDBMScheduler
        assert DDBMScheduler is not None

    def test_scheduler_output_import(self):
        """Test that the scheduler output can be imported."""
        from ddbm.schedulers import DDBMSchedulerOutput
        assert DDBMSchedulerOutput is not None

    def test_scheduler_init_default(self):
        """Test scheduler initialization with default parameters."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler()

        assert scheduler.sigma_min == 0.002
        assert scheduler.sigma_max == 80.0
        assert scheduler.sigma_data == 0.5
        assert scheduler.pred_mode == "vp"
        assert scheduler.num_train_timesteps == 40

    def test_scheduler_init_custom(self):
        """Test scheduler initialization with custom parameters."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler(
            sigma_min=0.001,
            sigma_max=1.0,
            pred_mode="ve",
            num_train_timesteps=100,
        )

        assert scheduler.sigma_min == 0.001
        assert scheduler.sigma_max == 1.0
        assert scheduler.pred_mode == "ve"

    def test_set_timesteps(self):
        """Test setting timesteps."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler()
        scheduler.set_timesteps(40)

        assert scheduler.sigmas is not None
        assert len(scheduler.sigmas) == 41  # 40 steps + 1 zero at end
        assert scheduler.sigmas[-1] == 0  # Last sigma should be 0
        assert scheduler.num_inference_steps == 40

    def test_sigmas_are_decreasing(self):
        """Test that sigmas are monotonically decreasing."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler()
        scheduler.set_timesteps(40)

        sigmas = scheduler.sigmas[:-1]  # Exclude final zero
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1], f"Sigmas not decreasing at index {i}"

    def test_scale_model_input(self):
        """Test scale_model_input returns input unchanged."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler()
        sample = torch.randn(2, 3, 64, 64)

        scaled = scheduler.scale_model_input(sample, timestep=0)

        assert torch.allclose(sample, scaled)

    def test_config_mixin(self):
        """Test that scheduler properly inherits from ConfigMixin."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler(sigma_max=20.0, pred_mode="ve")

        # ConfigMixin should provide config attribute
        assert hasattr(scheduler, "config")
        assert scheduler.config.sigma_max == 20.0
        assert scheduler.config.pred_mode == "ve"

    def test_config_save_load(self):
        """Test saving and loading scheduler config."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler(
            sigma_max=1.0,
            sigma_min=0.001,
            pred_mode="vp",
            beta_d=2.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler.save_config(tmpdir)
            loaded_scheduler = DDBMScheduler.from_config(tmpdir)

            assert loaded_scheduler.config.sigma_max == scheduler.config.sigma_max
            assert loaded_scheduler.config.pred_mode == scheduler.config.pred_mode
            assert loaded_scheduler.config.beta_d == scheduler.config.beta_d

    def test_init_noise_sigma(self):
        """Test that init_noise_sigma is set correctly."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler(sigma_max=80.0)
        assert scheduler.init_noise_sigma == 80.0

        scheduler = DDBMScheduler(sigma_max=1.0)
        assert scheduler.init_noise_sigma == 1.0


class TestDDBMPipeline:
    """Tests for the DDBMPipeline class."""

    def test_pipeline_import(self):
        """Test that the pipeline can be imported."""
        from ddbm.pipelines import DDBMPipeline
        assert DDBMPipeline is not None

    def test_pipeline_output_import(self):
        """Test that the pipeline output can be imported."""
        from ddbm.pipelines import DDBMPipelineOutput
        assert DDBMPipelineOutput is not None

    def test_pipeline_with_mock_model(self):
        """Test pipeline initialization with a mock model."""
        from ddbm.pipelines import DDBMPipeline
        from ddbm.schedulers import DDBMScheduler

        # Create a simple mock model
        class MockUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)  # Dummy parameter

            def forward(self, x, timesteps, xT=None):
                return x  # Return input as denoised output

        model = MockUNet()
        scheduler = DDBMScheduler()

        pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

        assert pipeline.unet is model
        assert pipeline.scheduler is scheduler

    def test_pipeline_device_property(self):
        """Test pipeline device property."""
        from ddbm.pipelines import DDBMPipeline
        from ddbm.schedulers import DDBMScheduler

        class MockUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x, timesteps, xT=None):
                return x

        model = MockUNet()
        scheduler = DDBMScheduler()
        pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

        assert pipeline.device == next(model.parameters()).device

    def test_pipeline_dtype_property(self):
        """Test pipeline dtype property."""
        from ddbm.pipelines import DDBMPipeline
        from ddbm.schedulers import DDBMScheduler

        class MockUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x, timesteps, xT=None):
                return x

        model = MockUNet()
        scheduler = DDBMScheduler()
        pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

        assert pipeline.dtype == next(model.parameters()).dtype

    def test_pipeline_inference(self):
        """Test pipeline inference produces valid output."""
        from ddbm.pipelines import DDBMPipeline, DDBMPipelineOutput
        from ddbm.schedulers import DDBMScheduler

        class MockUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x, timesteps, xT=None):
                # Simple mock: return slightly modified input
                return x * 0.9 + 0.1 * torch.randn_like(x)

        model = MockUNet()
        scheduler = DDBMScheduler(pred_mode='vp', sigma_max=1.0)
        pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

        source_image = torch.randn(1, 3, 32, 32)
        result = pipeline(
            source_image=source_image,
            num_inference_steps=5,
            output_type='pt',
        )

        assert isinstance(result, DDBMPipelineOutput)
        assert result.images.shape == (1, 3, 32, 32)
        assert result.nfe > 0


class TestDDBMIntegration:
    """Integration tests for DDBM diffusers components."""

    def test_main_package_exports(self):
        """Test that main package exports diffusers components."""
        from ddbm import DDBMScheduler, DDBMPipeline

        assert DDBMScheduler is not None
        assert DDBMPipeline is not None

    def test_output_classes_exported(self):
        """Test that output classes are exported."""
        from ddbm import DDBMSchedulerOutput, DDBMPipelineOutput

        assert DDBMSchedulerOutput is not None
        assert DDBMPipelineOutput is not None

    def test_scheduler_config_mixin(self):
        """Test that scheduler properly inherits from ConfigMixin."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler(sigma_max=20.0, pred_mode="ve")

        # ConfigMixin should provide config attribute
        assert hasattr(scheduler, "config")
        assert scheduler.config.sigma_max == 20.0
        assert scheduler.config.pred_mode == "ve"

    def test_backward_compatibility(self):
        """Test backward compatibility with original API (when dependencies available)."""
        from ddbm import (
            DDBMScheduler,
            DDBMPipeline,
        )

        # Core diffusers components should always work
        assert DDBMScheduler is not None
        assert DDBMPipeline is not None

        # Original components require additional dependencies (piq, mpi4py, etc.)
        # Only test if those are available
        try:
            from ddbm import UNetModel, SongUNet
            assert UNetModel is not None
            assert SongUNet is not None
        except ImportError:
            # Expected if optional dependencies not installed
            pass

        try:
            from ddbm import create_model, model_and_diffusion_defaults
            assert create_model is not None
            assert model_and_diffusion_defaults is not None
        except ImportError:
            # Expected if optional dependencies not installed
            pass


class TestAddNoise:
    """Tests for the add_noise method."""

    def test_add_noise_ve_mode(self):
        """Test add_noise with VE mode."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler(pred_mode="ve", sigma_max=80.0)

        batch_size = 2
        channels = 3
        height = width = 32

        original_samples = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(original_samples)
        x_T = torch.randn_like(original_samples)
        timesteps = torch.tensor([0.5, 0.8])  # Sigma values

        noisy = scheduler.add_noise(original_samples, noise, timesteps, x_T)

        assert noisy.shape == original_samples.shape

    def test_add_noise_vp_mode(self):
        """Test add_noise with VP mode."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler(pred_mode="vp", sigma_max=1.0)

        batch_size = 2
        channels = 3
        height = width = 32

        original_samples = torch.randn(batch_size, channels, height, width)
        noise = torch.randn_like(original_samples)
        x_T = torch.randn_like(original_samples)
        timesteps = torch.tensor([0.5, 0.8])

        noisy = scheduler.add_noise(original_samples, noise, timesteps, x_T)

        assert noisy.shape == original_samples.shape

    def test_add_noise_deterministic(self):
        """Test that add_noise is deterministic with same inputs."""
        from ddbm.schedulers import DDBMScheduler

        scheduler = DDBMScheduler(pred_mode="vp", sigma_max=1.0)

        original_samples = torch.randn(2, 3, 32, 32)
        noise = torch.randn_like(original_samples)
        x_T = torch.randn_like(original_samples)
        timesteps = torch.tensor([0.5, 0.8])

        noisy1 = scheduler.add_noise(original_samples, noise, timesteps, x_T)
        noisy2 = scheduler.add_noise(original_samples, noise, timesteps, x_T)

        assert torch.allclose(noisy1, noisy2)


class TestSchedulerStep:
    """Tests for the scheduler step methods."""

    def test_step_basic(self):
        """Test basic step function."""
        from ddbm.schedulers import DDBMScheduler, DDBMSchedulerOutput

        scheduler = DDBMScheduler(pred_mode="vp", sigma_max=1.0)
        scheduler.set_timesteps(10)

        sample = torch.randn(1, 3, 32, 32)
        model_output = torch.randn_like(sample)
        x_T = torch.randn_like(sample)

        result = scheduler.step(
            model_output=model_output,
            timestep=0,
            sample=sample,
            x_T=x_T,
        )

        assert isinstance(result, DDBMSchedulerOutput)
        assert result.prev_sample.shape == sample.shape
        assert result.pred_original_sample.shape == sample.shape

    def test_step_heun(self):
        """Test Heun step function."""
        from ddbm.schedulers import DDBMScheduler, DDBMSchedulerOutput

        scheduler = DDBMScheduler(pred_mode="vp", sigma_max=1.0)
        scheduler.set_timesteps(10)

        sample = torch.randn(1, 3, 32, 32)
        denoised_1 = torch.randn_like(sample)
        denoised_2 = torch.randn_like(sample)
        x_T = torch.randn_like(sample)

        result = scheduler.step_heun(
            denoised_1=denoised_1,
            denoised_2=denoised_2,
            timestep=0,
            sample=sample,
            x_T=x_T,
        )

        assert isinstance(result, DDBMSchedulerOutput)
        assert result.prev_sample.shape == sample.shape


class TestPipelineOutputTypes:
    """Tests for different pipeline output types."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline for testing."""
        from ddbm.pipelines import DDBMPipeline
        from ddbm.schedulers import DDBMScheduler

        class MockUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)

            def forward(self, x, timesteps, xT=None):
                return x

        model = MockUNet()
        scheduler = DDBMScheduler(pred_mode='vp', sigma_max=1.0)
        return DDBMPipeline(unet=model, scheduler=scheduler)

    def test_output_pil(self, pipeline):
        """Test PIL output type."""
        from PIL import Image

        source_image = torch.randn(1, 3, 32, 32)
        result = pipeline(
            source_image=source_image,
            num_inference_steps=3,
            output_type='pil',
        )

        assert isinstance(result.images, list)
        assert isinstance(result.images[0], Image.Image)

    def test_output_numpy(self, pipeline):
        """Test numpy output type."""
        source_image = torch.randn(1, 3, 32, 32)
        result = pipeline(
            source_image=source_image,
            num_inference_steps=3,
            output_type='np',
        )

        assert isinstance(result.images, np.ndarray)
        assert result.images.shape == (1, 32, 32, 3)

    def test_output_tensor(self, pipeline):
        """Test tensor output type."""
        source_image = torch.randn(1, 3, 32, 32)
        result = pipeline(
            source_image=source_image,
            num_inference_steps=3,
            output_type='pt',
        )

        assert isinstance(result.images, torch.Tensor)
        assert result.images.shape == (1, 3, 32, 32)


class TestModuleStructure:
    """Tests for the package module structure."""

    def test_models_module_import(self):
        """Test that models module can be imported."""
        from ddbm.models import UNetModel, SongUNet
        assert UNetModel is not None
        assert SongUNet is not None

    def test_utils_module_import(self):
        """Test that utils module can be imported."""
        from ddbm.utils import (
            append_dims,
            mean_flat,
            timestep_embedding,
            normalization,
        )
        assert append_dims is not None
        assert mean_flat is not None
        assert timestep_embedding is not None
        assert normalization is not None

    def test_utils_append_dims(self):
        """Test append_dims utility function."""
        from ddbm.utils import append_dims

        x = torch.tensor([1.0, 2.0])  # shape: (2,)
        result = append_dims(x, 4)  # target: (2, 1, 1, 1)
        assert result.shape == (2, 1, 1, 1)

    def test_utils_mean_flat(self):
        """Test mean_flat utility function."""
        from ddbm.utils import mean_flat

        x = torch.randn(2, 3, 4, 4)
        result = mean_flat(x)
        assert result.shape == (2,)  # mean over all dims except batch

    def test_utils_timestep_embedding(self):
        """Test timestep_embedding utility function."""
        from ddbm.utils import timestep_embedding

        timesteps = torch.tensor([0, 500, 1000])
        embedding = timestep_embedding(timesteps, dim=128)
        assert embedding.shape == (3, 128)

    def test_version_attribute(self):
        """Test that version is accessible."""
        import ddbm
        assert hasattr(ddbm, "__version__")
        assert isinstance(ddbm.__version__, str)
        assert ddbm.__version__ == "0.4.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
