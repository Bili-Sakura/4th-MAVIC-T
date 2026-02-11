"""Tests for latent-space pipeline variants.

Verifies that each ``*LatentPipeline`` can be constructed and that it
correctly inherits from ``DiffusionPipeline``.  These are import and
construction smoke tests — full inference requires a real VAE checkpoint.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from diffusers import DiffusionPipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_CHANNELS = 32  # GroupNorm needs >= 32 channels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_vae(latent_channels=4, scaling_factor=0.18215):
    """Create a minimal mock AutoencoderKL for testing."""
    vae = MagicMock()
    vae.config = MagicMock()
    vae.config.scaling_factor = scaling_factor

    # encode returns a latent_dist with .mean
    latent_dist = MagicMock()
    latent_dist.mean = torch.randn(1, latent_channels, 4, 4)
    encode_output = MagicMock()
    encode_output.latent_dist = latent_dist
    vae.encode.return_value = encode_output

    # decode returns .sample
    decode_output = MagicMock()
    decode_output.sample = torch.randn(1, 3, 32, 32)
    vae.decode.return_value = decode_output

    return vae


# ---------------------------------------------------------------------------
# DDBM Latent Pipeline
# ---------------------------------------------------------------------------


class TestDDBMLatentPipeline:
    def test_is_diffusion_pipeline_subclass(self):
        from src.pipelines.ddbm import DDBMLatentPipeline

        assert issubclass(DDBMLatentPipeline, DiffusionPipeline)

    def test_construction(self):
        from src.models.unet_ddbm import DDBMUNet
        from src.schedulers import DDBMScheduler
        from src.pipelines.ddbm import DDBMLatentPipeline

        unet = DDBMUNet(image_size=32, in_channels=_MIN_CHANNELS, model_channels=_MIN_CHANNELS)
        scheduler = DDBMScheduler(sigma_min=0.002, sigma_max=80.0, sigma_data=0.5)
        vae = _make_mock_vae()

        pipe = DDBMLatentPipeline(unet=unet, scheduler=scheduler, vae=vae)

        assert pipe.unet is not None
        assert pipe.scheduler is not None
        assert pipe.vae is not None

    def test_has_encode_decode(self):
        from src.pipelines.ddbm import DDBMLatentPipeline

        assert hasattr(DDBMLatentPipeline, "_encode")
        assert hasattr(DDBMLatentPipeline, "_decode")

    def test_adapt_channels_noop_for_3ch(self):
        from src.pipelines.ddbm import DDBMLatentPipeline

        img = torch.randn(2, 3, 8, 8)
        out = DDBMLatentPipeline._adapt_channels(img)
        assert out.shape == (2, 3, 8, 8)

    def test_adapt_channels_repeats_1ch(self):
        from src.pipelines.ddbm import DDBMLatentPipeline

        img = torch.randn(2, 1, 8, 8)
        out = DDBMLatentPipeline._adapt_channels(img)
        assert out.shape == (2, 3, 8, 8)

    def test_restore_channels_averages_to_1ch(self):
        from src.pipelines.ddbm import DDBMLatentPipeline

        img = torch.randn(2, 3, 8, 8)
        out = DDBMLatentPipeline._restore_channels(img, 1)
        assert out.shape == (2, 1, 8, 8)

    def test_restore_channels_noop_for_3ch(self):
        from src.pipelines.ddbm import DDBMLatentPipeline

        img = torch.randn(2, 3, 8, 8)
        out = DDBMLatentPipeline._restore_channels(img, 3)
        assert out.shape == (2, 3, 8, 8)


# ---------------------------------------------------------------------------
# BiBBDM Latent Pipeline
# ---------------------------------------------------------------------------


class TestBiBBDMLatentPipeline:
    def test_is_diffusion_pipeline_subclass(self):
        from src.pipelines.bibbdm import BiBBDMLatentPipeline

        assert issubclass(BiBBDMLatentPipeline, DiffusionPipeline)

    def test_construction(self):
        from src.models.unet_bibbdm import BiBBDMUNet
        from src.schedulers import BiBBDMScheduler
        from src.pipelines.bibbdm import BiBBDMLatentPipeline

        unet = BiBBDMUNet(image_size=32, in_channels=_MIN_CHANNELS, out_channels=2 * _MIN_CHANNELS, model_channels=_MIN_CHANNELS)
        scheduler = BiBBDMScheduler(num_timesteps=1000, objective="dlns")
        vae = _make_mock_vae()

        pipe = BiBBDMLatentPipeline(unet=unet, scheduler=scheduler, vae=vae)

        assert pipe.unet is not None
        assert pipe.scheduler is not None
        assert pipe.vae is not None


# ---------------------------------------------------------------------------
# I2SB Latent Pipeline
# ---------------------------------------------------------------------------


class TestI2SBLatentPipeline:
    def test_is_diffusion_pipeline_subclass(self):
        from src.pipelines.i2sb import I2SBLatentPipeline

        assert issubclass(I2SBLatentPipeline, DiffusionPipeline)

    def test_construction(self):
        from src.models.unet_i2sb import I2SBUNet
        from src.schedulers import I2SBScheduler
        from src.pipelines.i2sb import I2SBLatentPipeline

        unet = I2SBUNet(image_size=32, in_channels=_MIN_CHANNELS, model_channels=_MIN_CHANNELS)
        scheduler = I2SBScheduler(interval=100, beta_max=0.3)
        vae = _make_mock_vae()

        pipe = I2SBLatentPipeline(unet=unet, scheduler=scheduler, vae=vae)

        assert pipe.unet is not None
        assert pipe.scheduler is not None
        assert pipe.vae is not None


# ---------------------------------------------------------------------------
# DDIB Latent Pipeline
# ---------------------------------------------------------------------------


class TestDDIBLatentPipeline:
    def test_is_diffusion_pipeline_subclass(self):
        from src.pipelines.ddib import DDIBLatentPipeline

        assert issubclass(DDIBLatentPipeline, DiffusionPipeline)

    def test_construction(self):
        from src.models.unet_ddib import DDIBUNet
        from src.schedulers import DDIBScheduler
        from src.pipelines.ddib import DDIBLatentPipeline

        source = DDIBUNet(image_size=32, in_channels=_MIN_CHANNELS, model_channels=_MIN_CHANNELS)
        target = DDIBUNet(image_size=32, in_channels=_MIN_CHANNELS, model_channels=_MIN_CHANNELS)
        scheduler = DDIBScheduler(num_train_timesteps=1000, noise_schedule="linear")
        vae = _make_mock_vae()

        pipe = DDIBLatentPipeline(
            source_unet=source,
            target_unet=target,
            scheduler=scheduler,
            vae=vae,
        )

        assert pipe.source_unet is not None
        assert pipe.target_unet is not None
        assert pipe.scheduler is not None
        assert pipe.vae is not None


# ---------------------------------------------------------------------------
# CUT Latent Pipeline
# ---------------------------------------------------------------------------


class TestCUTLatentPipeline:
    def test_is_diffusion_pipeline_subclass(self):
        from src.pipelines.cut import CUTLatentPipeline

        assert issubclass(CUTLatentPipeline, DiffusionPipeline)

    def test_construction(self):
        from src.models.cut_model import CUTGenerator
        from src.pipelines.cut import CUTLatentPipeline

        generator = CUTGenerator(input_nc=_MIN_CHANNELS, output_nc=_MIN_CHANNELS, ngf=32, n_blocks=2)
        vae = _make_mock_vae()

        pipe = CUTLatentPipeline(generator=generator, vae=vae)

        assert pipe.generator is not None
        assert pipe.vae is not None


# ---------------------------------------------------------------------------
# Img2Img-Turbo should NOT have a latent pipeline
# ---------------------------------------------------------------------------


class TestImg2ImgTurboNoLatent:
    def test_no_latent_pipeline_exported(self):
        try:
            from src import pipelines
        except ImportError:
            pytest.skip("img2img_turbo dependencies not installed")

        assert not hasattr(pipelines, "TurboLatentPipeline")
