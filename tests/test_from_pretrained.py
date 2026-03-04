# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for save / load round-trip via ``from_pretrained``.

Each baseline's UNet wrapper now inherits from ``ModelMixin + ConfigMixin``
so that ``save_pretrained`` / ``from_pretrained`` work out of the box.
These tests verify that a freshly-created model can be saved in diffusers
format and reloaded with identical weights and config.
"""

import os
import tempfile

import pytest
import torch

from src.utils.training_utils import save_checkpoint_diffusers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN_CHANNELS = 32  # UNet2DModel requires >= 32 for GroupNorm


def _assert_state_dicts_equal(sd1, sd2):
    """Assert that two state dicts have the same keys and identical tensors."""
    assert set(sd1.keys()) == set(sd2.keys()), "State dict keys differ"
    for key in sd1:
        assert torch.equal(sd1[key], sd2[key]), f"Mismatch at key {key}"


# ---------------------------------------------------------------------------
# DDBM
# ---------------------------------------------------------------------------


class TestDDBMFromPretrained:
    def test_round_trip(self):
        from src.models.unet.unet_ddbm import DDBMUNet
        from src.schedulers import DDBMScheduler

        model = DDBMUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
        scheduler = DDBMScheduler(sigma_min=0.002, sigma_max=80.0, sigma_data=0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(tmpdir, model, scheduler=scheduler, model_name="unet")

            loaded = DDBMUNet.from_pretrained(tmpdir, subfolder="unet")
            loaded_sched = DDBMScheduler.from_pretrained(tmpdir, subfolder="scheduler")

        _assert_state_dicts_equal(model.state_dict(), loaded.state_dict())
        assert loaded.config["image_size"] == 32
        assert loaded.config["in_channels"] == 1
        assert loaded_sched.config["sigma_min"] == 0.002

    def test_pipeline_construction(self):
        from src.models.unet.unet_ddbm import DDBMUNet
        from src.schedulers import DDBMScheduler
        from src.pipelines.ddbm import DDBMPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            model = DDBMUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
            scheduler = DDBMScheduler(sigma_min=0.002, sigma_max=80.0, sigma_data=0.5)
            save_checkpoint_diffusers(tmpdir, model, scheduler=scheduler, model_name="unet")

            unet = DDBMUNet.from_pretrained(tmpdir, subfolder="unet")
            sched = DDBMScheduler.from_pretrained(tmpdir, subfolder="scheduler")
            pipe = DDBMPipeline(unet=unet, scheduler=sched)

        assert pipe.unet is not None
        assert pipe.scheduler is not None


# ---------------------------------------------------------------------------
# DBIM
# ---------------------------------------------------------------------------


class TestDBIMFromPretrained:
    def test_round_trip(self):
        from src.models.unet.unet_dbim import DBIMUNet
        from src.schedulers import DBIMScheduler

        model = DBIMUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
        scheduler = DBIMScheduler(sigma_min=0.002, sigma_max=1.0, sigma_data=0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(tmpdir, model, scheduler=scheduler, model_name="unet")

            loaded = DBIMUNet.from_pretrained(tmpdir, subfolder="unet")
            loaded_sched = DBIMScheduler.from_pretrained(tmpdir, subfolder="scheduler")

        _assert_state_dicts_equal(model.state_dict(), loaded.state_dict())
        assert loaded.config["image_size"] == 32
        assert loaded.config["in_channels"] == 1
        assert loaded_sched.config["sigma_min"] == 0.002

    def test_pipeline_construction(self):
        from src.models.unet.unet_dbim import DBIMUNet
        from src.schedulers import DBIMScheduler
        from src.pipelines.dbim import DBIMPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            model = DBIMUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
            scheduler = DBIMScheduler(sigma_min=0.002, sigma_max=1.0, sigma_data=0.5)
            save_checkpoint_diffusers(tmpdir, model, scheduler=scheduler, model_name="unet")

            unet = DBIMUNet.from_pretrained(tmpdir, subfolder="unet")
            sched = DBIMScheduler.from_pretrained(tmpdir, subfolder="scheduler")
            pipe = DBIMPipeline(unet=unet, scheduler=sched)

        assert pipe.unet is not None
        assert pipe.scheduler is not None


# ---------------------------------------------------------------------------
# CDTSDE
# ---------------------------------------------------------------------------


class TestCDTSDEFromPretrained:
    def test_round_trip(self):
        from src.models.unet.unet_cdtsde import CDTSDEUNet
        from src.schedulers import CDTSDEScheduler

        model = CDTSDEUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
        scheduler = CDTSDEScheduler(num_train_timesteps=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(tmpdir, model, scheduler=scheduler, model_name="unet")

            loaded = CDTSDEUNet.from_pretrained(tmpdir, subfolder="unet")
            loaded_sched = CDTSDEScheduler.from_pretrained(tmpdir, subfolder="scheduler")

        _assert_state_dicts_equal(model.state_dict(), loaded.state_dict())
        assert loaded.config["image_size"] == 32
        assert loaded.config["in_channels"] == 1
        assert loaded_sched.config["num_train_timesteps"] == 100

    def test_pipeline_construction(self):
        from src.models.unet.unet_cdtsde import CDTSDEUNet
        from src.schedulers import CDTSDEScheduler
        from src.pipelines.cdtsde import CDTSDEPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            model = CDTSDEUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
            scheduler = CDTSDEScheduler(num_train_timesteps=100)
            save_checkpoint_diffusers(tmpdir, model, scheduler=scheduler, model_name="unet")

            unet = CDTSDEUNet.from_pretrained(tmpdir, subfolder="unet")
            sched = CDTSDEScheduler.from_pretrained(tmpdir, subfolder="scheduler")
            pipe = CDTSDEPipeline(unet=unet, scheduler=sched)

        assert pipe.unet is not None
        assert pipe.scheduler is not None


# ---------------------------------------------------------------------------
# BiBBDM
# ---------------------------------------------------------------------------


class TestBiBBDMFromPretrained:
    def test_round_trip(self):
        from src.models.unet.unet_bibbdm import BiBBDMUNet
        from src.schedulers import BiBBDMScheduler

        model = BiBBDMUNet(image_size=32, in_channels=1, out_channels=2, model_channels=_MIN_CHANNELS)
        scheduler = BiBBDMScheduler(num_timesteps=1000, objective="dlns")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(tmpdir, model, scheduler=scheduler, model_name="unet")

            loaded = BiBBDMUNet.from_pretrained(tmpdir, subfolder="unet")
            loaded_sched = BiBBDMScheduler.from_pretrained(tmpdir, subfolder="scheduler")

        _assert_state_dicts_equal(model.state_dict(), loaded.state_dict())
        assert loaded.config["out_channels"] == 2
        assert loaded_sched.config["objective"] == "dlns"


# ---------------------------------------------------------------------------
# BDBM
# ---------------------------------------------------------------------------


class TestBDBMFromPretrained:
    def test_round_trip(self):
        from src.models.unet.unet_bdbm import BDBMUNet
        from src.schedulers import BDBMScheduler

        model = BDBMUNet(
            image_size=32,
            in_channels=1,
            out_channels=2,
            model_channels=_MIN_CHANNELS,
            condition_mode="dual",
        )
        scheduler = BDBMScheduler(num_timesteps=1000, objective="both")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(tmpdir, model, scheduler=scheduler, model_name="unet")

            loaded = BDBMUNet.from_pretrained(tmpdir, subfolder="unet")
            loaded_sched = BDBMScheduler.from_pretrained(tmpdir, subfolder="scheduler")

        _assert_state_dicts_equal(model.state_dict(), loaded.state_dict())
        assert loaded.config["out_channels"] == 2
        assert loaded_sched.config["objective"] == "both"


# ---------------------------------------------------------------------------
# I2SB
# ---------------------------------------------------------------------------


class TestI2SBFromPretrained:
    def test_round_trip(self):
        from src.models.unet.unet_i2sb import I2SBUNet
        from src.schedulers import I2SBScheduler

        model = I2SBUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
        scheduler = I2SBScheduler(interval=100, beta_max=0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(tmpdir, model, scheduler=scheduler, model_name="unet")

            loaded = I2SBUNet.from_pretrained(tmpdir, subfolder="unet")
            loaded_sched = I2SBScheduler.from_pretrained(tmpdir, subfolder="scheduler")

        _assert_state_dicts_equal(model.state_dict(), loaded.state_dict())
        assert loaded.config["image_size"] == 32
        assert loaded_sched.config["interval"] == 100


# ---------------------------------------------------------------------------
# DDIB
# ---------------------------------------------------------------------------


class TestDDIBFromPretrained:
    def test_round_trip(self):
        from src.models.unet.unet_ddib import DDIBUNet
        from src.schedulers import DDIBScheduler

        model = DDIBUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS, learn_sigma=False)
        scheduler = DDIBScheduler(num_train_timesteps=1000, noise_schedule="linear")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(tmpdir, model, scheduler=scheduler, model_name="unet")

            loaded = DDIBUNet.from_pretrained(tmpdir, subfolder="unet")
            loaded_sched = DDIBScheduler.from_pretrained(tmpdir, subfolder="scheduler")

        _assert_state_dicts_equal(model.state_dict(), loaded.state_dict())
        assert loaded.config["learn_sigma"] is False
        assert loaded_sched.config["noise_schedule"] == "linear"


# ---------------------------------------------------------------------------
# CUT
# ---------------------------------------------------------------------------


class TestCUTFromPretrained:
    def test_round_trip(self):
        from src.models.cut_model import CUTGenerator

        model = CUTGenerator(input_nc=1, output_nc=1, ngf=32, n_blocks=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(tmpdir, model, scheduler=None, model_name="unet")

            loaded = CUTGenerator.from_pretrained(tmpdir, subfolder="unet")

        _assert_state_dicts_equal(model.state_dict(), loaded.state_dict())
        assert loaded.config["ngf"] == 32
        assert loaded.config["n_blocks"] == 2

    def test_cut_pipeline_inherits_diffusion_pipeline(self):
        from diffusers import DiffusionPipeline
        from src.pipelines.cut import CUTPipeline

        assert issubclass(CUTPipeline, DiffusionPipeline)


# ---------------------------------------------------------------------------
# Pipeline.from_pretrained() one-liner tests
# ---------------------------------------------------------------------------


class TestPipelineFromPretrained:
    """Verify that ``Pipeline.from_pretrained(path)`` works for all baselines."""

    def test_ddbm_pipeline_from_pretrained(self):
        from src.models.unet.unet_ddbm import DDBMUNet
        from src.schedulers import DDBMScheduler
        from src.pipelines.ddbm import DDBMPipeline

        model = DDBMUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
        scheduler = DDBMScheduler(sigma_min=0.002, sigma_max=80.0, sigma_data=0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(
                tmpdir, model, scheduler=scheduler, model_name="unet",
                pipeline_class_name="DDBMPipeline",
            )
            pipe = DDBMPipeline.from_pretrained(tmpdir)

        assert pipe.unet is not None
        assert pipe.scheduler is not None
        _assert_state_dicts_equal(model.state_dict(), pipe.unet.state_dict())

    def test_dbim_pipeline_from_pretrained(self):
        from src.models.unet.unet_dbim import DBIMUNet
        from src.schedulers import DBIMScheduler
        from src.pipelines.dbim import DBIMPipeline

        model = DBIMUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
        scheduler = DBIMScheduler(sigma_min=0.002, sigma_max=1.0, sigma_data=0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(
                tmpdir,
                model,
                scheduler=scheduler,
                model_name="unet",
                pipeline_class_name="DBIMPipeline",
            )
            pipe = DBIMPipeline.from_pretrained(tmpdir)

        assert pipe.unet is not None
        assert pipe.scheduler is not None
        _assert_state_dicts_equal(model.state_dict(), pipe.unet.state_dict())

    def test_cdtsde_pipeline_from_pretrained(self):
        from src.models.unet.unet_cdtsde import CDTSDEUNet
        from src.schedulers import CDTSDEScheduler
        from src.pipelines.cdtsde import CDTSDEPipeline

        model = CDTSDEUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
        scheduler = CDTSDEScheduler(num_train_timesteps=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(
                tmpdir, model, scheduler=scheduler, model_name="unet",
                pipeline_class_name="CDTSDEPipeline",
            )
            pipe = CDTSDEPipeline.from_pretrained(tmpdir)

        assert pipe.unet is not None
        assert pipe.scheduler is not None
        _assert_state_dicts_equal(model.state_dict(), pipe.unet.state_dict())

    def test_bibbdm_pipeline_from_pretrained(self):
        from src.models.unet.unet_bibbdm import BiBBDMUNet
        from src.schedulers import BiBBDMScheduler
        from src.pipelines.bibbdm import BiBBDMPipeline

        model = BiBBDMUNet(image_size=32, in_channels=1, out_channels=2, model_channels=_MIN_CHANNELS)
        scheduler = BiBBDMScheduler(num_timesteps=1000, objective="dlns")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(
                tmpdir, model, scheduler=scheduler, model_name="unet",
                pipeline_class_name="BiBBDMPipeline",
            )
            pipe = BiBBDMPipeline.from_pretrained(tmpdir)

        assert pipe.unet is not None
        assert pipe.scheduler is not None

    def test_bdbm_pipeline_from_pretrained(self):
        from src.models.unet.unet_bdbm import BDBMUNet
        from src.schedulers import BDBMScheduler
        from src.pipelines.bdbm import BDBMPipeline

        model = BDBMUNet(
            image_size=32,
            in_channels=1,
            out_channels=2,
            model_channels=_MIN_CHANNELS,
            condition_mode="dual",
        )
        scheduler = BDBMScheduler(num_timesteps=1000, objective="both")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(
                tmpdir, model, scheduler=scheduler, model_name="unet",
                pipeline_class_name="BDBMPipeline",
            )
            pipe = BDBMPipeline.from_pretrained(tmpdir)

        assert pipe.unet is not None
        assert pipe.scheduler is not None

    def test_i2sb_pipeline_from_pretrained(self):
        from src.models.unet.unet_i2sb import I2SBUNet
        from src.schedulers import I2SBScheduler
        from src.pipelines.i2sb import I2SBPipeline

        model = I2SBUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
        scheduler = I2SBScheduler(interval=100, beta_max=0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(
                tmpdir, model, scheduler=scheduler, model_name="unet",
                pipeline_class_name="I2SBPipeline",
            )
            pipe = I2SBPipeline.from_pretrained(tmpdir)

        assert pipe.unet is not None
        assert pipe.scheduler is not None

    def test_cut_pipeline_from_pretrained(self):
        from src.models.cut_model import CUTGenerator
        from src.pipelines.cut import CUTPipeline

        model = CUTGenerator(input_nc=1, output_nc=1, ngf=32, n_blocks=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint_diffusers(
                tmpdir, model, scheduler=None, model_name="generator",
                pipeline_class_name="CUTPipeline",
            )
            pipe = CUTPipeline.from_pretrained(tmpdir)

        assert pipe.generator is not None
        _assert_state_dicts_equal(model.state_dict(), pipe.generator.state_dict())

    def test_ddib_pipeline_from_pretrained(self):
        from src.models.unet.unet_ddib import DDIBUNet
        from src.schedulers import DDIBScheduler
        from src.pipelines.ddib import DDIBPipeline

        source = DDIBUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
        target = DDIBUNet(image_size=32, in_channels=1, model_channels=_MIN_CHANNELS)
        scheduler = DDIBScheduler(num_train_timesteps=1000, noise_schedule="linear")

        with tempfile.TemporaryDirectory() as tmpdir:
            orig = DDIBPipeline(source_unet=source, target_unet=target, scheduler=scheduler)
            orig.save_pretrained(tmpdir)
            pipe = DDIBPipeline.from_pretrained(tmpdir)

        assert pipe.source_unet is not None
        assert pipe.target_unet is not None
        assert pipe.scheduler is not None
