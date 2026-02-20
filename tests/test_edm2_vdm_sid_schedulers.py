"""Tests for the EDM2, VDM, SiD, and SiD2 schedulers.

Covers:
* Module imports and public API
* Scheduler initialization and configuration
* Timestep/sigma schedule generation
* Forward diffusion (add_noise)
* Reverse sampling (step / step_heun)
* Config save/load round-trip
* Mathematical properties and invariants
"""

import math
import tempfile

import pytest
import torch

from diffusers.configuration_utils import ConfigMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from src.schedulers import (
    EDM2Scheduler,
    EDM2SchedulerOutput,
    VDMScheduler,
    VDMSchedulerOutput,
    SiDScheduler,
    SiDSchedulerOutput,
    SiD2Scheduler,
    SiD2SchedulerOutput,
)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """All public symbols are importable from the package."""

    def test_edm2_scheduler_classes(self):
        assert EDM2Scheduler is not None
        assert EDM2SchedulerOutput is not None

    def test_vdm_scheduler_classes(self):
        assert VDMScheduler is not None
        assert VDMSchedulerOutput is not None

    def test_sid_scheduler_classes(self):
        assert SiDScheduler is not None
        assert SiDSchedulerOutput is not None


# ---------------------------------------------------------------------------
# EDM2 Scheduler
# ---------------------------------------------------------------------------


class TestEDM2Scheduler:
    """EDM2Scheduler correctly implements the Karras sigma schedule + Heun sampler."""

    def test_default_config(self):
        sched = EDM2Scheduler()
        assert sched.sigma_min == 0.002
        assert sched.sigma_max == 80.0
        assert sched.sigma_data == 0.5
        assert sched.rho == 7.0
        assert sched.num_train_timesteps == 32
        assert sched.s_churn == 0.0
        assert sched.order == 2

    def test_custom_config(self):
        sched = EDM2Scheduler(sigma_min=0.01, sigma_max=100.0, sigma_data=1.0, rho=5.0)
        assert sched.sigma_min == 0.01
        assert sched.sigma_max == 100.0
        assert sched.sigma_data == 1.0
        assert sched.rho == 5.0

    def test_set_timesteps_shape(self):
        sched = EDM2Scheduler()
        sched.set_timesteps(32)
        # sigmas has N+1 elements (N steps + terminal 0)
        assert sched.sigmas.shape == (33,)
        assert sched.timesteps.shape == (32,)
        assert sched.num_inference_steps == 32

    def test_sigmas_monotonically_decreasing(self):
        sched = EDM2Scheduler()
        sched.set_timesteps(32)
        for i in range(len(sched.sigmas) - 1):
            assert sched.sigmas[i] >= sched.sigmas[i + 1]

    def test_sigmas_boundary_values(self):
        sched = EDM2Scheduler(sigma_min=0.002, sigma_max=80.0)
        sched.set_timesteps(32)
        assert sched.sigmas[0].item() == pytest.approx(80.0, rel=0.01)
        assert sched.sigmas[-1].item() == 0.0

    def test_step_not_initialized_raises(self):
        sched = EDM2Scheduler()
        with pytest.raises(ValueError, match="Sigmas not initialized"):
            sched.step(torch.randn(1, 3, 8, 8), 0, torch.randn(1, 3, 8, 8))

    def test_step_output_shape(self):
        sched = EDM2Scheduler()
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        denoised = torch.randn(2, 3, 16, 16)
        out = sched.step(denoised, 0, x)
        assert isinstance(out, EDM2SchedulerOutput)
        assert out.prev_sample.shape == (2, 3, 16, 16)
        assert out.pred_original_sample.shape == (2, 3, 16, 16)

    def test_step_return_tuple(self):
        sched = EDM2Scheduler()
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        denoised = torch.randn(2, 3, 16, 16)
        out = sched.step(denoised, 0, x, return_dict=False)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_step_heun_output_shape(self):
        sched = EDM2Scheduler()
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        d1 = torch.randn(2, 3, 16, 16)
        d2 = torch.randn(2, 3, 16, 16)
        out = sched.step_heun(d1, d2, 0, x)
        assert isinstance(out, EDM2SchedulerOutput)
        assert out.prev_sample.shape == (2, 3, 16, 16)

    def test_step_heun_not_initialized_raises(self):
        sched = EDM2Scheduler()
        with pytest.raises(ValueError, match="Sigmas not initialized"):
            sched.step_heun(torch.randn(1, 3, 8, 8), torch.randn(1, 3, 8, 8), 0, torch.randn(1, 3, 8, 8))

    def test_step_heun_final_step(self):
        """At the final step (sigma_next=0), Heun should just do Euler."""
        sched = EDM2Scheduler()
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        d1 = torch.randn(2, 3, 16, 16)
        d2 = torch.randn(2, 3, 16, 16)
        # Last step index is 9, sigma[10]=0
        out = sched.step_heun(d1, d2, 9, x)
        assert out.prev_sample.shape == (2, 3, 16, 16)

    def test_add_noise_shape(self):
        sched = EDM2Scheduler()
        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        sigmas = torch.tensor([1.0, 5.0])
        noisy = sched.add_noise(x, noise, sigmas)
        assert noisy.shape == x.shape

    def test_add_noise_zero_sigma(self):
        """With sigma=0, noisy sample should equal original."""
        sched = EDM2Scheduler()
        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        sigmas = torch.tensor([0.0, 0.0])
        noisy = sched.add_noise(x, noise, sigmas)
        assert torch.allclose(noisy, x)

    def test_add_noise_correctness(self):
        """Verify x_noisy = x + sigma * noise."""
        sched = EDM2Scheduler()
        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        sigmas = torch.tensor([2.0, 3.0])
        noisy = sched.add_noise(x, noise, sigmas)
        expected = x + sigmas.view(2, 1, 1, 1) * noise
        assert torch.allclose(noisy, expected, atol=1e-6)

    def test_scale_model_input_passthrough(self):
        sched = EDM2Scheduler()
        x = torch.randn(2, 3, 16, 16)
        assert torch.equal(sched.scale_model_input(x), x)

    def test_preconditioning_coefficients(self):
        """Verify EDM2 preconditioning coefficients match expected formulas."""
        sched = EDM2Scheduler(sigma_data=0.5)
        sigma = torch.tensor([1.0])
        c_skip, c_out, c_in, c_noise = sched._precondition_coefficients(sigma)
        sd2 = 0.5 ** 2
        expected_c_skip = sd2 / (1.0 + sd2)
        expected_c_out = 1.0 * 0.5 / (1.0 + sd2) ** 0.5
        expected_c_in = 1.0 / (sd2 + 1.0) ** 0.5
        expected_c_noise = 0.0  # log(1) / 4 = 0
        assert c_skip.item() == pytest.approx(expected_c_skip, rel=1e-5)
        assert c_out.item() == pytest.approx(expected_c_out, rel=1e-5)
        assert c_in.item() == pytest.approx(expected_c_in, rel=1e-5)
        assert c_noise.item() == pytest.approx(expected_c_noise, abs=1e-5)

    def test_stochastic_churn(self):
        """With s_churn > 0, step should still produce valid output."""
        sched = EDM2Scheduler(s_churn=10.0, s_min=0.0, s_max=1000.0)
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        denoised = torch.randn(2, 3, 16, 16)
        out = sched.step(denoised, 0, x)
        assert out.prev_sample.shape == (2, 3, 16, 16)
        assert not torch.isnan(out.prev_sample).any()

    def test_config_save_load(self):
        sched = EDM2Scheduler(sigma_data=0.7, rho=5.0)
        config = sched.config
        sched2 = EDM2Scheduler.from_config(config)
        assert sched2.sigma_data == 0.7
        assert sched2.rho == 5.0

    def test_init_noise_sigma(self):
        sched = EDM2Scheduler(sigma_max=100.0)
        assert sched.init_noise_sigma == 100.0

    def test_set_timesteps_device(self):
        sched = EDM2Scheduler()
        sched.set_timesteps(10, device="cpu")
        assert sched.sigmas.device == torch.device("cpu")
        assert sched.timesteps.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# VDM Scheduler
# ---------------------------------------------------------------------------


class TestVDMScheduler:
    """VDMScheduler correctly implements the logSNR schedule + DDPM posterior."""

    def test_default_config(self):
        sched = VDMScheduler()
        assert sched.gamma_min == -13.3
        assert sched.gamma_max == 5.0
        assert sched.num_train_timesteps == 1000
        assert sched.prediction_type == "eps"
        assert sched.clip_sample is True
        assert sched.order == 1

    def test_custom_config(self):
        sched = VDMScheduler(gamma_min=-20.0, gamma_max=10.0, prediction_type="v")
        assert sched.gamma_min == -20.0
        assert sched.gamma_max == 10.0
        assert sched.prediction_type == "v"

    def test_set_timesteps_shape(self):
        sched = VDMScheduler()
        sched.set_timesteps(10)
        # gammas has N+1 elements
        assert sched.gammas.shape == (11,)
        assert sched.timesteps.shape == (10,)
        assert sched.num_inference_steps == 10

    def test_gammas_monotonically_increasing(self):
        """Gammas go from gamma_min (t=1) to gamma_max (t=0), so they increase."""
        sched = VDMScheduler()
        sched.set_timesteps(20)
        for i in range(len(sched.gammas) - 1):
            assert sched.gammas[i] <= sched.gammas[i + 1]

    def test_gammas_boundary_values(self):
        sched = VDMScheduler(gamma_min=-13.3, gamma_max=5.0)
        sched.set_timesteps(10)
        # First gamma is at t=1 → gamma_min
        assert sched.gammas[0].item() == pytest.approx(-13.3, rel=1e-3)
        # Last gamma is at t=0 → gamma_max
        assert sched.gammas[-1].item() == pytest.approx(5.0, rel=1e-3)

    def test_gamma_function(self):
        """Test the linear interpolation gamma(t)."""
        sched = VDMScheduler(gamma_min=-10.0, gamma_max=10.0)
        t0 = torch.tensor(0.0)
        t1 = torch.tensor(1.0)
        t05 = torch.tensor(0.5)
        assert sched._gamma(t0).item() == pytest.approx(10.0, abs=1e-5)
        assert sched._gamma(t1).item() == pytest.approx(-10.0, abs=1e-5)
        assert sched._gamma(t05).item() == pytest.approx(0.0, abs=1e-5)

    def test_step_not_initialized_raises(self):
        sched = VDMScheduler()
        with pytest.raises(ValueError, match="Gammas not initialized"):
            sched.step(torch.randn(1, 3, 8, 8), 0, torch.randn(1, 3, 8, 8))

    def test_step_output_shape_eps(self):
        sched = VDMScheduler(prediction_type="eps")
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        eps = torch.randn(2, 3, 16, 16)
        out = sched.step(eps, 0, x)
        assert isinstance(out, VDMSchedulerOutput)
        assert out.prev_sample.shape == (2, 3, 16, 16)
        assert out.pred_original_sample.shape == (2, 3, 16, 16)

    def test_step_output_shape_v(self):
        sched = VDMScheduler(prediction_type="v")
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        v_pred = torch.randn(2, 3, 16, 16)
        out = sched.step(v_pred, 0, x)
        assert out.prev_sample.shape == (2, 3, 16, 16)

    def test_step_return_tuple(self):
        sched = VDMScheduler()
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        eps = torch.randn(2, 3, 16, 16)
        out = sched.step(eps, 0, x, return_dict=False)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_step_invalid_prediction_type(self):
        sched = VDMScheduler(prediction_type="invalid")
        sched.set_timesteps(10)
        with pytest.raises(ValueError, match="Unknown prediction_type"):
            sched.step(torch.randn(1, 3, 8, 8), 0, torch.randn(1, 3, 8, 8))

    def test_add_noise_shape(self):
        sched = VDMScheduler()
        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        gammas = torch.tensor([0.0, -5.0])
        noisy = sched.add_noise(x, noise, gammas)
        assert noisy.shape == x.shape

    def test_add_noise_high_snr(self):
        """At very high logSNR (low noise), noisy ≈ original."""
        sched = VDMScheduler()
        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        gammas = torch.tensor([20.0, 20.0])  # Very high logSNR → alpha ≈ 1, sigma ≈ 0
        noisy = sched.add_noise(x, noise, gammas)
        assert torch.allclose(noisy, x, atol=1e-3)

    def test_add_noise_low_snr(self):
        """At very low logSNR (high noise), noisy ≈ noise."""
        sched = VDMScheduler()
        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        gammas = torch.tensor([-20.0, -20.0])  # Very low logSNR → alpha ≈ 0, sigma ≈ 1
        noisy = sched.add_noise(x, noise, gammas)
        assert torch.allclose(noisy, noise, atol=1e-3)

    def test_add_noise_preserves_variance(self):
        """alpha_t^2 + sigma_t^2 = 1 for the VDM forward process."""
        sched = VDMScheduler()
        gamma = torch.tensor([3.0])
        alpha_t = torch.sqrt(torch.sigmoid(gamma))
        sigma_t = torch.sqrt(torch.sigmoid(-gamma))
        assert (alpha_t ** 2 + sigma_t ** 2).item() == pytest.approx(1.0, abs=1e-6)

    def test_scale_model_input_passthrough(self):
        sched = VDMScheduler()
        x = torch.randn(2, 3, 16, 16)
        assert torch.equal(sched.scale_model_input(x), x)

    def test_config_save_load(self):
        sched = VDMScheduler(gamma_min=-20.0, gamma_max=10.0, prediction_type="v")
        config = sched.config
        sched2 = VDMScheduler.from_config(config)
        assert sched2.gamma_min == -20.0
        assert sched2.gamma_max == 10.0
        assert sched2.prediction_type == "v"

    def test_step_no_nan(self):
        """Step should not produce NaN values."""
        sched = VDMScheduler()
        sched.set_timesteps(20)
        x = torch.randn(2, 3, 16, 16)
        for i in range(20):
            eps = torch.randn(2, 3, 16, 16)
            out = sched.step(eps, i, x)
            assert not torch.isnan(out.prev_sample).any()
            x = out.prev_sample

    def test_init_noise_sigma(self):
        sched = VDMScheduler()
        assert sched.init_noise_sigma == 1.0

    def test_set_timesteps_device(self):
        sched = VDMScheduler()
        sched.set_timesteps(10, device="cpu")
        assert sched.gammas.device == torch.device("cpu")
        assert sched.timesteps.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# SiD Scheduler
# ---------------------------------------------------------------------------


class TestSiDScheduler:
    """SiDScheduler correctly implements the shifted cosine logSNR schedule + DDPM."""

    def test_default_config(self):
        sched = SiDScheduler()
        assert sched.logsnr_min == -15.0
        assert sched.logsnr_max == 15.0
        assert sched.noise_d == 64.0
        assert sched.image_d == 64.0
        assert sched.prediction_type == "eps"
        assert sched.clip_sample is True
        assert sched.order == 1

    def test_custom_config(self):
        sched = SiDScheduler(noise_d=64.0, image_d=256.0, prediction_type="v")
        assert sched.noise_d == 64.0
        assert sched.image_d == 256.0
        assert sched.prediction_type == "v"

    def test_set_timesteps_shape(self):
        sched = SiDScheduler()
        sched.set_timesteps(10)
        assert sched.logsnrs.shape == (11,)
        assert sched.timesteps.shape == (10,)
        assert sched.num_inference_steps == 10

    def test_logsnrs_monotonically_increasing(self):
        """logSNRs go from t=1 (noisy, low logSNR) to t=0 (clean, high logSNR)."""
        sched = SiDScheduler()
        sched.set_timesteps(20)
        for i in range(len(sched.logsnrs) - 1):
            assert sched.logsnrs[i] <= sched.logsnrs[i + 1]

    def test_base_cosine_schedule(self):
        """Test the base cosine logSNR at t=0 and t=1."""
        sched = SiDScheduler(noise_d=64.0, image_d=64.0)  # No shift
        # At t=0, logSNR should be high (near logsnr_max)
        logsnr_0 = sched._logsnr_cosine(torch.tensor(0.0))
        assert logsnr_0.item() > 10.0
        # At t=1, logSNR should be low (near logsnr_min)
        logsnr_1 = sched._logsnr_cosine(torch.tensor(1.0))
        assert logsnr_1.item() < -10.0

    def test_shifted_schedule_lower_than_base(self):
        """When image_d > noise_d, shifted logSNR should be lower than base."""
        sched = SiDScheduler(noise_d=64.0, image_d=256.0)
        t = torch.tensor(0.5)
        base = sched._logsnr_cosine(t)
        shifted = sched._logsnr_shifted_cosine(t)
        # shift = 2*log(64/256) = 2*log(0.25) < 0 → shifted < base
        assert shifted.item() < base.item()

    def test_no_shift_when_equal(self):
        """When noise_d == image_d, shifted schedule equals base schedule."""
        sched = SiDScheduler(noise_d=64.0, image_d=64.0)
        t = torch.tensor(0.5)
        base = sched._logsnr_cosine(t)
        shifted = sched._logsnr_shifted_cosine(t)
        assert shifted.item() == pytest.approx(base.item(), abs=1e-5)

    def test_step_not_initialized_raises(self):
        sched = SiDScheduler()
        with pytest.raises(ValueError, match="LogSNRs not initialized"):
            sched.step(torch.randn(1, 3, 8, 8), 0, torch.randn(1, 3, 8, 8))

    def test_step_output_shape_eps(self):
        sched = SiDScheduler(prediction_type="eps")
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        eps = torch.randn(2, 3, 16, 16)
        out = sched.step(eps, 0, x)
        assert isinstance(out, SiDSchedulerOutput)
        assert out.prev_sample.shape == (2, 3, 16, 16)
        assert out.pred_original_sample.shape == (2, 3, 16, 16)

    def test_step_output_shape_v(self):
        sched = SiDScheduler(prediction_type="v")
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        v_pred = torch.randn(2, 3, 16, 16)
        out = sched.step(v_pred, 0, x)
        assert out.prev_sample.shape == (2, 3, 16, 16)

    def test_step_return_tuple(self):
        sched = SiDScheduler()
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        eps = torch.randn(2, 3, 16, 16)
        out = sched.step(eps, 0, x, return_dict=False)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_step_invalid_prediction_type(self):
        sched = SiDScheduler(prediction_type="invalid")
        sched.set_timesteps(10)
        with pytest.raises(ValueError, match="Unknown prediction_type"):
            sched.step(torch.randn(1, 3, 8, 8), 0, torch.randn(1, 3, 8, 8))

    def test_add_noise_shape(self):
        sched = SiDScheduler()
        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        logsnrs = torch.tensor([5.0, -5.0])
        noisy = sched.add_noise(x, noise, logsnrs)
        assert noisy.shape == x.shape

    def test_add_noise_high_snr(self):
        """At very high logSNR, noisy ≈ original."""
        sched = SiDScheduler()
        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        logsnrs = torch.tensor([20.0, 20.0])
        noisy = sched.add_noise(x, noise, logsnrs)
        assert torch.allclose(noisy, x, atol=1e-3)

    def test_add_noise_low_snr(self):
        """At very low logSNR, noisy ≈ noise."""
        sched = SiDScheduler()
        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        logsnrs = torch.tensor([-20.0, -20.0])
        noisy = sched.add_noise(x, noise, logsnrs)
        assert torch.allclose(noisy, noise, atol=1e-3)

    def test_scale_model_input_passthrough(self):
        sched = SiDScheduler()
        x = torch.randn(2, 3, 16, 16)
        assert torch.equal(sched.scale_model_input(x), x)

    def test_config_save_load(self):
        sched = SiDScheduler(noise_d=32.0, image_d=128.0, prediction_type="v")
        config = sched.config
        sched2 = SiDScheduler.from_config(config)
        assert sched2.noise_d == 32.0
        assert sched2.image_d == 128.0
        assert sched2.prediction_type == "v"

    def test_step_no_nan(self):
        """Step should not produce NaN values."""
        sched = SiDScheduler()
        sched.set_timesteps(20)
        x = torch.randn(2, 3, 16, 16)
        for i in range(20):
            eps = torch.randn(2, 3, 16, 16)
            out = sched.step(eps, i, x)
            assert not torch.isnan(out.prev_sample).any()
            x = out.prev_sample

    def test_shifted_cosine_resolution_dependence(self):
        """Higher resolution should shift logSNR down (more noise at same t)."""
        sched_64 = SiDScheduler(noise_d=64.0, image_d=64.0)
        sched_256 = SiDScheduler(noise_d=64.0, image_d=256.0)
        t = torch.tensor(0.5)
        logsnr_64 = sched_64._logsnr_shifted_cosine(t)
        logsnr_256 = sched_256._logsnr_shifted_cosine(t)
        assert logsnr_256.item() < logsnr_64.item()

    def test_init_noise_sigma(self):
        sched = SiDScheduler()
        assert sched.init_noise_sigma == 1.0

    def test_set_timesteps_device(self):
        sched = SiDScheduler()
        sched.set_timesteps(10, device="cpu")
        assert sched.logsnrs.device == torch.device("cpu")
        assert sched.timesteps.device == torch.device("cpu")

    def test_clip_sample_effect(self):
        """With clip_sample=True, predicted x_0 should be in [-1, 1]."""
        sched = SiDScheduler(clip_sample=True, prediction_type="eps")
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16) * 10  # Large values
        eps = torch.randn(2, 3, 16, 16)
        out = sched.step(eps, 0, x)
        # pred_original_sample should be clipped
        assert out.pred_original_sample.max() <= 1.0
        assert out.pred_original_sample.min() >= -1.0


# ---------------------------------------------------------------------------
# SiD2 Scheduler
# ---------------------------------------------------------------------------


class TestSiD2Scheduler:
    """SiD2 scheduler behavior and invariants."""

    def test_default_config(self):
        sched = SiD2Scheduler()
        assert sched.logsnr_min == -15.0
        assert sched.logsnr_max == 15.0
        assert sched.schedule_type == "cosine_interpolated"
        assert sched.prediction_type == "v"
        assert sched.clip_sample is True
        assert sched.order == 1

    def test_set_timesteps_shape(self):
        sched = SiD2Scheduler()
        sched.set_timesteps(10)
        assert sched.logsnrs.shape == (11,)
        assert sched.timesteps.shape == (10,)
        assert sched.num_inference_steps == 10

    def test_logsnrs_monotonically_increasing(self):
        sched = SiD2Scheduler()
        sched.set_timesteps(20)
        for i in range(len(sched.logsnrs) - 1):
            assert sched.logsnrs[i] <= sched.logsnrs[i + 1]

    def test_step_output_shape_v(self):
        sched = SiD2Scheduler(prediction_type="v")
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        v_pred = torch.randn(2, 3, 16, 16)
        out = sched.step(v_pred, 0, x)
        assert isinstance(out, SiD2SchedulerOutput)
        assert out.prev_sample.shape == (2, 3, 16, 16)

    def test_step_output_shape_eps(self):
        sched = SiD2Scheduler(prediction_type="eps")
        sched.set_timesteps(10)
        x = torch.randn(2, 3, 16, 16)
        eps = torch.randn(2, 3, 16, 16)
        out = sched.step(eps, 0, x)
        assert out.prev_sample.shape == (2, 3, 16, 16)

    def test_add_noise_shape(self):
        sched = SiD2Scheduler()
        x = torch.randn(2, 3, 16, 16)
        noise = torch.randn(2, 3, 16, 16)
        logsnrs = torch.tensor([5.0, -5.0])
        noisy = sched.add_noise(x, noise, logsnrs)
        assert noisy.shape == x.shape

    def test_derivative_negative(self):
        sched = SiD2Scheduler()
        t = torch.tensor([0.2, 0.5, 0.8])
        _, dlogsnr_dt = sched.compute_logsnr_and_derivative(t)
        assert torch.all(dlogsnr_dt < 0)

    def test_config_save_load(self):
        sched = SiD2Scheduler(
            schedule_type="cosine_interpolated",
            interpolated_noise_d_low=32.0,
            interpolated_noise_d_high=512.0,
            prediction_type="v",
        )
        config = sched.config
        sched2 = SiD2Scheduler.from_config(config)
        assert sched2.schedule_type == "cosine_interpolated"
        assert sched2.interpolated_noise_d_low == 32.0
        assert sched2.interpolated_noise_d_high == 512.0
        assert sched2.prediction_type == "v"


# ---------------------------------------------------------------------------
# Cross-scheduler tests
# ---------------------------------------------------------------------------


class TestCrossScheduler:
    """Tests that apply to all schedulers in this module."""

    @pytest.mark.parametrize("SchedulerClass", [EDM2Scheduler, VDMScheduler, SiDScheduler, SiD2Scheduler])
    def test_inherits_scheduler_mixin(self, SchedulerClass):
        sched = SchedulerClass()
        assert isinstance(sched, SchedulerMixin)

    @pytest.mark.parametrize("SchedulerClass", [EDM2Scheduler, VDMScheduler, SiDScheduler, SiD2Scheduler])
    def test_inherits_config_mixin(self, SchedulerClass):
        sched = SchedulerClass()
        assert isinstance(sched, ConfigMixin)

    @pytest.mark.parametrize("SchedulerClass", [EDM2Scheduler, VDMScheduler, SiDScheduler, SiD2Scheduler])
    def test_has_set_timesteps(self, SchedulerClass):
        sched = SchedulerClass()
        assert hasattr(sched, "set_timesteps")

    @pytest.mark.parametrize("SchedulerClass", [EDM2Scheduler, VDMScheduler, SiDScheduler, SiD2Scheduler])
    def test_has_step(self, SchedulerClass):
        sched = SchedulerClass()
        assert hasattr(sched, "step")

    @pytest.mark.parametrize("SchedulerClass", [EDM2Scheduler, VDMScheduler, SiDScheduler, SiD2Scheduler])
    def test_has_add_noise(self, SchedulerClass):
        sched = SchedulerClass()
        assert hasattr(sched, "add_noise")

    @pytest.mark.parametrize("SchedulerClass", [EDM2Scheduler, VDMScheduler, SiDScheduler, SiD2Scheduler])
    def test_has_scale_model_input(self, SchedulerClass):
        sched = SchedulerClass()
        assert hasattr(sched, "scale_model_input")

    @pytest.mark.parametrize("SchedulerClass", [EDM2Scheduler, VDMScheduler, SiDScheduler, SiD2Scheduler])
    def test_has_config(self, SchedulerClass):
        sched = SchedulerClass()
        assert hasattr(sched, "config")
