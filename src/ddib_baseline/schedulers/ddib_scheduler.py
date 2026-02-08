# Copyright 2024 The DDIB Authors and The Hugging Face Team.
# Licensed under the MIT License (the "License");
#
# Gaussian diffusion scheduler with DDIM forward and reverse sampling for
# Dual Diffusion Implicit Bridges (DDIB).
# Based on: https://arxiv.org/abs/2203.08382

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class DDIBSchedulerOutput(BaseOutput):
    """
    Output class for the DDIB scheduler's ``step`` function.

    Args:
        prev_sample: Computed sample ``(x_{t-1})`` of previous timestep.
        pred_original_sample: The predicted denoised sample ``(x_0)``.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def _get_named_beta_schedule(schedule_name: str, num_diffusion_steps: int) -> np.ndarray:
    """Return a beta schedule matching the guided-diffusion conventions."""
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    elif schedule_name == "cosine":
        max_beta = 0.999
        s = 0.008

        def alpha_bar(t):
            return np.cos((t + s) / (1 + s) * np.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas, dtype=np.float64)
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class DDIBScheduler(SchedulerMixin, ConfigMixin):
    """
    Gaussian diffusion scheduler with DDIM sampling for DDIB.

    DDIB uses standard Gaussian diffusion with DDIM deterministic sampling to
    *encode* source images to a shared latent space (reverse ODE) and *decode*
    latent representations to the target domain (forward ODE).

    Args:
        num_train_timesteps: Number of diffusion steps.
        noise_schedule: Beta schedule name (``"linear"`` or ``"cosine"``).
        learn_sigma: Whether the model predicts variance as well.
        predict_xstart: If ``True`` model predicts ``x_0`` directly; else ``eps``.
        rescale_timesteps: Scale timesteps to [0, 1000] regardless of actual count.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        noise_schedule: str = "linear",
        learn_sigma: bool = False,
        predict_xstart: bool = False,
        rescale_timesteps: bool = False,
    ):
        betas = _get_named_beta_schedule(noise_schedule, num_train_timesteps)
        betas = np.array(betas, dtype=np.float64)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_train_timesteps = num_train_timesteps
        self.learn_sigma = learn_sigma
        self.predict_xstart = predict_xstart
        self.rescale_timesteps = rescale_timesteps

        # Convert to tensors stored as float64 buffers
        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).float()
        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).float()

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # State
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None
        self.init_noise_sigma = 1.0

    # ------------------------------------------------------------------
    # Timestep management
    # ------------------------------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """Set the discrete timesteps used for the diffusion chain."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_train_timesteps)
        return t

    # ------------------------------------------------------------------
    # Core q-sample (forward noising)
    # ------------------------------------------------------------------

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """q(x_t | x_0): add noise to clean samples."""
        device = original_samples.device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps]
        while sqrt_alpha.ndim < original_samples.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise

    # ------------------------------------------------------------------
    # Model output → x_0 / eps prediction
    # ------------------------------------------------------------------

    def _predict_xstart_from_eps(self, x_t, t, eps):
        device = x_t.device
        coeff1 = self.sqrt_recip_alphas_cumprod.to(device)[t]
        coeff2 = self.sqrt_recipm1_alphas_cumprod.to(device)[t]
        while coeff1.ndim < x_t.ndim:
            coeff1 = coeff1.unsqueeze(-1)
            coeff2 = coeff2.unsqueeze(-1)
        return coeff1 * x_t - coeff2 * eps

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        device = x_t.device
        sqrt_alpha = self.sqrt_alphas_cumprod.to(device)[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(device)[t]
        while sqrt_alpha.ndim < x_t.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return (x_t - sqrt_alpha * pred_xstart) / sqrt_one_minus_alpha

    def _get_xstart_and_eps(self, model_output, x_t, t):
        """Derive both x_0 and eps from model output."""
        if self.learn_sigma:
            C = model_output.shape[1] // 2
            model_output = model_output[:, :C]

        if self.predict_xstart:
            pred_xstart = model_output
            pred_eps = self._predict_eps_from_xstart(x_t, t, pred_xstart)
        else:
            pred_eps = model_output
            pred_xstart = self._predict_xstart_from_eps(x_t, t, pred_eps)
        return pred_xstart, pred_eps

    # ------------------------------------------------------------------
    # DDIM step (decode: latent → data)
    # ------------------------------------------------------------------

    def ddim_step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        timestep_prev: torch.Tensor,
        sample: torch.Tensor,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> DDIBSchedulerOutput:
        """Single DDIM forward (denoising) step: x_t → x_{t-1}."""
        pred_xstart, pred_eps = self._get_xstart_and_eps(model_output, sample, timestep)

        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        device = sample.device
        alpha_bar = self.alphas_cumprod.to(device)[timestep]
        alpha_bar_prev = self.alphas_cumprod.to(device)[timestep_prev]
        while alpha_bar.ndim < sample.ndim:
            alpha_bar = alpha_bar.unsqueeze(-1)
            alpha_bar_prev = alpha_bar_prev.unsqueeze(-1)

        # DDIM deterministic step (eta=0)
        prev_sample = (
            alpha_bar_prev.sqrt() * pred_xstart
            + (1.0 - alpha_bar_prev).sqrt() * pred_eps
        )

        return DDIBSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_xstart,
        )

    # ------------------------------------------------------------------
    # DDIM reverse step (encode: data → latent)
    # ------------------------------------------------------------------

    def ddim_reverse_step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        timestep_next: torch.Tensor,
        sample: torch.Tensor,
        clip_denoised: bool = True,
    ) -> DDIBSchedulerOutput:
        """Single DDIM reverse step: x_t → x_{t+1} (encode direction)."""
        pred_xstart, pred_eps = self._get_xstart_and_eps(model_output, sample, timestep)

        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        device = sample.device
        alpha_bar_next = self.alphas_cumprod.to(device)[timestep_next]
        while alpha_bar_next.ndim < sample.ndim:
            alpha_bar_next = alpha_bar_next.unsqueeze(-1)

        next_sample = (
            alpha_bar_next.sqrt() * pred_xstart
            + (1.0 - alpha_bar_next).sqrt() * pred_eps
        )

        return DDIBSchedulerOutput(
            prev_sample=next_sample,
            pred_original_sample=pred_xstart,
        )

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def compute_training_loss(self, model, x_0, noise=None):
        """Compute simple MSE denoising loss (noise prediction).

        Parameters
        ----------
        model : callable
            ``model(x_t, scaled_t) → eps_pred`` (or ``x_0_pred``).
        x_0 : Tensor (B, C, H, W)
            Clean samples in ``[-1, 1]``.
        noise : Tensor or None
            If ``None`` random noise is sampled.

        Returns
        -------
        loss : Tensor (scalar)
        """
        device = x_0.device
        bsz = x_0.shape[0]
        t = torch.randint(0, self.num_train_timesteps, (bsz,), device=device).long()

        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.add_noise(x_0, noise, t)
        scaled_t = self._scale_timesteps(t)
        model_output = model(x_t, scaled_t)

        if self.learn_sigma:
            C = model_output.shape[1] // 2
            model_output = model_output[:, :C]

        if self.predict_xstart:
            target = x_0
        else:
            target = noise

        loss = ((model_output - target) ** 2).mean()
        return loss

    # ------------------------------------------------------------------
    # Diffusers compatibility
    # ------------------------------------------------------------------

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        return sample
