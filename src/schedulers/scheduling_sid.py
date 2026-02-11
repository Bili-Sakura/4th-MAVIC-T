# Copyright 2023 The SimpleDiffusion Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# This scheduler implements the Simple Diffusion (SiD) sampling algorithm
# compatible with the Hugging Face diffusers library.
# Based on: https://arxiv.org/abs/2301.11093 (Hoogeboom et al., 2023)
# Reference: https://github.com/faverogian/simpleDiffusion

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import numpy as np
import torch
from torch.special import expm1

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def _log(t, eps=1e-20):
    """Safe log to avoid log(0)."""
    return torch.log(t.clamp(min=eps))


@dataclass
class SiDSchedulerOutput(BaseOutput):
    """
    Output class for the SiD scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Computed sample `(z_{s})` at the previous logSNR level.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class SiDScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for Simple Diffusion (SiD).

    This scheduler implements the shifted cosine logSNR schedule from the
    "Simple Diffusion" paper with standard DDPM posterior sampling. The shifted
    schedule adjusts the noise levels based on the image resolution to ensure
    that high-resolution images receive appropriate noise.

    The shifted cosine schedule is:
    ``logSNR(t) = -2 * log(tan(t_min + t * (t_max - t_min))) + 2 * log(noise_d / image_d)``

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`].

    Args:
        logsnr_min (`float`, defaults to `-15.0`):
            Minimum logSNR value for the base cosine schedule.
        logsnr_max (`float`, defaults to `15.0`):
            Maximum logSNR value for the base cosine schedule.
        noise_d (`float`, defaults to `64.0`):
            Base noise dimension for the shifted cosine schedule.
        image_d (`float`, defaults to `64.0`):
            Image dimension (resolution) for schedule shifting. When ``noise_d == image_d``,
            the schedule reduces to the standard cosine schedule.
        num_train_timesteps (`int`, defaults to `1000`):
            Number of diffusion steps used during training.
        prediction_type (`str`, defaults to `"eps"`):
            Prediction type. One of ``"eps"`` (noise) or ``"v"`` (velocity).
        clip_sample (`bool`, defaults to `True`):
            Whether to clip the predicted x_0 to [-1, 1].
    """

    _compatibles = []
    order = 1  # DDPM is a 1st order method

    @register_to_config
    def __init__(
        self,
        logsnr_min: float = -15.0,
        logsnr_max: float = 15.0,
        noise_d: float = 64.0,
        image_d: float = 64.0,
        num_train_timesteps: int = 1000,
        prediction_type: str = "eps",
        clip_sample: bool = True,
    ):
        self.logsnr_min = logsnr_min
        self.logsnr_max = logsnr_max
        if noise_d <= 0 or image_d <= 0:
            raise ValueError(f"noise_d ({noise_d}) and image_d ({image_d}) must be positive")
        self.noise_d = noise_d
        self.image_d = image_d
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        # Initialize state
        self.logsnrs: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

        self.init_noise_sigma = 1.0

    def _logsnr_cosine(self, t: torch.Tensor) -> torch.Tensor:
        """Compute base cosine logSNR schedule.

        ``logSNR(t) = -2 * log(tan(t_min + t * (t_max - t_min)))``

        where t_min and t_max are chosen to map logsnr_max and logsnr_min.
        """
        t_min = math.atan(math.exp(-0.5 * self.logsnr_max))
        t_max = math.atan(math.exp(-0.5 * self.logsnr_min))
        return -2.0 * _log(torch.tan(t_min + t * (t_max - t_min)))

    def _logsnr_shifted_cosine(self, t: torch.Tensor) -> torch.Tensor:
        """Compute shifted cosine logSNR schedule.

        ``logSNR_shifted(t) = logSNR_cosine(t) + 2 * log(noise_d / image_d)``
        """
        logsnr = self._logsnr_cosine(t)
        shift = 2.0 * math.log(self.noise_d / self.image_d)
        return logsnr + shift

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain.

        Generates evenly spaced steps from t=1 (noisy) to t=0 (clean),
        computing the corresponding shifted cosine logSNR values.

        Args:
            num_inference_steps (`int`):
                Number of diffusion steps.
            device (`str` or `torch.device`, *optional*):
                Device to place tensors on.
        """
        self.num_inference_steps = num_inference_steps

        # Evenly spaced steps from 1.0 down to 0.0
        steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)

        # Compute logSNR at each step using the shifted cosine schedule
        logsnrs = self._logsnr_shifted_cosine(steps)

        self.logsnrs = logsnrs.to(device)
        self.timesteps = torch.arange(num_inference_steps, device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SiDSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep using DDPM posterior sampling.

        Implements the DDPM reverse step from the Simple Diffusion framework:
        ``z_s = mu + sqrt(variance) * noise``

        where ``mu = alpha_s * (z_t * (1-c) / alpha_t + c * x_pred)``
        and ``variance = sigma_s^2 * c``, with ``c = -expm1(logsnr_t - logsnr_s)``.

        Args:
            model_output (`torch.Tensor`):
                Direct output from the model (noise or velocity prediction).
            timestep (`int`):
                Current discrete timestep index.
            sample (`torch.Tensor`):
                Current noisy sample z_t.
            generator (`torch.Generator`, *optional*):
                Random number generator.
            return_dict (`bool`, defaults to `True`):
                Whether to return a `SiDSchedulerOutput` or tuple.
        """
        if self.logsnrs is None:
            raise ValueError("LogSNRs not initialized. Call `set_timesteps` first.")

        i = timestep
        logsnr_t = self.logsnrs[i]
        logsnr_s = self.logsnrs[i + 1]

        # Compute alpha and sigma from logSNR
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        # c = -expm1(logsnr_t - logsnr_s) = 1 - exp(logsnr_t - logsnr_s)
        c = -torch.expm1(logsnr_t - logsnr_s)

        # Recover x_pred from model output
        if self.prediction_type == "eps":
            x_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.prediction_type == "v":
            x_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if self.clip_sample:
            x_pred = torch.clamp(x_pred, -1.0, 1.0)

        # DDPM posterior mean and variance
        mu = alpha_s * (sample * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s ** 2) * c

        # Sample z_s
        if i < self.num_inference_steps - 1:
            # Not the final step: add noise
            noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = mu + torch.sqrt(variance) * noise
        else:
            # Final step: deterministic
            prev_sample = mu

        if not return_dict:
            return (prev_sample, x_pred)

        return SiDSchedulerOutput(prev_sample=prev_sample, pred_original_sample=x_pred)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples for the SiD forward diffusion process.

        ``z_t = alpha_t * x_0 + sigma_t * eps``

        where ``alpha_t = sqrt(sigmoid(logsnr_t))`` and ``sigma_t = sqrt(sigmoid(-logsnr_t))``.

        Args:
            original_samples: Clean samples (x_0).
            noise: Random noise.
            timesteps: logSNR values used as timesteps.
        """
        logsnrs = timesteps.float()
        dims = original_samples.ndim
        logsnrs = self._append_dims(logsnrs, dims)

        alpha_t = torch.sqrt(torch.sigmoid(logsnrs))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnrs))

        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def _append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input.
        """
        return sample
