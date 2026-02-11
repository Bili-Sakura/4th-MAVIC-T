# Copyright 2022 The VDM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# This scheduler implements the Variational Diffusion Model (VDM) sampling
# algorithm compatible with the Hugging Face diffusers library.
# Based on: https://arxiv.org/abs/2107.00630 (Kingma et al., 2021)
# Reference: https://github.com/google-research/vdm

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class VDMSchedulerOutput(BaseOutput):
    """
    Output class for the VDM scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Computed sample `(z_{s})` at the previous logSNR level.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class VDMScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for Variational Diffusion Models (VDM).

    This scheduler implements a logSNR-based noise schedule with DDPM posterior
    sampling. The noise schedule is parameterized by ``gamma(t) = logSNR(t)``
    which linearly interpolates from ``gamma_max`` (low noise) to ``gamma_min``
    (high noise) as ``t`` goes from 0 to 1.

    The forward process is:
    ``z_t = sqrt(sigmoid(gamma_t)) * x_0 + sqrt(sigmoid(-gamma_t)) * eps``

    where ``sigmoid(gamma_t) = alpha_t^2`` and ``sigmoid(-gamma_t) = sigma_t^2``.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`].

    Args:
        gamma_min (`float`, defaults to `-13.3`):
            Minimum logSNR value (at t=1, maximum noise).
        gamma_max (`float`, defaults to `5.0`):
            Maximum logSNR value (at t=0, minimum noise).
        num_train_timesteps (`int`, defaults to `1000`):
            Number of diffusion steps used during training.
        prediction_type (`str`, defaults to `"eps"`):
            Prediction type of the model. One of ``"eps"`` (noise prediction) or
            ``"v"`` (velocity prediction).
        clip_sample (`bool`, defaults to `True`):
            Whether to clip the predicted x_0 to [-1, 1].
    """

    _compatibles = []
    order = 1  # DDPM is a 1st order method

    @register_to_config
    def __init__(
        self,
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        num_train_timesteps: int = 1000,
        prediction_type: str = "eps",
        clip_sample: bool = True,
    ):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        # Initialize state
        self.gammas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

        self.init_noise_sigma = 1.0

    def _gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute logSNR at continuous time t in [0, 1].

        Uses a fixed linear schedule: ``gamma(t) = gamma_max + t * (gamma_min - gamma_max)``.
        """
        return self.gamma_max + t * (self.gamma_min - self.gamma_max)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain.

        Generates evenly spaced steps from t=1 (noisy) to t=0 (clean),
        computing the corresponding logSNR (gamma) values.

        Args:
            num_inference_steps (`int`):
                Number of diffusion steps.
            device (`str` or `torch.device`, *optional*):
                Device to place tensors on.
        """
        self.num_inference_steps = num_inference_steps

        # Evenly spaced steps from 1.0 down to 0.0 (inclusive)
        # steps[i] = t, steps[i+1] = s (previous time, closer to clean)
        steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)

        # Compute gammas (logSNR) at each step
        gammas = self._gamma(steps)

        self.gammas = gammas.to(device)
        self.timesteps = torch.arange(num_inference_steps, device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[VDMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep using DDPM posterior sampling.

        Implements the VDM reverse step (Algorithm 1 of VDM paper):
        ``z_s = sqrt(alpha_s^2 / alpha_t^2) * (z_t * (1-c) + c * alpha_t * x_pred) + sqrt(sigma_s^2 * c) * eps``

        where ``c = -expm1(gamma_t - gamma_s)`` and ``x_pred`` is derived from model output.

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
                Whether to return a `VDMSchedulerOutput` or tuple.
        """
        if self.gammas is None:
            raise ValueError("Gammas not initialized. Call `set_timesteps` first.")

        i = timestep
        gamma_t = self.gammas[i]
        gamma_s = self.gammas[i + 1]

        # Compute alpha and sigma from logSNR
        alpha_t = torch.sqrt(torch.sigmoid(gamma_t))
        alpha_s = torch.sqrt(torch.sigmoid(gamma_s))
        sigma_t = torch.sqrt(torch.sigmoid(-gamma_t))
        sigma_s = torch.sqrt(torch.sigmoid(-gamma_s))

        # c = -expm1(gamma_t - gamma_s) = 1 - exp(gamma_t - gamma_s)
        c = -torch.expm1(gamma_t - gamma_s)

        # Recover x_pred from model output
        if self.prediction_type == "eps":
            x_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.prediction_type == "v":
            x_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if self.clip_sample:
            x_pred = torch.clamp(x_pred, -1.0, 1.0)

        # DDPM posterior (Eq. from VDM paper)
        # mu = alpha_s / alpha_t * (z_t * (1 - c) + c * alpha_t * x_pred)
        # Note: alpha_t * x_pred ≈ z_t without noise component
        mu = (alpha_s / alpha_t) * (sample * (1 - c) + c * alpha_t * x_pred)
        variance = (sigma_s ** 2) * c

        # Sample z_s
        if i < self.num_inference_steps - 1:
            # Not the final step: add noise
            noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = mu + torch.sqrt(variance) * noise
        else:
            # Final step (t_s = 0): deterministic
            prev_sample = mu

        if not return_dict:
            return (prev_sample, x_pred)

        return VDMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=x_pred)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples for the VDM forward diffusion process.

        ``z_t = alpha_t * x_0 + sigma_t * eps``

        where ``alpha_t = sqrt(sigmoid(gamma_t))`` and ``sigma_t = sqrt(sigmoid(-gamma_t))``.

        Args:
            original_samples: Clean samples (x_0).
            noise: Random noise.
            timesteps: logSNR (gamma) values used as timesteps.
        """
        gammas = timesteps.float()
        dims = original_samples.ndim
        gammas = self._append_dims(gammas, dims)

        alpha_t = torch.sqrt(torch.sigmoid(gammas))
        sigma_t = torch.sqrt(torch.sigmoid(-gammas))

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
