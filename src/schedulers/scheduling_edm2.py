# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2024 The EDM2 Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# This scheduler implements the EDM2 sampling algorithm compatible with the
# Hugging Face diffusers library.
# Based on: https://arxiv.org/abs/2312.02696 (Karras et al., 2024)
# Reference: https://github.com/NVlabs/edm2

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class EDM2SchedulerOutput(BaseOutput):
    """
    Output class for the EDM2 scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Computed sample `(x_{t-1})` of previous timestep.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class EDM2Scheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for EDM2 (Elucidating the Design Space of Diffusion Models v2).

    This scheduler implements the Karras sigma schedule and Heun (2nd order)
    ODE sampler from the EDM2 paper. It supports optional stochastic churn
    for improved sample quality.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        sigma_min (`float`, defaults to `0.002`):
            Minimum sigma value for the noise schedule.
        sigma_max (`float`, defaults to `80.0`):
            Maximum sigma value for the noise schedule.
        sigma_data (`float`, defaults to `0.5`):
            Standard deviation of the data distribution, used for preconditioning.
        rho (`float`, defaults to `7.0`):
            Rho parameter for Karras noise schedule discretization.
        num_train_timesteps (`int`, defaults to `32`):
            Default number of diffusion steps used during sampling.
        s_churn (`float`, defaults to `0.0`):
            Stochastic churn parameter. 0 = deterministic ODE sampling.
        s_min (`float`, defaults to `0.0`):
            Minimum sigma for stochastic churn.
        s_max (`float`, defaults to `inf`):
            Maximum sigma for stochastic churn.
        s_noise (`float`, defaults to `1.0`):
            Stochastic noise scaling factor.
    """

    _compatibles = []
    order = 2  # Heun is a 2nd order method

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        num_train_timesteps: int = 32,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = float("inf"),
        s_noise: float = 1.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.num_train_timesteps = num_train_timesteps
        self.s_churn = s_churn
        self.s_min = s_min
        self.s_max = s_max
        self.s_noise = s_noise

        # Initialize state
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

        # Standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps (sigma schedule) used for the diffusion chain.

        Implements the Karras sigma schedule:
        ``sigma_i = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho``

        Args:
            num_inference_steps (`int`):
                Number of diffusion steps.
            device (`str` or `torch.device`, *optional*):
                Device to place tensors on.
        """
        self.num_inference_steps = num_inference_steps

        # Karras sigma schedule (Eq. 5 of EDM paper)
        step_indices = torch.arange(num_inference_steps, dtype=torch.float64)
        sigmas = (
            self.sigma_max ** (1 / self.rho)
            + step_indices / (num_inference_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        sigmas = torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)])  # t_N = 0

        self.sigmas = sigmas.to(device=device, dtype=torch.float32)
        self.timesteps = torch.arange(num_inference_steps, device=device)

    def _precondition_coefficients(self, sigma: torch.Tensor):
        """Compute EDM2 preconditioning coefficients.

        Args:
            sigma: Noise level tensor.

        Returns:
            Tuple of (c_skip, c_out, c_in, c_noise).
        """
        sd2 = self.sigma_data ** 2
        c_skip = sd2 / (sigma ** 2 + sd2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + sd2).sqrt()
        c_in = 1.0 / (sd2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4.0
        return c_skip, c_out, c_in, c_noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EDM2SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the ODE.

        This implements a single Euler step. For full Heun (2nd order) sampling,
        use ``step_heun`` which takes two model evaluations.

        The model output is interpreted as the denoised prediction ``D(x; sigma)``.
        The ODE derivative is: ``d = (x - D(x, sigma)) / sigma``.

        Args:
            model_output (`torch.Tensor`):
                Direct output from the learned diffusion model (denoised prediction D(x; sigma)).
            timestep (`int`):
                Current discrete timestep index in the diffusion chain.
            sample (`torch.Tensor`):
                Current instance of sample being created by diffusion process.
            generator (`torch.Generator`, *optional*):
                Random number generator for stochastic churn.
            return_dict (`bool`, defaults to `True`):
                Whether to return a `EDM2SchedulerOutput` or tuple.

        Returns:
            [`EDM2SchedulerOutput`] or `tuple`:
                The predicted sample and optionally the predicted denoised sample.
        """
        if self.sigmas is None:
            raise ValueError("Sigmas not initialized. Call `set_timesteps` first.")

        i = timestep
        t_cur = self.sigmas[i]
        t_next = self.sigmas[i + 1]
        denoised = model_output

        x_cur = sample

        # Optional stochastic churn
        if self.s_churn > 0 and self.s_min <= t_cur <= self.s_max:
            gamma = min(self.s_churn / self.num_inference_steps, (2 ** 0.5) - 1)
            t_hat = t_cur + gamma * t_cur
            noise = randn_tensor(x_cur.shape, generator=generator, device=x_cur.device, dtype=x_cur.dtype)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.s_noise * noise
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step: d = (x - D(x, sigma)) / sigma
        d_cur = (x_hat - denoised) / t_hat
        prev_sample = x_hat + (t_next - t_hat) * d_cur

        if not return_dict:
            return (prev_sample, denoised)

        return EDM2SchedulerOutput(prev_sample=prev_sample, pred_original_sample=denoised)

    def step_heun(
        self,
        denoised_1: torch.Tensor,
        denoised_2: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EDM2SchedulerOutput, Tuple]:
        """
        Perform a full Heun step with two model evaluations.

        Args:
            denoised_1: Denoised prediction D(x, t_cur) at current sigma.
            denoised_2: Denoised prediction D(x', t_next) at next sigma.
            timestep: Current timestep index.
            sample: Current sample.
            generator: Random number generator.
            return_dict: Whether to return a dict.
        """
        if self.sigmas is None:
            raise ValueError("Sigmas not initialized. Call `set_timesteps` first.")

        i = timestep
        t_cur = self.sigmas[i]
        t_next = self.sigmas[i + 1]

        x_cur = sample

        # Optional stochastic churn
        if self.s_churn > 0 and self.s_min <= t_cur <= self.s_max:
            gamma = min(self.s_churn / self.num_inference_steps, (2 ** 0.5) - 1)
            t_hat = t_cur + gamma * t_cur
            noise = randn_tensor(x_cur.shape, generator=generator, device=x_cur.device, dtype=x_cur.dtype)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.s_noise * noise
        else:
            t_hat = t_cur
            x_hat = x_cur

        # First derivative (Euler)
        d_cur = (x_hat - denoised_1) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if t_next == 0:
            # Final step - no 2nd order correction
            prev_sample = x_next
        else:
            # Second derivative
            d_prime = (x_next - denoised_2) / t_next
            # Heun average
            d_avg = 0.5 * d_cur + 0.5 * d_prime
            prev_sample = x_hat + (t_next - t_hat) * d_avg

        if not return_dict:
            return (prev_sample, denoised_1)

        return EDM2SchedulerOutput(prev_sample=prev_sample, pred_original_sample=denoised_1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples for the EDM2 diffusion process.

        In EDM2, noisy samples are: ``x_noisy = x_0 + sigma * noise``.

        Args:
            original_samples: Clean samples (x_0).
            noise: Random noise.
            timesteps: Sigma values as timesteps.
        """
        sigmas = timesteps.float()
        dims = original_samples.ndim
        sigmas = self._append_dims(sigmas, dims)
        noisy_samples = original_samples + sigmas * noise
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
