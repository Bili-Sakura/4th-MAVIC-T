# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2025 Chadebec et al. and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# This scheduler implements the Latent Bridge Matching (LBM) diffusion process
# compatible with the Hugging Face diffusers library.
# Based on: https://arxiv.org/abs/2503.07535

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class LBMSchedulerOutput(BaseOutput):
    """
    Output class for the LBM scheduler's ``step`` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class LBMScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for Latent Bridge Matching (LBM).

    This scheduler implements the LBM flow-matching bridge process from the paper
    `LBM: Latent Bridge Matching for Fast Image-to-Image Translation
    <https://arxiv.org/abs/2503.07535>`_.  LBM uses flow-matching to interpolate
    between source and target distributions in latent space, enabling single-step
    or few-step high-quality image-to-image translation.

    The forward process creates interpolants:

        ``x_t = sigma * x_source + (1 - sigma) * x_target + bridge_noise * sqrt(sigma * (1 - sigma)) * eps``

    The model is trained to predict ``x_source - x_target`` (the flow direction),
    and sampling uses Euler steps with optional bridge noise injection.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to `1000`):
            Number of training timesteps.
        bridge_noise_sigma (`float`, defaults to `0.001`):
            Bridge noise magnitude used during the interpolant construction and sampling.
        timestep_sampling (`str`, defaults to ``"uniform"``):
            How to sample timesteps during training.  One of ``"uniform"``, ``"log_normal"``, or ``"custom_timesteps"``.
        logit_mean (`float`, defaults to `0.0`):
            Mean for ``log_normal`` timestep sampling.
        logit_std (`float`, defaults to `1.0`):
            Standard deviation for ``log_normal`` timestep sampling.
        selected_timesteps (`list` of `float`, optional):
            Timesteps used when ``timestep_sampling="custom_timesteps"``.
        prob (`list` of `float`, optional):
            Probabilities for ``selected_timesteps``.
    """

    _compatibles = []
    order = 1  # Euler first-order sampling

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        bridge_noise_sigma: float = 0.001,
        timestep_sampling: str = "uniform",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        selected_timesteps: Optional[List[float]] = None,
        prob: Optional[List[float]] = None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.bridge_noise_sigma = bridge_noise_sigma
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.selected_timesteps = selected_timesteps
        self.prob = prob

        # Build evenly-spaced sigma schedule for training (1.0 → 0.0 in
        # *num_train_timesteps* steps).  Sigma = 1 means pure source;
        # sigma = 0 means pure target.
        sigmas = np.linspace(1.0, 0.0, num_train_timesteps + 1, dtype=np.float64)
        timesteps = np.arange(0, num_train_timesteps, dtype=np.int64)

        self.sigmas = torch.from_numpy(sigmas).float()
        self.timesteps = torch.from_numpy(timesteps).long()

        # Sampling state (set via :meth:`set_timesteps`)
        self.num_inference_steps: Optional[int] = None

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def sample_timesteps(
        self,
        n_samples: int,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Sample training timesteps according to the configured strategy.

        Returns
        -------
        timesteps : Tensor (n_samples,)
            Sampled timesteps mapped to the internal schedule.
        """
        if self.timestep_sampling == "uniform":
            idx = torch.randint(0, self.num_train_timesteps, (n_samples,), device="cpu")
            return self.timesteps[idx].to(device=device)

        elif self.timestep_sampling == "log_normal":
            u = torch.normal(
                mean=self.logit_mean,
                std=self.logit_std,
                size=(n_samples,),
                device="cpu",
            )
            u = torch.sigmoid(u)
            indices = (u * self.num_train_timesteps).long().clamp(0, self.num_train_timesteps - 1)
            return self.timesteps[indices].to(device=device)

        elif self.timestep_sampling == "custom_timesteps":
            assert self.selected_timesteps is not None and self.prob is not None, (
                "selected_timesteps and prob must be set for custom_timesteps sampling"
            )
            idx = np.random.choice(len(self.selected_timesteps), n_samples, p=self.prob)
            return torch.tensor(
                [self.selected_timesteps[i] for i in idx],
                device=device,
                dtype=torch.long,
            )

        else:
            raise ValueError(f"Unknown timestep_sampling: {self.timestep_sampling}")

    def get_sigmas(
        self,
        timesteps: torch.Tensor,
        n_dim: int = 4,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Look up sigma values for given timesteps.

        Parameters
        ----------
        timesteps : Tensor (B,)
            Timestep indices.
        n_dim : int
            Number of output dimensions (extra trailing ones are added).
        device, dtype : torch.device, torch.dtype
            Device and dtype for the output tensor.

        Returns
        -------
        sigmas : Tensor (B, 1, …, 1)
            Sigma values expanded to *n_dim* dimensions.
        """
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def add_noise(
        self,
        x_target: torch.Tensor,
        x_source: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Create an interpolant between source and target (forward process).

        ``x_t = sigma * x_source + (1 - sigma) * x_target + bridge_noise * sqrt(sigma * (1 - sigma)) * eps``

        Parameters
        ----------
        x_target : Tensor (B, C, H, W)
            Target (clean) images / latents.
        x_source : Tensor (B, C, H, W)
            Source (condition) images / latents.
        timesteps : Tensor (B,)
            Timestep indices.

        Returns
        -------
        noisy_sample : Tensor (B, C, H, W)
        """
        sigmas = self.get_sigmas(timesteps, n_dim=x_target.ndim, device=x_target.device, dtype=x_target.dtype)
        noisy_sample = (
            sigmas * x_source
            + (1.0 - sigmas) * x_target
            + self.bridge_noise_sigma * (sigmas * (1.0 - sigmas)) ** 0.5 * torch.randn_like(x_target)
        )
        return noisy_sample

    def compute_target(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the training target for the denoiser: ``x_source - x_target``.

        Parameters
        ----------
        x_source, x_target : Tensor (B, C, H, W)

        Returns
        -------
        target : Tensor (B, C, H, W)
        """
        return x_source - x_target

    def compute_pred_x0(
        self,
        noisy_sample: torch.Tensor,
        model_output: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the predicted target (x_0) from the model output.

        ``pred_x0 = noisy_sample - model_output * sigma``

        Parameters
        ----------
        noisy_sample : Tensor (B, C, H, W)
            Current noisy sample.
        model_output : Tensor (B, C, H, W)
            Network prediction (flow direction).
        sigmas : Tensor (B, 1, 1, 1)
            Sigma values.

        Returns
        -------
        pred_x0 : Tensor (B, C, H, W)
        """
        return noisy_sample - model_output * sigmas

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """Prepare the scheduler for inference with ``num_inference_steps`` Euler steps.

        Parameters
        ----------
        num_inference_steps : int
            Number of sampling steps.
        device : str or torch.device, optional
            Device to place tensors on.
        """
        self.num_inference_steps = num_inference_steps

        # Linearly-spaced sigma schedule from 1.0 to 1/num_inference_steps
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        self.sigmas = torch.from_numpy(sigmas).float()
        if device is not None:
            self.sigmas = self.sigmas.to(device)

        # Map sigma values back to closest training timesteps
        self.timesteps = torch.linspace(
            0, self.num_train_timesteps - 1, num_inference_steps, device=device
        ).long()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[LBMSchedulerOutput, Tuple]:
        """Perform a single Euler step on the reverse (sampling) process.

        This delegates to the FlowMatchEulerDiscreteScheduler-style update:
        ``sample = sample - sigma * model_output``.

        Parameters
        ----------
        model_output : Tensor (B, C, H, W)
            Network prediction (flow direction).
        timestep : int or Tensor
            Current timestep (used to look up sigma).
        sample : Tensor (B, C, H, W)
            Current noisy sample.
        generator : torch.Generator, optional
            Not used; kept for API compatibility.
        return_dict : bool
            Whether to return an :class:`LBMSchedulerOutput`.
        """
        device = sample.device
        dtype = sample.dtype

        # Get sigma for current timestep
        if isinstance(timestep, torch.Tensor):
            t = timestep.to(device)
        else:
            t = torch.tensor([timestep], device=device)

        if t.ndim == 0:
            t = t.unsqueeze(0)

        sigmas = self.get_sigmas(t, n_dim=sample.ndim, device=device, dtype=dtype)

        # Euler update: x_{t-1} = x_t - sigma * model_output
        pred_x0 = self.compute_pred_x0(sample, model_output, sigmas)
        prev_sample = pred_x0  # In Euler flow-matching, one step directly gives x_0

        if not return_dict:
            return (prev_sample, pred_x0)

        return LBMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """Ensures interchangeability with schedulers that need to scale the denoising model input."""
        return sample
