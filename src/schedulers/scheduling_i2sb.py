# Copyright 2024 The I2SB Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# This scheduler implements the Image-to-Image Schrödinger Bridge (I2SB) diffusion
# process compatible with the Hugging Face diffusers library.
# Based on: https://arxiv.org/abs/2302.05872

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class I2SBSchedulerOutput(BaseOutput):
    """
    Output class for the I2SB scheduler's ``step`` function output.

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


def _make_beta_schedule(n_timestep: int, linear_start: float = 1e-4, linear_end: float = 2e-2) -> np.ndarray:
    """I2SB symmetric linear beta schedule.

    ``betas = linspace(sqrt(linear_start), sqrt(linear_end), n_timestep) ** 2``
    then mirrored: ``concat(betas[:half], flip(betas[:half]))``.
    """
    betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    return betas


class I2SBScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for Image-to-Image Schrödinger Bridge (I2SB).

    This scheduler implements the I2SB diffusion process from the paper
    [I2SB: Image-to-Image Schrödinger Bridge](https://arxiv.org/abs/2302.05872). I2SB learns to
    transform between two data distributions using a Schrödinger bridge, enabling high-quality
    image-to-image translation.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        interval (`int`, defaults to `1000`):
            Number of diffusion timesteps.
        beta_max (`float`, defaults to `0.3`):
            Maximum diffusion rate. Used as ``linear_end = beta_max / interval``.
        t0 (`float`, defaults to `1e-4`):
            Start time for the diffusion process.
        T (`float`, defaults to `1.0`):
            End time for the diffusion process.
    """

    _compatibles = []
    order = 1  # DDPM-style first-order sampling

    @register_to_config
    def __init__(
        self,
        interval: int = 1000,
        beta_max: float = 0.3,
        t0: float = 1e-4,
        T: float = 1.0,
    ):
        self.interval = interval

        # Build symmetric beta schedule
        linear_end = beta_max / interval
        betas = _make_beta_schedule(interval, linear_start=1e-4, linear_end=linear_end)
        half = interval // 2
        betas = np.concatenate([betas[:half], betas[:half][::-1]])

        # Cumulative forward / backward standard deviations
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))

        # Gaussian product coefficients (eq 11 / eq 4 in I2SB paper)
        denom = std_fwd ** 2 + std_bwd ** 2
        mu_x0 = std_bwd ** 2 / denom       # weight for x0 (target)
        mu_x1 = std_fwd ** 2 / denom       # weight for x1 (source)
        var = (std_fwd ** 2 * std_bwd ** 2) / denom
        std_sb = np.sqrt(var)

        # Store as numpy arrays; convert to torch when needed
        self.betas_np = betas
        self.std_fwd_np = std_fwd
        self.std_bwd_np = std_bwd
        self.mu_x0_np = mu_x0
        self.mu_x1_np = mu_x1
        self.std_sb_np = std_sb

        # Pre-compute torch versions on CPU
        self.betas = torch.from_numpy(betas).float()
        self.std_fwd = torch.from_numpy(std_fwd).float()
        self.std_bwd = torch.from_numpy(std_bwd).float()
        self.mu_x0 = torch.from_numpy(mu_x0).float()
        self.mu_x1 = torch.from_numpy(mu_x1).float()
        self.std_sb = torch.from_numpy(std_sb).float()

        # Sampling state
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

    def q_sample(
        self,
        step: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        ot_ode: bool = False,
    ) -> torch.Tensor:
        """Forward sampling (eq 11): sample x_t given x0 (target) and x1 (source).

        Parameters
        ----------
        step : Tensor (B,)
            Discrete timestep indices.
        x0 : Tensor (B, C, H, W)
            Target (clean) images.
        x1 : Tensor (B, C, H, W)
            Source (corrupt) images.
        ot_ode : bool
            If True, return the deterministic OT-ODE mean (no noise).
        """
        batch_size = x0.shape[0]
        device = x0.device

        mu_x0_t = self.mu_x0.to(device)[step]                        # (B,)
        mu_x1_t = self.mu_x1.to(device)[step]                        # (B,)
        std_sb_t = self.std_sb.to(device)[step]                       # (B,)

        # Expand to (B, 1, 1, 1) for broadcasting
        mu_x0_t = mu_x0_t.view(batch_size, 1, 1, 1)
        mu_x1_t = mu_x1_t.view(batch_size, 1, 1, 1)
        std_sb_t = std_sb_t.view(batch_size, 1, 1, 1)

        xt = mu_x0_t * x0 + mu_x1_t * x1
        if not ot_ode:
            noise = torch.randn_like(x0)
            xt = xt + std_sb_t * noise
        return xt

    def p_posterior(
        self,
        nprev: torch.Tensor,
        n: torch.Tensor,
        x_n: torch.Tensor,
        x0: torch.Tensor,
        ot_ode: bool = False,
    ) -> torch.Tensor:
        """Backward posterior step (eq 4): sample x_{nprev} given x_n and predicted x0.

        Parameters
        ----------
        nprev : Tensor (B,) or int
            Previous (earlier) timestep indices.
        n : Tensor (B,) or int
            Current timestep indices.
        x_n : Tensor (B, C, H, W)
            Current noisy sample.
        x0 : Tensor (B, C, H, W)
            Predicted clean image.
        ot_ode : bool
            If True, use deterministic step (no noise).
        """
        device = x_n.device
        batch_size = x_n.shape[0]

        # Ensure step indices are tensors
        if not isinstance(n, torch.Tensor):
            n = torch.full((batch_size,), n, device=device, dtype=torch.long)
        if not isinstance(nprev, torch.Tensor):
            nprev = torch.full((batch_size,), nprev, device=device, dtype=torch.long)

        # Flatten to 1-D so indexing always returns a 1-D tensor
        n = n.reshape(-1)
        nprev = nprev.reshape(-1)

        # If scalar (single step shared across batch), expand to batch
        if n.numel() == 1:
            n = n.expand(batch_size)
        if nprev.numel() == 1:
            nprev = nprev.expand(batch_size)

        std_fwd_n = self.std_fwd.to(device)[n].view(batch_size, 1, 1, 1)
        std_fwd_nprev = self.std_fwd.to(device)[nprev].view(batch_size, 1, 1, 1)
        std_delta = (std_fwd_n ** 2 - std_fwd_nprev ** 2).sqrt()

        # Posterior mean: linear interpolation weighted by variances
        mu = (std_fwd_nprev ** 2) / (std_fwd_n ** 2) * x_n + \
             (std_delta ** 2) / (std_fwd_n ** 2) * x0

        if ot_ode:
            return mu

        # Posterior variance
        var = (std_fwd_nprev ** 2 * std_delta ** 2) / (std_fwd_n ** 2)
        noise = torch.randn_like(x_n)
        return mu + var.sqrt() * noise

    def compute_label(
        self,
        step: torch.Tensor,
        x0: torch.Tensor,
        xt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the noise label for training: ``(xt - x0) / std_fwd[step]``.

        Parameters
        ----------
        step : Tensor (B,)
            Discrete timestep indices.
        x0 : Tensor (B, C, H, W)
            Target (clean) images.
        xt : Tensor (B, C, H, W)
            Noisy samples at timestep *step*.
        """
        device = x0.device
        batch_size = x0.shape[0]
        std_fwd_t = self.std_fwd.to(device)[step].view(batch_size, 1, 1, 1)
        label = (xt - x0) / std_fwd_t
        return label

    def compute_pred_x0(
        self,
        step: torch.Tensor,
        xt: torch.Tensor,
        net_out: torch.Tensor,
        clip_denoise: bool = False,
    ) -> torch.Tensor:
        """Recover predicted x0 from the network output: ``xt - std_fwd[step] * net_out``.

        Parameters
        ----------
        step : Tensor (B,) or int
            Discrete timestep indices.
        xt : Tensor (B, C, H, W)
            Noisy samples.
        net_out : Tensor (B, C, H, W)
            Network prediction (noise label).
        clip_denoise : bool
            If True, clamp predicted x0 to [-1, 1].
        """
        device = xt.device
        batch_size = xt.shape[0]
        if not isinstance(step, torch.Tensor):
            step = torch.full((batch_size,), step, device=device, dtype=torch.long)
        std_fwd_t = self.std_fwd.to(device)[step].view(batch_size, 1, 1, 1)
        pred_x0 = xt - std_fwd_t * net_out
        if clip_denoise:
            pred_x0 = pred_x0.clamp(-1, 1)
        return pred_x0

    def set_timesteps(
        self,
        nfe: int,
        device: Union[str, torch.device] = None,
    ):
        """Create evenly spaced step indices from 0 to interval for sampling.

        Parameters
        ----------
        nfe : int
            Number of function evaluations (sampling steps).
        device : str or torch.device, optional
            Device to place tensors on.
        """
        self.num_inference_steps = nfe

        # Evenly spaced indices spanning [0, interval - 1]
        steps = torch.linspace(0, self.interval - 1, nfe + 1, device=device).long()
        self.timesteps = steps

    def add_noise(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        step: torch.Tensor,
        ot_ode: bool = False,
    ) -> torch.Tensor:
        """Add noise to produce x_t from x0 and x1.  Same as :meth:`q_sample`."""
        return self.q_sample(step, x0, x1, ot_ode=ot_ode)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.Tensor,
        ot_ode: bool = False,
        clip_denoise: bool = False,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[I2SBSchedulerOutput, Tuple]:
        """Perform a single backward step using the I2SB posterior.

        Parameters
        ----------
        model_output : Tensor (B, C, H, W)
            Network prediction (noise label).
        timestep : int
            Current timestep index.
        prev_timestep : int
            Previous (earlier) timestep index.
        sample : Tensor (B, C, H, W)
            Current noisy sample x_n.
        ot_ode : bool
            If True, use deterministic step.
        clip_denoise : bool
            If True, clamp predicted x0.
        generator : torch.Generator, optional
            Not used; kept for API compatibility.
        return_dict : bool
            Whether to return an :class:`I2SBSchedulerOutput`.
        """
        pred_x0 = self.compute_pred_x0(timestep, sample, model_output, clip_denoise=clip_denoise)
        prev_sample = self.p_posterior(prev_timestep, timestep, sample, pred_x0, ot_ode=ot_ode)

        if not return_dict:
            return (prev_sample, pred_x0)

        return I2SBSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_x0)

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """Ensures interchangeability with schedulers that need to scale the denoising model input."""
        return sample
