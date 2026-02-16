# Copyright 2024 The UniDB Authors (https://github.com/2769433owo/UniDB-plusplus).
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Scheduler for UniDB/UniDB++ compatible with the Hugging Face diffusers library.
# Based on: https://github.com/2769433owo/UniDB-plusplus
#
# UniDB uses a stochastic optimal control formulation with gamma-weighted terminal
# penalty. Supports multiple solvers: euler, noise-solver-1, noise-solver-2,
# data-solver-1, data-solver-2 with sde/mean-ode/pf-ode variants.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import math

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class UniDBSchedulerOutput(BaseOutput):
    """
    Output class for UniDB scheduler step functions.

    Args:
        prev_sample (`torch.Tensor`):
            Computed sample for the next timestep.
        pred_original_sample (`torch.Tensor`, *optional*):
            Predicted denoised sample used in the update.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def _cosine_theta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule for UniDB thetas."""
    timesteps = timesteps + 2
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:-1]
    return betas


def _linear_theta_schedule(timesteps: int) -> torch.Tensor:
    """Linear schedule for UniDB thetas."""
    timesteps = timesteps + 1
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _constant_theta_schedule(timesteps: int, v: float = 1.0) -> torch.Tensor:
    """Constant schedule for UniDB thetas."""
    return torch.ones(timesteps + 1, dtype=torch.float32)


class UniDBScheduler(SchedulerMixin, ConfigMixin):
    """Scheduler for UniDB/UniDB++ diffusion bridge models.

    UniDB uses a stochastic optimal control formulation with gamma-weighted
    terminal penalty. The model predicts noise; score is derived as -noise/f_sigma(t).

    Args:
        lambda_square (`float`, defaults to `1.0`):
            Diffusion coefficient squared. Scaled by 1/255 if >= 1.
        gamma (`float`, defaults to `1e7`):
            Terminal penalty weight. Must match pretrained checkpoint.
        num_train_timesteps (`int`, defaults to `100`):
            Number of diffusion steps (T). UniDB typically uses 100.
        schedule (`str`, defaults to `"cosine"`):
            Theta schedule: `"cosine"`, `"linear"`, or `"constant"`.
        eps (`float`, defaults to `0.01`):
            Small constant for dt computation.
        method (`str`, defaults to `"euler"`):
            Solver method: `"euler"`, `"noise-solver-1"`, `"noise-solver-2"`,
            `"data-solver-1"`, `"data-solver-2"`.
        solver_type (`str`, defaults to `"mean-ode"`):
            For euler: `"sde"`, `"mean-ode"`, or `"pf-ode"`.
            For noise-solver-1: `"sde"` or `"mean-ode"`.
            For noise-solver-2/data-solver-2: `"sde"` or `"mean-ode"`.
            For data-solver-1: `"sde"`, `"mean-ode"`, or `"pf-ode"`.
        solver_step (`int`, defaults to `100`):
            For UniDB++: reduced steps (e.g. 5, 10, 20, 25, 50, 100).
            For Euler: fixed at num_train_timesteps.
    """

    _compatibles = []
    order = 2

    @register_to_config
    def __init__(
        self,
        lambda_square: float = 1.0,
        gamma: float = 1e7,
        num_train_timesteps: int = 100,
        schedule: str = "cosine",
        eps: float = 0.01,
        method: str = "euler",
        solver_type: str = "mean-ode",
        solver_step: int = 100,
    ):
        self.lambda_square = (
            lambda_square / 255.0 if lambda_square >= 1.0 else lambda_square
        )
        self.gamma = gamma
        self.T = num_train_timesteps
        self.dt_val = 1.0 / self.T
        self.schedule = schedule
        self.eps = eps
        self.method = method
        self.solver_type = solver_type
        self.solver_step = min(solver_step, num_train_timesteps)

        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

        # Precomputed tensors (set in _initialize)
        self._thetas: Optional[torch.Tensor] = None
        self._sigmas: Optional[torch.Tensor] = None
        self._thetas_cumsum: Optional[torch.Tensor] = None
        self._sigma_bars: Optional[torch.Tensor] = None
        self._sigma_t_T: Optional[torch.Tensor] = None
        self._f_sigmas: Optional[torch.Tensor] = None
        self._dt: Optional[float] = None

        # Condition (x_T / mu) - set by pipeline before step/add_noise
        self._mu: Optional[torch.Tensor] = None

        self.init_noise_sigma = 1.0

    def _initialize(self, device: Optional[torch.device] = None) -> None:
        """Precompute schedule tensors."""
        if self._thetas is not None:
            return

        T = self.T
        if self.schedule == "cosine":
            thetas = _cosine_theta_schedule(T)
        elif self.schedule == "linear":
            thetas = _linear_theta_schedule(T)
        elif self.schedule == "constant":
            thetas = _constant_theta_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        thetas_cumsum = torch.cumsum(thetas, dim=0) - thetas[0]
        self._dt = -1.0 / thetas_cumsum[-1].item() * math.log(self.eps)

        sigmas = torch.sqrt(self.lambda_square**2 * 2 * thetas)
        sigma_bars = torch.sqrt(
            self.lambda_square**2
            * (1.0 - torch.exp(-2.0 * thetas_cumsum * self._dt))
        )
        sigma_t_T = torch.sqrt(
            self.lambda_square**2
            * (
                1.0
                - torch.exp(
                    -2.0 * (thetas_cumsum[-1] - thetas_cumsum) * self._dt
                )
            )
        )
        f_sigmas = sigma_bars * sigma_t_T / sigma_bars[-1]

        self._thetas = thetas
        self._sigmas = sigmas
        self._thetas_cumsum = thetas_cumsum
        self._sigma_bars = sigma_bars
        self._sigma_t_T = sigma_t_T
        self._f_sigmas = f_sigmas

        if device is not None:
            self._thetas = self._thetas.to(device)
            self._sigmas = self._sigmas.to(device)
            self._thetas_cumsum = self._thetas_cumsum.to(device)
            self._sigma_bars = self._sigma_bars.to(device)
            self._sigma_t_T = self._sigma_t_T.to(device)
            self._f_sigmas = self._f_sigmas.to(device)

    @staticmethod
    def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}"
            )
        return x[(...,) + (None,) * dims_to_append]

    def _m(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Coefficient of x0 in marginal forward process."""
        if isinstance(t, int):
            t = torch.tensor(t, device=self._thetas.device, dtype=torch.long)
        return (
            torch.exp(-self._thetas_cumsum[t] * self._dt)
            * (1.0 + self.gamma * self._sigma_t_T[t] ** 2)
            / (1.0 + self.gamma * self._sigma_bars[-1] ** 2)
        )

    def _n(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Coefficient of xT in marginal forward process."""
        return 1.0 - self._m(t)

    def _f_sigma(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Marginal forward sigma at timestep t."""
        if isinstance(t, int):
            t = torch.tensor(t, device=self._thetas.device, dtype=torch.long)
        return self._f_sigmas[t]

    def _f_mean(self, x0: torch.Tensor, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Forward mean with t."""
        mu = self._mu
        if mu is None:
            mu = torch.zeros_like(x0)
        return self._m(t) * x0 + self._n(t) * mu

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """Set sampling timesteps."""
        self.num_inference_steps = num_inference_steps
        self._initialize(device=device)

        if self.method == "euler":
            steps = self.T
        else:
            steps = self.solver_step

        self.timesteps = torch.arange(steps, 0, -1, device=device, dtype=torch.long)
        self.sigmas = self._f_sigmas.to(device)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        x_T: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add UniDB forward noise: x_t = m(t)*x0 + n(t)*mu + f_sigma(t)*noise."""
        self._initialize(device=original_samples.device)
        mu = x_T if x_T is not None else torch.zeros_like(original_samples)
        self._mu = mu

        t = timesteps.to(
            device=original_samples.device,
            dtype=torch.long,
        )
        m_t = self._append_dims(self._m(t), original_samples.ndim)
        n_t = self._append_dims(self._n(t), original_samples.ndim)
        f_sigma_t = self._append_dims(self._f_sigma(t), original_samples.ndim)

        if mu.ndim < original_samples.ndim:
            mu = self._append_dims(mu, original_samples.ndim)

        return m_t * original_samples + n_t * mu + f_sigma_t * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        x_T: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[UniDBSchedulerOutput, Tuple]:
        """Single UniDB reverse step (Euler methods only).

        For noise-solver-* and data-solver-* methods, use the pipeline's
        full sampling loop which implements the multi-step solvers.
        """
        self._initialize(device=sample.device)
        self._mu = x_T

        # timestep is index into timesteps (0..num_steps-1); UniDB uses t=1..T
        # Our timesteps go T, T-1, ..., 1 so timestep 0 -> t=T, timestep 1 -> t=T-1, etc.
        if self.timesteps is not None and timestep < len(self.timesteps):
            t = int(self.timesteps[timestep].item())
        else:
            t = self.T - timestep
        if t < 1:
            t = 1
        if t > self.T:
            t = self.T

        score = -model_output / self._f_sigma(t)
        if t == self.T:
            score = torch.zeros_like(model_output)

        # sde_reverse_drift_1 with gamma
        tmp = torch.exp(
            2.0
            * (self._thetas_cumsum[t] - self._thetas_cumsum[-1])
            * self._dt
        )
        drift_h = (
            -(self.gamma * self._sigmas[t] ** 2 * tmp)
            / (1.0 + self.gamma * self._sigma_t_T[t] ** 2)
            * (sample - self._mu)
        )
        if t == self.T:
            drift_h = torch.zeros_like(drift_h)

        reverse_drift = (
            self._thetas[t] * (self._mu - sample) + drift_h - self._sigmas[t] ** 2 * score
        ) * self._dt

        if self.solver_type == "sde":
            disp = self._sigmas[t] * (
                randn_tensor(
                    sample.shape,
                    generator=generator,
                    device=sample.device,
                    dtype=sample.dtype,
                )
                * math.sqrt(self._dt)
            )
            prev_sample = sample - reverse_drift - disp
        elif self.solver_type == "mean-ode":
            prev_sample = sample - reverse_drift
        elif self.solver_type == "pf-ode":
            reverse_drift_pf = (
                self._thetas[t] * (self._mu - sample)
                + drift_h
                - 0.5 * self._sigmas[t] ** 2 * score
            ) * self._dt
            prev_sample = sample - reverse_drift_pf
        else:
            raise ValueError(f"Unknown solver_type: {self.solver_type}")

        pred_x0 = self._predict_x0(sample, t, model_output)
        if not return_dict:
            return (prev_sample, pred_x0)
        return UniDBSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_x0,
        )

    def _predict_x0(
        self,
        xt: torch.Tensor,
        t: Union[int, torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x0 from xt and noise: (xt - n(t)*mu - f_sigma(t)*noise) / m(t)."""
        if t == self.T:
            return self._mu
        m_t = self._m(t)
        n_t = self._n(t)
        f_sigma_t = self._f_sigma(t)
        if isinstance(m_t, torch.Tensor):
            m_t = self._append_dims(m_t, xt.ndim)
            n_t = self._append_dims(n_t, xt.ndim)
            f_sigma_t = self._append_dims(f_sigma_t, xt.ndim)
        return (xt - n_t * self._mu - f_sigma_t * noise) / m_t

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """Compatibility hook."""
        return sample
