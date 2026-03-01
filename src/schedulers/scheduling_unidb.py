# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

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

    def _f_m(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Coefficient of x_{t-1} in forward process: m(t)/m(t-1)."""
        m_t = self._m(t)
        t_prev = t - 1 if isinstance(t, int) else t - 1
        m_t_prev = self._m(t_prev)
        return m_t / m_t_prev

    def _f_n(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Coefficient of x_T in forward process: n(t) - n(t-1)*m(t)/m(t-1)."""
        return self._n(t) - self._n(t - 1) * self._f_m(t)

    def _f_sigma_1(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Forward sigma from t-1 to t."""
        f_sig = self._f_sigma(t)
        f_sig_prev = self._f_sigma(t - 1)
        f_m = self._f_m(t)
        return torch.sqrt(
            torch.clamp(f_sig**2 - f_sig_prev**2 * f_m**2, min=1e-20)
        )

    def _r_mean_1(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Reverse mean from t to t-1 (optimum step)."""
        mu = self._mu
        if mu is None:
            mu = torch.zeros_like(xt)
        f_sig = self._f_sigma(t)
        f_sig_prev = self._f_sigma(t - 1)
        f_sig_1 = self._f_sigma_1(t)
        f_m = self._f_m(t)
        f_n = self._f_n(t)
        f_mean_prev = self._f_mean(x0, t - 1)
        if isinstance(f_sig, torch.Tensor) and f_sig.ndim < xt.ndim:
            f_sig = self._append_dims(f_sig, xt.ndim)
            f_sig_prev = self._append_dims(f_sig_prev, xt.ndim)
            f_sig_1 = self._append_dims(f_sig_1, xt.ndim)
            f_m = self._append_dims(f_m, xt.ndim)
            f_n = self._append_dims(f_n, xt.ndim)
            f_mean_prev = f_mean_prev if f_mean_prev.shape == xt.shape else self._append_dims(f_mean_prev, xt.ndim)
        return (
            f_sig_prev**2 * f_m * (xt - f_n * mu) + f_sig_1**2 * f_mean_prev
        ) / (f_sig**2)

    def get_score_from_noise(
        self,
        noise: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Convert model noise output to score: score = -noise / f_sigma(t)."""
        f_sig = self._f_sigma(t)
        if isinstance(f_sig, torch.Tensor) and f_sig.ndim < noise.ndim:
            f_sig = self._append_dims(f_sig, noise.ndim)
        return -noise / f_sig

    def reverse_sde_step_mean(
        self,
        x: torch.Tensor,
        score: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """One reverse step (mean, no dispersion) for training."""
        reverse_drift = self._sde_reverse_drift_1(x, score, t)
        return x - reverse_drift

    def _sde_reverse_drift_1(
        self,
        x: torch.Tensor,
        score: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """SDE reverse drift term (with gamma)."""
        is_scalar = isinstance(t, int)
        if is_scalar:
            t_flat = t
        else:
            t_flat = t.view(-1)

        th_t = self._thetas[t_flat]
        sig_t = self._sigmas[t_flat]
        th_cumsum_t = self._thetas_cumsum[t_flat]
        th_cumsum_T = self._thetas_cumsum[-1]
        sigma_t_T_t = self._sigma_t_T[t_flat]

        if not is_scalar:
            th_t = self._append_dims(th_t, x.ndim)
            sig_t = self._append_dims(sig_t, x.ndim)
            th_cumsum_t = self._append_dims(th_cumsum_t, x.ndim)
            th_cumsum_T = self._append_dims(
                th_cumsum_T.expand(x.shape[0]), x.ndim
            )
            sigma_t_T_t = self._append_dims(sigma_t_T_t, x.ndim)

        base_drift = (th_t * (self._mu - x) - sig_t**2 * score) * self._dt

        tmp = torch.exp(2.0 * (th_cumsum_t - th_cumsum_T) * self._dt)
        drift_h = (
            -(self.gamma * sig_t**2 * tmp)
            / (1.0 + self.gamma * sigma_t_T_t**2)
            * (x - self._mu)
        )
        if is_scalar and t == self.T:
            drift_h = torch.zeros_like(drift_h)
        elif not is_scalar:
            mask = (t_flat == self.T).view(-1, 1, 1, 1).expand_as(drift_h)
            drift_h = torch.where(mask, torch.zeros_like(drift_h), drift_h)

        return base_drift + drift_h

    def reverse_optimum_step(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Optimum x_{t-1} given xt and x0 (for training target)."""
        return self._r_mean_1(xt, x0, t)

    def generate_random_states(
        self,
        x0: torch.Tensor,
        mu: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random (timesteps, noisy_states) for training.

        Returns (timesteps, noisy_states) where timesteps are in 1..T-1.
        """
        self._initialize(device=x0.device if device is None else device)
        self._mu = mu

        x0 = x0.to(self._thetas.device)
        mu = mu.to(self._thetas.device)
        batch = x0.shape[0]

        timesteps = torch.randint(
            1, self.T, (batch, 1, 1, 1), device=x0.device, dtype=torch.long
        )

        state_mean = self._f_mean(x0, timesteps)
        noises = torch.randn_like(state_mean, device=x0.device, dtype=x0.dtype)
        noise_level = self._f_sigma(timesteps)
        if isinstance(noise_level, torch.Tensor) and noise_level.ndim < 4:
            noise_level = self._append_dims(noise_level, 4)
        noisy_states = noises * noise_level + state_mean
        return timesteps, noisy_states.to(torch.float32)

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
