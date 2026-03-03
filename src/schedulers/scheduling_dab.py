# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Brownian-bridge scheduler for the DAB baseline.

Ported from ``dual-app-bridge/model/BrownianBridge/ABridge.py`` into a
diffusers-compatible scheduler API.

Reference:
    Xiao, Bohan, Peiyong Wang, Qisheng He, and Ming Dong.
    "Deterministic Image-to-Image Translation via Denoising Brownian Bridge
    Models with Dual Approximators," 28232–41, 2025.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class DABSchedulerOutput(BaseOutput):
    """Output of one DAB sampling step."""

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


def _extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Gather timestep-indexed scalars and reshape for broadcasting."""
    bsz, *_ = timesteps.shape
    out = values.gather(-1, timesteps)
    return out.reshape(bsz, *((1,) * (len(x_shape) - 1)))


class DABScheduler(SchedulerMixin, ConfigMixin):
    """Dual-Approximator Brownian Bridge scheduler used by DAB.

    The forward process interpolates between target (``x0``) and source (``y``)
    using the ABridge formulation::

        m_t  = t / T
        B_t  = (1 - m_t) * sqrt(m_t / (1 - m_t))
        x_t  = (1 - m_t) * x0 + m_t * y + B_t * noise

    The reverse step updates::

        x_{t-1} = x_t - (1 / t) * objective_recon - (1 / sqrt(T)) * noise
    """

    @register_to_config
    def __init__(
        self,
        num_timesteps: int = 1000,
        objective: str = "grad",
        loss_type: str = "l1",
        skip_sample: bool = False,
        sample_step: int = 1000,
        sample_type: str = "linear",
        eta: float = 1.0,
        max_var: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.objective = objective
        self.loss_type = loss_type
        self.skip_sample = skip_sample
        self.sample_step = sample_step
        self.sample_type = sample_type
        self.eta = eta
        self.max_var = max_var

        self.steps: Optional[torch.Tensor] = None
        self._build_steps()

    def _build_steps(self) -> None:
        T = self.num_timesteps
        if self.skip_sample and self.sample_step < T:
            if self.sample_type == "linear":
                indices = np.linspace(T - 1, 1, self.sample_step).astype(np.int64)
                self.steps = torch.from_numpy(indices)
            elif self.sample_type == "cosine":
                raw = np.linspace(0, T, self.sample_step + 1)
                cos_vals = (np.cos(raw / T * np.pi) + 1.0) / 2.0 * (T - 1)
                self.steps = torch.from_numpy(np.round(cos_vals).astype(np.int64))
            else:
                raise NotImplementedError(f"Unknown sample_type: {self.sample_type}")
        else:
            self.steps = torch.arange(T - 1, 0, -1, dtype=torch.long)

    # ------------------------------------------------------------------
    # Forward process (q-sample)
    # ------------------------------------------------------------------

    def add_noise(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample bridge state ``x_t`` using ABridge formulation.

        ``m_t = t / T``, ``B_t = (1 - m_t) * sqrt(m_t / (1 - m_t))``
        """
        if noise is None:
            noise = torch.randn_like(target)
        T = float(self.num_timesteps)
        t_float = timesteps.float()
        m_t = (t_float / T).view(-1, *([1] * (target.ndim - 1)))
        # Clamp to avoid division by zero
        m_t = m_t.clamp(min=1e-6, max=1.0 - 1e-6)
        B_t = (1.0 - m_t) * torch.sqrt(m_t / (1.0 - m_t))
        return (1.0 - m_t) * target + m_t * source + B_t * noise

    def get_objective(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        x_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return DAB training objective for the selected objective mode."""
        if self.objective == "grad":
            if x_t is None:
                x_t = self.add_noise(target, source, timesteps, noise)
            return x_t - target
        if self.objective == "noise":
            return noise
        if self.objective == "ysubx":
            return source - target
        raise NotImplementedError(f"Unknown objective: {self.objective}")

    # ------------------------------------------------------------------
    # Reconstruction helpers
    # ------------------------------------------------------------------

    def predict_target_from_objective(
        self,
        x_t: torch.Tensor,
        source: torch.Tensor,
        timesteps: torch.Tensor,
        objective_recon: torch.Tensor,
    ) -> torch.Tensor:
        """Recover target endpoint from objective prediction."""
        if self.objective == "grad":
            return x_t - objective_recon
        if self.objective == "noise":
            T = float(self.num_timesteps)
            t_float = timesteps.float()
            m_t = (t_float / T).view(-1, *([1] * (x_t.ndim - 1)))
            m_t = m_t.clamp(min=1e-6, max=1.0 - 1e-6)
            B_t = (1.0 - m_t) * torch.sqrt(m_t / (1.0 - m_t))
            return (x_t - B_t * objective_recon - m_t * source) / (1.0 - m_t)
        if self.objective == "ysubx":
            return source - objective_recon
        raise NotImplementedError(f"Unknown objective: {self.objective}")

    # ------------------------------------------------------------------
    # Reverse sampling step
    # ------------------------------------------------------------------

    def step(
        self,
        model_output: torch.Tensor,
        step_index: int,
        x_t: torch.Tensor,
        source: torch.Tensor,
        clip_denoised: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> DABSchedulerOutput:
        """One reverse bridge step from ``x_t`` towards the target."""
        device = x_t.device
        t_value = int(self.steps[step_index].item())
        t = torch.full((x_t.shape[0],), t_value, device=device, dtype=torch.long)

        target_recon = self.predict_target_from_objective(x_t, source, t, model_output)
        if clip_denoised:
            target_recon = target_recon.clamp(-1.0, 1.0)

        if t_value <= 1:
            return DABSchedulerOutput(prev_sample=target_recon, pred_original_sample=target_recon)

        noise = torch.randn(x_t.shape, device=device, dtype=x_t.dtype, generator=generator)
        T = float(self.num_timesteps)
        t_broad = t.float().view(-1, *([1] * (x_t.ndim - 1)))
        x_prev = x_t - (1.0 / t_broad) * model_output - (1.0 / (T ** 0.5)) * noise
        return DABSchedulerOutput(prev_sample=x_prev, pred_original_sample=target_recon)

    def set_timesteps(self, num_inference_steps: Optional[int] = None) -> None:
        """Rebuild skip schedule, optionally overriding sample step count."""
        if num_inference_steps is not None:
            self.sample_step = num_inference_steps
            self.skip_sample = True
        self._build_steps()
