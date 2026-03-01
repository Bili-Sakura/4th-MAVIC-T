# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Scheduler for Simpler Diffusion (SiD2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import math
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


def _safe_log(tensor: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """Numerically stable log."""
    return torch.log(tensor.clamp(min=eps))


@dataclass
class SiD2SchedulerOutput(BaseOutput):
    """Output container for one SiD2 scheduler step."""

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class SiD2Scheduler(SchedulerMixin, ConfigMixin):
    """
    DDPM-style scheduler used by the standalone SiD2 baseline.

    The implementation follows the SiD family with support for:
    - cosine logSNR schedule
    - shifted cosine logSNR schedule
    - cosine-interpolated schedule (paper appendix setting)
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        logsnr_min: float = -15.0,
        logsnr_max: float = 15.0,
        schedule_type: str = "cosine_interpolated",
        noise_d: float = 64.0,
        image_d: float = 64.0,
        interpolated_noise_d_low: Optional[float] = None,
        interpolated_noise_d_high: Optional[float] = None,
        num_train_timesteps: int = 1000,
        prediction_type: str = "v",
        clip_sample: bool = True,
    ) -> None:
        if noise_d <= 0 or image_d <= 0:
            raise ValueError(f"noise_d ({noise_d}) and image_d ({image_d}) must be positive")
        if schedule_type not in {"cosine", "shifted_cosine", "cosine_interpolated"}:
            raise ValueError(
                "schedule_type must be one of {'cosine', 'shifted_cosine', 'cosine_interpolated'}"
            )
        if prediction_type not in {"eps", "v"}:
            raise ValueError("prediction_type must be 'eps' or 'v'")

        if interpolated_noise_d_low is None:
            interpolated_noise_d_low = max(1.0, float(image_d) / 16.0)
        if interpolated_noise_d_high is None:
            interpolated_noise_d_high = float(image_d)
        if interpolated_noise_d_low <= 0 or interpolated_noise_d_high <= 0:
            raise ValueError("interpolated_noise_d_low/high must be positive")

        self.logsnr_min = float(logsnr_min)
        self.logsnr_max = float(logsnr_max)
        self.schedule_type = str(schedule_type)
        self.noise_d = float(noise_d)
        self.image_d = float(image_d)
        self.interpolated_noise_d_low = float(interpolated_noise_d_low)
        self.interpolated_noise_d_high = float(interpolated_noise_d_high)
        self.num_train_timesteps = int(num_train_timesteps)
        self.prediction_type = str(prediction_type)
        self.clip_sample = bool(clip_sample)

        self.logsnrs: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None
        self.init_noise_sigma = 1.0

    def _append_dims(self, x: torch.Tensor, target_dims: int) -> torch.Tensor:
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    def _logsnr_cosine(self, t: torch.Tensor) -> torch.Tensor:
        t_min = math.atan(math.exp(-0.5 * self.logsnr_max))
        t_max = math.atan(math.exp(-0.5 * self.logsnr_min))
        u = t_min + t * (t_max - t_min)
        return -2.0 * _safe_log(torch.tan(u))

    def _dlogsnr_cosine_dt(self, t: torch.Tensor) -> torch.Tensor:
        t_min = math.atan(math.exp(-0.5 * self.logsnr_max))
        t_max = math.atan(math.exp(-0.5 * self.logsnr_min))
        u = t_min + t * (t_max - t_min)
        sin_u = torch.sin(u).clamp(min=1e-20)
        cos_u = torch.cos(u).clamp(min=1e-20)
        return -2.0 * (t_max - t_min) / (sin_u * cos_u)

    def _schedule_shift(self, noise_dim: float) -> float:
        return 2.0 * math.log(float(noise_dim) / self.image_d)

    def compute_logsnr_and_derivative(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ``(lambda_t, d lambda_t / dt)`` for continuous t in [0, 1].
        """
        t = t.float().clamp(1e-5, 1.0 - 1e-5)
        base = self._logsnr_cosine(t)
        d_base = self._dlogsnr_cosine_dt(t)

        if self.schedule_type == "cosine":
            return base, d_base

        if self.schedule_type == "shifted_cosine":
            return base + self._schedule_shift(self.noise_d), d_base

        # cosine_interpolated:
        # lambda(t) = t * lambda_high(t) + (1 - t) * lambda_low(t)
        # with lambda_high/low being shifted cosine schedules.
        shift_low = self._schedule_shift(self.interpolated_noise_d_low)
        shift_high = self._schedule_shift(self.interpolated_noise_d_high)
        logsnr = base + shift_low + t * (shift_high - shift_low)
        dlogsnr_dt = d_base + (shift_high - shift_low)
        return logsnr, dlogsnr_dt

    def compute_logsnr(self, t: torch.Tensor) -> torch.Tensor:
        """Compute schedule value ``lambda_t`` for continuous t."""
        logsnr, _ = self.compute_logsnr_and_derivative(t)
        return logsnr

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        self.num_inference_steps = int(num_inference_steps)
        steps = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, dtype=torch.float32)
        logsnrs = self.compute_logsnr(steps)
        self.logsnrs = logsnrs.to(device)
        self.timesteps = torch.arange(self.num_inference_steps, device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SiD2SchedulerOutput, Tuple[torch.Tensor, torch.Tensor]]:
        if self.logsnrs is None:
            raise ValueError("LogSNRs not initialized. Call `set_timesteps` first.")
        if self.num_inference_steps is None:
            raise ValueError("num_inference_steps not set. Call `set_timesteps` first.")

        i = int(timestep)
        logsnr_t = self.logsnrs[i]
        logsnr_s = self.logsnrs[i + 1]

        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        c = -torch.expm1(logsnr_t - logsnr_s)

        if self.prediction_type == "eps":
            x_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.prediction_type == "v":
            x_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        if self.clip_sample:
            x_pred = torch.clamp(x_pred, -1.0, 1.0)

        mu = alpha_s * (sample * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s**2) * c

        if i < self.num_inference_steps - 1:
            noise = randn_tensor(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            prev_sample = mu + torch.sqrt(variance) * noise
        else:
            prev_sample = mu

        if not return_dict:
            return (prev_sample, x_pred)
        return SiD2SchedulerOutput(prev_sample=prev_sample, pred_original_sample=x_pred)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        logsnrs = timesteps.float()
        logsnrs = self._append_dims(logsnrs, original_samples.ndim)
        alpha_t = torch.sqrt(torch.sigmoid(logsnrs))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnrs))
        return alpha_t * original_samples + sigma_t * noise

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Optional[int] = None,
    ) -> torch.Tensor:
        del timestep
        return sample
