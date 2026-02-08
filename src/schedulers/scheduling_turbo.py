"""Scheduler wrapper for Img2Image-Turbo single-step denoising.

Ported from ``vendor/Img2Image-Turbo/src/model.py::make_1step_sched``.
"""

from __future__ import annotations

from diffusers import DDPMScheduler


def make_1step_sched(
    pretrained_model_name_or_path: str = "stabilityai/sd-turbo",
    device: str = "cuda",
) -> DDPMScheduler:
    """Create a single-step DDPM noise scheduler for Img2Image-Turbo.

    The scheduler is initialised from the SD-Turbo checkpoint, configured for
    one timestep, and its ``alphas_cumprod`` tensor is moved to *device*.

    Parameters
    ----------
    pretrained_model_name_or_path : str
        HuggingFace model identifier for the base scheduler config.
    device : str
        Device to place the ``alphas_cumprod`` on.

    Returns
    -------
    DDPMScheduler
        Configured single-step scheduler.
    """
    noise_scheduler_1step = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_1step.set_timesteps(1, device=device)
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.to(device)
    return noise_scheduler_1step
