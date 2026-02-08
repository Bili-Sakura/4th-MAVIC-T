"""Latent-space inference pipeline for I2SB.

Wraps the I2SB Schrödinger Bridge process with a frozen VAE so that
the UNet operates entirely in latent space while the pipeline accepts
and produces pixel-space images.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput

from ..schedulers.i2sb_scheduler import I2SBScheduler
from ..models import I2SBUNet
from .i2sb_pipeline import I2SBPipeline


@dataclass
class I2SBLatentPipelineOutput(BaseOutput):
    """Output class for the I2SB latent pipeline.

    Attributes
    ----------
    images : list of PIL.Image or ndarray or Tensor
        Generated images in pixel space.
    nfe : int
        Number of function evaluations during sampling.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class I2SBLatentPipeline(DiffusionPipeline):
    """I2SB pipeline that operates in VAE latent space.

    The pipeline encodes pixel-space source images into the latent space
    of a frozen VAE, runs the I2SB Schrödinger Bridge process in that
    latent space, and decodes the result back to pixel space.

    Parameters
    ----------
    unet : I2SBUNet
        An I2SB UNet trained on VAE latents.
    scheduler : I2SBScheduler
        The I2SB scheduler.
    vae : AutoencoderKL
        A frozen pre-trained VAE for encoding/decoding.
    """

    model_cpu_offload_seq = "vae->unet"

    def __init__(
        self,
        unet: I2SBUNet,
        scheduler: I2SBScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

        # Inner pixel-space pipeline for helper methods
        self._i2sb = I2SBPipeline(unet=unet, scheduler=scheduler)

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        if target_channels == 1 and images.shape[1] == 3:
            return images.mean(dim=1, keepdim=True)
        return images

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        adapted = self._adapt_channels(images)
        posterior = self.vae.encode(adapted).latent_dist
        return posterior.mean * self.vae.config.scaling_factor

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        scaled = latents / self.vae.config.scaling_factor
        return self.vae.decode(scaled).sample

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        nfe: int = 100,
        ot_ode: bool = False,
        clip_denoise: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        """Translate a source image via I2SB in VAE latent space.

        Accepts the same arguments as :class:`I2SBPipeline` but internally
        operates in the VAE latent space.
        """
        device = self._i2sb.device
        dtype = self._i2sb.dtype

        # Prepare pixel inputs
        x_pixel = self._i2sb.prepare_inputs(source_image, device, dtype)
        orig_channels = x_pixel.shape[1]

        # Encode to latent space
        z1 = self._encode(x_pixel)
        batch_size = z1.shape[0]

        # Set timesteps
        self.scheduler.set_timesteps(nfe, device=device)
        steps = self.scheduler.timesteps

        zt = z1.clone()

        # Build noise levels for timestep embedding
        interval = self.scheduler.config.interval
        t0_val = self.scheduler.config.t0
        T_val = self.scheduler.config.T
        noise_levels = torch.linspace(t0_val, T_val, interval, device=device, dtype=dtype)

        has_condition = hasattr(self.unet, "condition_mode") and self.unet.condition_mode == "concat"
        cond = z1 if has_condition else None

        # Backward sampling loop
        num_steps = len(steps) - 1
        progress_bar = tqdm(range(num_steps), desc="I2SB Latent Sampling")
        nfe_count = 0

        for i in progress_bar:
            step = steps[num_steps - i]
            prev_step = steps[num_steps - i - 1]

            step_int = step.item() if isinstance(step, torch.Tensor) else step
            t_emb = noise_levels[step_int] * interval
            t_batch = torch.full((batch_size,), t_emb, device=device, dtype=dtype)

            pred = self.unet(zt, t_batch, cond=cond)
            nfe_count += 1

            # No clip_denoise for latent space
            pred_x0 = self.scheduler.compute_pred_x0(step_int, zt, pred, clip_denoise=False)
            zt = self.scheduler.p_posterior(prev_step, step, zt, pred_x0, ot_ode=ot_ode)

            if callback is not None and i % callback_steps == 0:
                callback(i, num_steps, zt)

        # Decode from latent to pixel space
        images = self._decode(zt)
        images = self._restore_channels(images, orig_channels)
        images = images.clamp(-1, 1)

        if output_type == "pil":
            images = self._i2sb._convert_to_pil(images)
        elif output_type == "np":
            images = self._i2sb._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe_count)

        return I2SBLatentPipelineOutput(images=images, nfe=nfe_count)
