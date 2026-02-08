"""Latent-space inference pipeline for DDBM.

Wraps the DDBM diffusion process with a frozen VAE so that the UNet
operates entirely in latent space while the pipeline accepts and
produces pixel-space images.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from ..schedulers.ddbm_scheduler import DDBMScheduler
from ..models import DDBMUNet
from .ddbm_pipeline import DDBMPipeline


@dataclass
class DDBMLatentPipelineOutput(BaseOutput):
    """Output class for the DDBM latent pipeline.

    Attributes
    ----------
    images : list of PIL.Image or ndarray or Tensor
        Generated images in pixel space.
    nfe : int
        Number of function evaluations during sampling.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class DDBMLatentPipeline(DiffusionPipeline):
    """DDBM pipeline that operates in VAE latent space.

    The pipeline encodes pixel-space source images into the latent space
    of a frozen VAE, runs the DDBM bridge diffusion process in that
    latent space, and decodes the result back to pixel space.

    Parameters
    ----------
    unet : DDBMUNet
        A DDBM UNet trained on VAE latents.
    scheduler : DDBMScheduler
        The DDBM noise scheduler.
    vae : AutoencoderKL
        A frozen pre-trained VAE for encoding/decoding.
    """

    model_cpu_offload_seq = "vae->unet"

    def __init__(
        self,
        unet: DDBMUNet,
        scheduler: DDBMScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

        # Wrap a pixel-space DDBMPipeline to reuse its helper methods
        self._ddbm = DDBMPipeline(unet=unet, scheduler=scheduler)

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        """Repeat single-channel images to 3 channels for the VAE."""
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Average 3-channel VAE output back to 1 channel if needed."""
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
        num_inference_steps: int = 40,
        guidance: float = 1.0,
        churn_step_ratio: float = 0.33,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        """Translate a source image via DDBM in VAE latent space.

        Accepts the same arguments as :class:`DDBMPipeline` but internally
        operates in the VAE latent space.
        """
        device = self._ddbm.device
        dtype = self._ddbm.dtype

        # Prepare pixel inputs
        x_pixel = self._ddbm.prepare_inputs(source_image, device, dtype)
        orig_channels = x_pixel.shape[1]

        # Encode to latent space
        z_T = self._encode(x_pixel)
        batch_size = z_T.shape[0]

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sigmas = self.scheduler.sigmas

        z = z_T.clone()
        s_in = z.new_ones([batch_size])

        # Sampling loop (same as DDBMPipeline but without intermediate/final clamp)
        nfe = 0
        progress_bar = tqdm(range(len(sigmas) - 1), desc="DDBM Latent Sampling")

        for i in progress_bar:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            if churn_step_ratio > 0 and sigma_next != 0:
                sigma_hat = (sigma_next - sigma) * churn_step_ratio + sigma

                denoised = self._ddbm.denoise(z, sigma * s_in, z_T, clip_denoised=False)
                nfe += 1

                d_1, gt2 = self._ddbm._get_d_stochastic(z, sigma, denoised, z_T, guidance)
                dt = sigma_hat - sigma
                noise = randn_tensor(z.shape, generator=generator, device=device, dtype=dtype)
                z = z + d_1 * dt + noise * (dt.abs() ** 0.5) * gt2.sqrt()
            else:
                sigma_hat = sigma

            denoised = self._ddbm.denoise(z, sigma_hat * s_in, z_T, clip_denoised=False)
            nfe += 1

            d = self._ddbm._get_d(z, sigma_hat, denoised, z_T, guidance)
            dt = sigma_next - sigma_hat

            if sigma_next == 0:
                z = z + d * dt
            else:
                z_2 = z + d * dt
                denoised_2 = self._ddbm.denoise(z_2, sigma_next * s_in, z_T, clip_denoised=False)
                nfe += 1
                d_2 = self._ddbm._get_d(z_2, sigma_next, denoised_2, z_T, guidance)
                d_prime = (d + d_2) / 2
                z = z + d_prime * dt

            if callback is not None and i % callback_steps == 0:
                callback(i, num_inference_steps, z)

        # Decode from latent to pixel space
        images = self._decode(z)
        images = self._restore_channels(images, orig_channels)
        images = images.clamp(-1, 1)

        if output_type == "pil":
            images = self._ddbm._convert_to_pil(images)
        elif output_type == "np":
            images = self._ddbm._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe)

        return DDBMLatentPipelineOutput(images=images, nfe=nfe)
