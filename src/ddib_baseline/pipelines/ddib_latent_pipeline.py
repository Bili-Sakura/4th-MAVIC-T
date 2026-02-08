"""Latent-space inference pipeline for DDIB.

Wraps the Dual Diffusion Implicit Bridges process with a frozen VAE
so that both UNets operate in latent space while the pipeline accepts
and produces pixel-space images.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput

from ..schedulers.ddib_scheduler import DDIBScheduler
from ..models import DDIBUNet
from .ddib_pipeline import DDIBPipeline


@dataclass
class DDIBLatentPipelineOutput(BaseOutput):
    """Output class for the DDIB latent pipeline.

    Attributes
    ----------
    images : list of PIL.Image or ndarray or Tensor
        Generated images in pixel space.
    latent : Tensor or None
        Shared latent representation (optional, for debugging).
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    latent: Optional[torch.Tensor] = None


class DDIBLatentPipeline(DiffusionPipeline):
    """DDIB pipeline that operates in VAE latent space.

    The pipeline encodes pixel-space source images into the latent space
    of a frozen VAE, runs the DDIB encode→decode process in that latent
    space, and decodes the result back to pixel space.

    Parameters
    ----------
    source_unet : DDIBUNet
        Diffusion model trained on the source domain (in latent space).
    target_unet : DDIBUNet
        Diffusion model trained on the target domain (in latent space).
    scheduler : DDIBScheduler
        Shared DDIB scheduler.
    vae : AutoencoderKL
        A frozen pre-trained VAE for encoding/decoding.
    """

    model_cpu_offload_seq = "vae->source_unet->target_unet"

    def __init__(
        self,
        source_unet: DDIBUNet,
        target_unet: DDIBUNet,
        scheduler: DDIBScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()
        self.register_modules(
            source_unet=source_unet,
            target_unet=target_unet,
            scheduler=scheduler,
            vae=vae,
        )

        # Inner pipeline for DDIM encode/decode loops (they don't clamp)
        self._ddib = DDIBPipeline(
            source_unet=source_unet,
            target_unet=target_unet,
            scheduler=scheduler,
        )

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
        num_inference_steps: int = 250,
        clip_denoised: bool = True,
        eta: float = 0.0,
        output_type: str = "pil",
        return_dict: bool = True,
        return_latent: bool = False,
    ):
        """Translate a source image via DDIB in VAE latent space.

        Accepts the same arguments as :class:`DDIBPipeline` but internally
        operates in the VAE latent space.
        """
        device = self._ddib.device
        dtype = self._ddib.dtype

        # Prepare pixel inputs
        x_pixel = self._ddib.prepare_inputs(source_image, device, dtype)
        orig_channels = x_pixel.shape[1]

        # Encode to latent space
        z_source = self._encode(x_pixel)

        # Build timestep sequences
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # DDIM reverse: source latent → shared noise (no clamp in sub-loops)
        z_latent = self._ddib._ddim_reverse_sample_loop(
            self.source_unet, z_source, timesteps, clip_denoised=clip_denoised,
        )

        # DDIM forward: shared noise → target latent
        z_target = self._ddib._ddim_sample_loop(
            self.target_unet, z_latent, timesteps, clip_denoised=clip_denoised, eta=eta,
        )

        # Decode from latent to pixel space
        images = self._decode(z_target)
        images = self._restore_channels(images, orig_channels)
        images = images.clamp(-1, 1)

        if output_type == "pil":
            images = DDIBPipeline._convert_to_pil(images)
        elif output_type == "np":
            images = DDIBPipeline._convert_to_numpy(images)

        if not return_dict:
            return (images,)

        return DDIBLatentPipelineOutput(
            images=images,
            latent=z_latent if return_latent else None,
        )
