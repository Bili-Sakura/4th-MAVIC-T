"""Latent-space inference pipeline for BiBBDM.

Wraps the BiBBDM diffusion process with a frozen VAE so that the UNet
operates entirely in latent space while the pipeline accepts and
produces pixel-space images.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput

from ..schedulers.bibbdm_scheduler import BiBBDMScheduler
from ..models import BiBBDMUNet
from .bibbdm_pipeline import BiBBDMPipeline, BiBBDMPipelineOutput


@dataclass
class BiBBDMLatentPipelineOutput(BaseOutput):
    """Output class for the BiBBDM latent pipeline.

    Attributes
    ----------
    images : list of PIL.Image or ndarray or Tensor
        Generated images in pixel space.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class BiBBDMLatentPipeline(DiffusionPipeline):
    """BiBBDM pipeline that operates in VAE latent space.

    The pipeline encodes pixel-space source images into the latent space
    of a frozen VAE, runs the BiBBDM Brownian Bridge process in that
    latent space, and decodes the result back to pixel space.

    Parameters
    ----------
    unet : BiBBDMUNet
        A BiBBDM UNet trained on VAE latents.
    scheduler : BiBBDMScheduler
        The Brownian Bridge noise scheduler.
    vae : AutoencoderKL
        A frozen pre-trained VAE for encoding/decoding.
    """

    model_cpu_offload_seq = "vae->unet"

    def __init__(
        self,
        unet: BiBBDMUNet,
        scheduler: BiBBDMScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

        # Inner pixel-space pipeline for sampling logic (no clamp for pt)
        self._bibbdm = BiBBDMPipeline(unet=unet, scheduler=scheduler)

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
        source_image: torch.Tensor,
        direction: str = "b2a",
        num_inference_steps: Optional[int] = None,
        clip_denoised: bool = False,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
    ) -> Union[BiBBDMLatentPipelineOutput, tuple]:
        """Translate a source image via BiBBDM in VAE latent space.

        Accepts the same arguments as :class:`BiBBDMPipeline` but internally
        operates in the VAE latent space.
        """
        device = source_image.device
        orig_channels = source_image.shape[1]

        # Encode to latent space
        z_source = self._encode(source_image)

        # BiBBDM sampling in latent space (output_type="pt" avoids clamping)
        result = self._bibbdm(
            source_image=z_source,
            direction=direction,
            num_inference_steps=num_inference_steps,
            clip_denoised=clip_denoised,
            output_type="pt",
            generator=generator,
        )

        # Decode from latent to pixel space
        images = self._decode(result.images)
        images = self._restore_channels(images, orig_channels)
        images = images.clamp(-1, 1)

        # Format output
        return self._bibbdm._format_output(images, output_type)
