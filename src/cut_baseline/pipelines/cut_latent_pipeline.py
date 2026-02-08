"""Latent-space inference pipeline for CUT.

Wraps the CUT generator with a frozen VAE so that the generator
operates entirely in latent space while the pipeline accepts and
produces pixel-space images.
"""

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput

from ..models import CUTGenerator
from .cut_pipeline import CUTPipeline


@dataclass
class CUTLatentPipelineOutput(BaseOutput):
    """Output class for the CUT latent pipeline.

    Attributes
    ----------
    images : list of PIL.Image or ndarray or Tensor
        Generated images in pixel space.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class CUTLatentPipeline(DiffusionPipeline):
    """CUT pipeline that operates in VAE latent space.

    The pipeline encodes pixel-space source images into the latent space
    of a frozen VAE, runs the CUT generator in that latent space, and
    decodes the result back to pixel space.

    Parameters
    ----------
    generator : CUTGenerator
        A CUT generator trained on VAE latents.
    vae : AutoencoderKL
        A frozen pre-trained VAE for encoding/decoding.
    """

    def __init__(self, generator: CUTGenerator, vae: AutoencoderKL) -> None:
        super().__init__()
        self.register_modules(generator=generator, vae=vae)

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.generator.parameters()).dtype

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
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[CUTLatentPipelineOutput, tuple]:
        """Translate a source image via CUT in VAE latent space.

        Accepts the same arguments as :class:`CUTPipeline` but internally
        operates in the VAE latent space.
        """
        device = self.device
        dtype = self.dtype

        # Prepare pixel inputs (reuse CUTPipeline's logic)
        x_pixel = CUTPipeline.prepare_inputs(self, source_image, device, dtype)
        orig_channels = x_pixel.shape[1]

        # Encode to latent space
        z = self._encode(x_pixel)

        # Single forward pass in latent space
        z_out = self.generator(z)

        # Decode from latent to pixel space
        images = self._decode(z_out)
        images = self._restore_channels(images, orig_channels)
        images = images.clamp(-1, 1)

        if output_type == "pil":
            images = CUTPipeline._convert_to_pil(images)
        elif output_type == "np":
            images = CUTPipeline._convert_to_numpy(images)

        if not return_dict:
            return (images,)

        return CUTLatentPipelineOutput(images=images)
