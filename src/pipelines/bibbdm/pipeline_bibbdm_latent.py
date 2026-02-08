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
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput

from src.schedulers.scheduling_bibbdm import BiBBDMScheduler
from src.models.unet_bibbdm import BiBBDMUNet


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

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        """Adapt single-channel images to 3-channel for VAE encoding."""
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Restore original channel count after VAE decoding."""
        if target_channels == 1 and images.shape[1] == 3:
            return images.mean(dim=1, keepdim=True)
        return images

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space images to VAE latent space.

        Parameters
        ----------
        images : torch.Tensor
            Input images in pixel space, shape (B, C, H, W) in [-1, 1].

        Returns
        -------
        torch.Tensor
            Encoded latents, shape (B, C_latent, H_latent, W_latent).
        """
        adapted = self._adapt_channels(images)
        posterior = self.vae.encode(adapted).latent_dist
        return posterior.mean * self.vae.config.scaling_factor

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to pixel space.

        Parameters
        ----------
        latents : torch.Tensor
            Latent representations, shape (B, C_latent, H_latent, W_latent).

        Returns
        -------
        torch.Tensor
            Decoded images in pixel space, shape (B, 3, H, W) in [-1, 1].
        """
        scaled = latents / self.vae.config.scaling_factor
        return self.vae.decode(scaled).sample

    # ------------------------------------------------------------------
    # Sampling methods
    # ------------------------------------------------------------------

    def _sample_b2a(
        self,
        source: torch.Tensor,
        steps: torch.Tensor,
        clip_denoised: bool,
        output_type: str,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Source → Target (reverse Brownian Bridge) in latent space.

        Parameters
        ----------
        source : torch.Tensor
            Source latent, shape (B, C_latent, H_latent, W_latent).
        steps : torch.Tensor
            Timestep schedule from scheduler.
        clip_denoised : bool
            Whether to clamp intermediate predictions to [-1, 1].
        output_type : str
            Output format: "pt", "np", or "pil".
        generator : torch.Generator or None
            RNG for reproducibility.

        Returns
        -------
        torch.Tensor
            Generated latents in pixel space after decoding.
        """
        device = source.device
        img = source.clone()

        for i in tqdm(range(len(steps)), desc="B2A sampling", total=len(steps)):
            t = torch.full((img.shape[0],), steps[i].item(), device=device, dtype=torch.long)
            model_output = self.unet(img, t, context=source)
            result = self.scheduler.step_b2a(
                model_output,
                step_index=i,
                x_t=img,
                source=source,
                clip_denoised=clip_denoised,
                generator=generator,
            )
            img = result.prev_sample

        return img

    def _sample_a2b(
        self,
        target: torch.Tensor,
        steps: torch.Tensor,
        clip_denoised: bool,
        output_type: str,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Target → Source (forward Brownian Bridge) in latent space.

        Parameters
        ----------
        target : torch.Tensor
            Target latent, shape (B, C_latent, H_latent, W_latent).
        steps : torch.Tensor
            Timestep schedule from scheduler.
        clip_denoised : bool
            Whether to clamp intermediate predictions to [-1, 1].
        output_type : str
            Output format: "pt", "np", or "pil".
        generator : torch.Generator or None
            RNG for reproducibility.

        Returns
        -------
        torch.Tensor
            Generated latents in pixel space after decoding.
        """
        device = target.device
        img = target.clone()

        for i in tqdm(reversed(range(len(steps))), desc="A2B sampling", total=len(steps)):
            t = torch.full((img.shape[0],), steps[i].item(), device=device, dtype=torch.long)
            model_output = self.unet(img, t, context=target)
            result = self.scheduler.step_a2b(
                model_output,
                step_index=i,
                x_t=img,
                target=target,
                clip_denoised=clip_denoised,
                generator=generator,
            )
            img = result.prev_sample

        return img

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(images: torch.Tensor, output_type: str) -> Union[BiBBDMLatentPipelineOutput, tuple]:
        """Format output images according to output_type.

        Parameters
        ----------
        images : torch.Tensor
            Images in pixel space, shape (B, C, H, W) in [-1, 1].
        output_type : str
            Output format: "pt" for tensors, "np" for numpy arrays, "pil" for PIL Images.

        Returns
        -------
        BiBBDMLatentPipelineOutput or tuple
            Formatted output.
        """
        if output_type == "pt":
            return BiBBDMLatentPipelineOutput(images=images)

        # Convert to numpy uint8 format
        images_np = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()

        if output_type == "np":
            return BiBBDMLatentPipelineOutput(images=images_np)

        # Convert to PIL Images
        pil_images = []
        for arr in images_np:
            if arr.shape[2] == 1:
                arr = arr.squeeze(2)
            pil_images.append(Image.fromarray(arr))

        return BiBBDMLatentPipelineOutput(images=pil_images)

    # ------------------------------------------------------------------
    # Main call method
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

        The pipeline encodes the source image to latent space, runs the
        BiBBDM Brownian Bridge diffusion process, and decodes the result
        back to pixel space.

        Parameters
        ----------
        source_image : torch.Tensor
            Source image in pixel space, shape (B, C, H, W) in [-1, 1].
        direction : str
            ``"b2a"`` for source→target (default) or ``"a2b"`` for reverse.
        num_inference_steps : int or None
            Override the scheduler's default step count.
        clip_denoised : bool
            Clamp intermediate predictions to [-1, 1].
        output_type : str
            ``"pt"`` for tensors, ``"pil"`` for PIL images, ``"np"`` for numpy.
        generator : torch.Generator or None
            RNG for reproducibility.

        Returns
        -------
        BiBBDMLatentPipelineOutput
            Generated images in the requested format.
        """
        device = source_image.device
        orig_channels = source_image.shape[1]

        # Set timesteps if specified
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)

        steps = self.scheduler.steps
        if steps is None:
            raise RuntimeError("Scheduler steps not initialised; call set_timesteps().")

        # Encode source image to latent space
        z_source = self._encode(source_image)

        # Run BiBBDM sampling in latent space
        if direction == "b2a":
            z_result = self._sample_b2a(z_source, steps, clip_denoised, output_type, generator)
        elif direction == "a2b":
            z_result = self._sample_a2b(z_source, steps, clip_denoised, output_type, generator)
        else:
            raise ValueError(f"Unknown direction: {direction!r}; expected 'b2a' or 'a2b'.")

        # Decode from latent to pixel space
        images = self._decode(z_result)
        images = self._restore_channels(images, orig_channels)
        images = images.clamp(-1, 1)

        # Format output
        return self._format_output(images, output_type)
