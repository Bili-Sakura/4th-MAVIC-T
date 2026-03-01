# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Latent-space inference pipeline for BDBM."""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput

from src.models.unet_bdbm import BDBMUNet
from src.schedulers.scheduling_bdbm import BDBMScheduler


@dataclass
class BDBMLatentPipelineOutput(BaseOutput):
    """Output class for BDBM latent pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class BDBMLatentPipeline(DiffusionPipeline):
    """BDBM pipeline that runs diffusion in VAE latent space."""

    model_cpu_offload_seq = "vae->unet"

    def __init__(
        self,
        unet: BDBMUNet,
        scheduler: BDBMScheduler,
        vae: AutoencoderKL,
    ) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

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
        posterior = self.vae.encode(self._adapt_channels(images)).latent_dist
        return posterior.mean * self.vae.config.scaling_factor

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        scaled = latents / self.vae.config.scaling_factor
        return self.vae.decode(scaled).sample

    def _make_context(self, endpoint: torch.Tensor, direction: str) -> Optional[torch.Tensor]:
        mode = getattr(self.unet, "condition_mode", "dual")
        if mode in (None, "nocond"):
            return None
        if mode == "concat":
            return endpoint
        if mode == "dual":
            zeros = torch.zeros_like(endpoint)
            if direction == "b2a":
                return torch.cat((zeros, endpoint), dim=1)
            return torch.cat((endpoint, zeros), dim=1)
        raise ValueError(f"Unknown condition_mode: {mode}")

    def _sample_b2a(
        self,
        source: torch.Tensor,
        steps: torch.Tensor,
        clip_denoised: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        img = source.clone()
        context = self._make_context(source, direction="b2a")
        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype
        for i in tqdm(range(len(steps)), desc="BDBM latent b2a", total=len(steps)):
            t = torch.full((img.shape[0],), int(steps[i].item()), device=img.device, dtype=torch.long)
            model_output = self.unet(
                img.to(device=model_device, dtype=model_dtype),
                t,
                context=context.to(device=model_device, dtype=model_dtype),
            )
            model_output = model_output.to(device=img.device, dtype=img.dtype)
            img = self.scheduler.step_b2a(
                model_output=model_output,
                step_index=i,
                x_t=img,
                source=source,
                clip_denoised=clip_denoised,
                generator=generator,
            ).prev_sample
        return img

    def _sample_a2b(
        self,
        target: torch.Tensor,
        asc_steps: torch.Tensor,
        clip_denoised: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        img = target.clone()
        context = self._make_context(target, direction="a2b")
        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype
        for i in tqdm(range(len(asc_steps)), desc="BDBM latent a2b", total=len(asc_steps)):
            t = torch.full((img.shape[0],), int(asc_steps[i].item()), device=img.device, dtype=torch.long)
            model_output = self.unet(
                img.to(device=model_device, dtype=model_dtype),
                t,
                context=context.to(device=model_device, dtype=model_dtype),
            )
            model_output = model_output.to(device=img.device, dtype=img.dtype)
            img = self.scheduler.step_a2b(
                model_output=model_output,
                step_index=i,
                x_t=img,
                target=target,
                clip_denoised=clip_denoised,
                generator=generator,
            ).prev_sample
        return img

    @staticmethod
    def _format_output(images: torch.Tensor, output_type: str) -> BDBMLatentPipelineOutput:
        if output_type == "pt":
            return BDBMLatentPipelineOutput(images=images)
        images_np = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return BDBMLatentPipelineOutput(images=images_np)
        pil_images = []
        for arr in images_np:
            if arr.shape[2] == 1:
                arr = arr.squeeze(2)
            pil_images.append(Image.fromarray(arr))
        return BDBMLatentPipelineOutput(images=pil_images)

    @torch.no_grad()
    def __call__(
        self,
        source_image: torch.Tensor,
        direction: str = "b2a",
        num_inference_steps: Optional[int] = None,
        clip_denoised: bool = False,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
        target_channels: Optional[int] = None,
    ) -> BDBMLatentPipelineOutput:
        """Run BDBM sampling in latent space and decode to pixels."""
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)

        if self.scheduler.steps is None or self.scheduler.asc_steps is None:
            raise RuntimeError("Scheduler steps not initialised; call set_timesteps().")

        original_channels = source_image.shape[1]
        z_source = self._encode(source_image)

        if direction == "b2a":
            z_result = self._sample_b2a(
                source=z_source,
                steps=self.scheduler.steps,
                clip_denoised=clip_denoised,
                generator=generator,
            )
        elif direction == "a2b":
            z_result = self._sample_a2b(
                target=z_source,
                asc_steps=self.scheduler.asc_steps,
                clip_denoised=clip_denoised,
                generator=generator,
            )
        else:
            raise ValueError(f"Unknown direction: {direction!r}; expected 'b2a' or 'a2b'.")

        images = self._decode(z_result)
        images = self._restore_channels(images, target_channels or original_channels)
        images = images.clamp(-1, 1)
        return self._format_output(images, output_type)
