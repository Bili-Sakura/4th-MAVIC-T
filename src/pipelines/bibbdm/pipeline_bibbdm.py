# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Inference pipeline for BiBBDM.

Supports bidirectional image-to-image translation using the Brownian Bridge
diffusion process.  The primary mode for MAVIC-T tasks is **b2a** (source →
target).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from src.schedulers.scheduling_bibbdm import BiBBDMScheduler
from src.models.unet_bibbdm import BiBBDMUNet


@dataclass
class BiBBDMPipelineOutput(BaseOutput):
    """Output class for BiBBDM pipeline.

    Attributes
    ----------
    images : list of PIL.Image or ndarray or Tensor
        Generated images.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class BiBBDMPipeline(DiffusionPipeline):
    """Pipeline for bidirectional image-to-image translation using BiBBDM.

    This pipeline implements the sampling loop from the BiBBDM paper.  For
    MAVIC-T tasks the default direction is **b2a** (source → target).

    Parameters
    ----------
    unet : nn.Module
        The BiBBDMUNet denoising model.
    scheduler : BiBBDMScheduler
        The Brownian Bridge noise scheduler.
    """

    def __init__(self, unet: BiBBDMUNet, scheduler: BiBBDMScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        source_image: torch.Tensor,
        direction: str = "b2a",
        num_inference_steps: Optional[int] = None,
        clip_denoised: bool = False,
        cfg_scale: float = 1.0,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
    ) -> BiBBDMPipelineOutput:
        """Run the BiBBDM sampling loop.

        Parameters
        ----------
        source_image : Tensor  (B, C, H, W) in [-1, 1]
            The source/conditioning image.
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
        BiBBDMPipelineOutput
        """
        device = source_image.device

        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)

        steps = self.scheduler.steps
        if steps is None:
            raise RuntimeError("Scheduler steps not initialised; call set_timesteps().")

        if direction == "b2a":
            return self._sample_b2a(
                source_image,
                steps,
                clip_denoised,
                output_type,
                generator,
                cfg_scale=cfg_scale,
            )
        elif direction == "a2b":
            return self._sample_a2b(
                source_image,
                steps,
                clip_denoised,
                output_type,
                generator,
                cfg_scale=cfg_scale,
            )
        else:
            raise ValueError(f"Unknown direction: {direction!r}; expected 'b2a' or 'a2b'.")

    # ------------------------------------------------------------------

    def _sample_b2a(self, source, steps, clip_denoised, output_type, generator, cfg_scale: float = 1.0):
        """Source → Target (reverse Brownian Bridge)."""
        img = source.clone()
        use_cfg = abs(float(cfg_scale) - 1.0) > 1e-6
        null_condition = torch.zeros_like(source) if use_cfg else None
        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype
        for i in tqdm(range(len(steps)), desc="B2A sampling", total=len(steps)):
            t = torch.full((img.shape[0],), steps[i].item(), device=img.device, dtype=torch.long)
            if use_cfg:
                model_input = torch.cat([img, img], dim=0).to(device=model_device, dtype=model_dtype)
                timestep_input = torch.cat([t, t], dim=0)
                cond_input = torch.cat([source, null_condition], dim=0).to(
                    device=model_device, dtype=model_dtype
                )
                model_output_batched = self.unet(model_input, timestep_input, context=cond_input)
                model_output_cond, model_output_uncond = model_output_batched.chunk(2, dim=0)
                model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
            else:
                model_output = self.unet(
                    img.to(device=model_device, dtype=model_dtype),
                    t,
                    context=source.to(device=model_device, dtype=model_dtype),
                )
            model_output = model_output.to(device=img.device, dtype=img.dtype)
            result = self.scheduler.step_b2a(
                model_output, step_index=i, x_t=img, source=source,
                clip_denoised=clip_denoised, generator=generator,
            )
            img = result.prev_sample
        return self._format_output(img, output_type)

    def _sample_a2b(self, target, steps, clip_denoised, output_type, generator, cfg_scale: float = 1.0):
        """Target → Source (forward Brownian Bridge)."""
        img = target.clone()
        use_cfg = abs(float(cfg_scale) - 1.0) > 1e-6
        null_condition = torch.zeros_like(target) if use_cfg else None
        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype
        for i in tqdm(reversed(range(len(steps))), desc="A2B sampling", total=len(steps)):
            t = torch.full((img.shape[0],), steps[i].item(), device=img.device, dtype=torch.long)
            if use_cfg:
                model_input = torch.cat([img, img], dim=0).to(device=model_device, dtype=model_dtype)
                timestep_input = torch.cat([t, t], dim=0)
                cond_input = torch.cat([target, null_condition], dim=0).to(
                    device=model_device, dtype=model_dtype
                )
                model_output_batched = self.unet(model_input, timestep_input, context=cond_input)
                model_output_cond, model_output_uncond = model_output_batched.chunk(2, dim=0)
                model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
            else:
                model_output = self.unet(
                    img.to(device=model_device, dtype=model_dtype),
                    t,
                    context=target.to(device=model_device, dtype=model_dtype),
                )
            model_output = model_output.to(device=img.device, dtype=img.dtype)
            result = self.scheduler.step_a2b(
                model_output, step_index=i, x_t=img, target=target,
                clip_denoised=clip_denoised, generator=generator,
            )
            img = result.prev_sample
        return self._format_output(img, output_type)

    # ------------------------------------------------------------------

    @staticmethod
    def _format_output(images, output_type):
        if output_type == "pt":
            return BiBBDMPipelineOutput(images=images)
        images_np = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return BiBBDMPipelineOutput(images=images_np)
        pil_images = []
        for arr in images_np:
            if arr.shape[2] == 1:
                arr = arr.squeeze(2)
            pil_images.append(Image.fromarray(arr))
        return BiBBDMPipelineOutput(images=pil_images)
