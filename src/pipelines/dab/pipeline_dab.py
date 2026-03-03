# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Inference pipeline for DAB (Dual-Approximator Bridge)."""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from src.models.unet_dab import DABUNet
from src.schedulers.scheduling_dab import DABScheduler


@dataclass
class DABPipelineOutput(BaseOutput):
    """Output class for DAB pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class DABPipeline(DiffusionPipeline):
    """Image-to-image pipeline for DAB (source -> target)."""

    def __init__(self, unet: DABUNet, scheduler: DABScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    def _make_context(self, source: torch.Tensor) -> Optional[torch.Tensor]:
        mode = getattr(self.unet, "condition_mode", "concat")
        if mode in (None, "nocond"):
            return None
        if mode == "concat":
            return source
        raise ValueError(f"Unknown condition_mode: {mode}")

    @staticmethod
    def _format_output(images: torch.Tensor, output_type: str) -> DABPipelineOutput:
        if output_type == "pt":
            return DABPipelineOutput(images=images)
        images_np = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return DABPipelineOutput(images=images_np)
        pil_images = []
        for arr in images_np:
            if arr.shape[2] == 1:
                arr = arr.squeeze(2)
            pil_images.append(Image.fromarray(arr))
        return DABPipelineOutput(images=pil_images)

    @torch.no_grad()
    def __call__(
        self,
        source_image: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        clip_denoised: bool = False,
        cfg_scale: float = 1.0,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
    ) -> DABPipelineOutput:
        """Run DAB sampling from source to target."""
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)

        if self.scheduler.steps is None:
            raise RuntimeError("Scheduler steps not initialised; call set_timesteps().")

        img = source_image.clone()
        context = self._make_context(source_image)
        use_cfg = context is not None and abs(float(cfg_scale) - 1.0) > 1e-6
        null_context = torch.zeros_like(context) if use_cfg else None
        model_device = next(self.unet.parameters()).device
        model_dtype = next(self.unet.parameters()).dtype

        for i in tqdm(range(len(self.scheduler.steps)), desc="DAB sampling", total=len(self.scheduler.steps)):
            t = torch.full(
                (img.shape[0],),
                int(self.scheduler.steps[i].item()),
                device=img.device,
                dtype=torch.long,
            )
            if use_cfg:
                model_input = torch.cat([img, img], dim=0).to(device=model_device, dtype=model_dtype)
                timestep_input = torch.cat([t, t], dim=0)
                context_input = torch.cat([context, null_context], dim=0).to(
                    device=model_device, dtype=model_dtype
                )
                model_output_batched = self.unet(model_input, timestep_input, context=context_input)
                model_output_cond, model_output_uncond = model_output_batched.chunk(2, dim=0)
                model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
            else:
                ctx = context.to(device=model_device, dtype=model_dtype) if context is not None else None
                model_output = self.unet(
                    img.to(device=model_device, dtype=model_dtype),
                    t,
                    context=ctx,
                )
            model_output = model_output.to(device=img.device, dtype=img.dtype)
            result = self.scheduler.step(
                model_output=model_output,
                step_index=i,
                x_t=img,
                source=source_image,
                clip_denoised=clip_denoised,
                generator=generator,
            )
            img = result.prev_sample

        return self._format_output(img, output_type)
