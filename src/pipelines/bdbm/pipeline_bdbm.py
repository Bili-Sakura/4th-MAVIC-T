"""Inference pipeline for BDBM (Bidirectional Diffusion Bridge Models)."""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from src.models.unet_bdbm import BDBMUNet
from src.schedulers.scheduling_bdbm import BDBMScheduler


@dataclass
class BDBMPipelineOutput(BaseOutput):
    """Output class for BDBM pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class BDBMPipeline(DiffusionPipeline):
    """Bidirectional image-to-image pipeline for BDBM."""

    def __init__(self, unet: BDBMUNet, scheduler: BDBMScheduler) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

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
        output_type: str,
        generator: Optional[torch.Generator],
        cfg_scale: float = 1.0,
    ) -> BDBMPipelineOutput:
        """Source -> target direction."""
        img = source.clone()
        context = self._make_context(source, direction="b2a")
        use_cfg = context is not None and abs(float(cfg_scale) - 1.0) > 1e-6
        null_context = torch.zeros_like(context) if use_cfg else None
        for i in tqdm(range(len(steps)), desc="BDBM b2a sampling", total=len(steps)):
            t = torch.full((img.shape[0],), int(steps[i].item()), device=img.device, dtype=torch.long)
            if use_cfg:
                model_input = torch.cat([img, img], dim=0)
                timestep_input = torch.cat([t, t], dim=0)
                context_input = torch.cat([context, null_context], dim=0)
                model_output_batched = self.unet(model_input, timestep_input, context=context_input)
                model_output_cond, model_output_uncond = model_output_batched.chunk(2, dim=0)
                model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
            else:
                model_output = self.unet(img, t, context=context)
            result = self.scheduler.step_b2a(
                model_output=model_output,
                step_index=i,
                x_t=img,
                source=source,
                clip_denoised=clip_denoised,
                generator=generator,
            )
            img = result.prev_sample
        return self._format_output(img, output_type)

    def _sample_a2b(
        self,
        target: torch.Tensor,
        asc_steps: torch.Tensor,
        clip_denoised: bool,
        output_type: str,
        generator: Optional[torch.Generator],
        cfg_scale: float = 1.0,
    ) -> BDBMPipelineOutput:
        """Target -> source direction."""
        img = target.clone()
        context = self._make_context(target, direction="a2b")
        use_cfg = context is not None and abs(float(cfg_scale) - 1.0) > 1e-6
        null_context = torch.zeros_like(context) if use_cfg else None
        for i in tqdm(range(len(asc_steps)), desc="BDBM a2b sampling", total=len(asc_steps)):
            t = torch.full((img.shape[0],), int(asc_steps[i].item()), device=img.device, dtype=torch.long)
            if use_cfg:
                model_input = torch.cat([img, img], dim=0)
                timestep_input = torch.cat([t, t], dim=0)
                context_input = torch.cat([context, null_context], dim=0)
                model_output_batched = self.unet(model_input, timestep_input, context=context_input)
                model_output_cond, model_output_uncond = model_output_batched.chunk(2, dim=0)
                model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
            else:
                model_output = self.unet(img, t, context=context)
            result = self.scheduler.step_a2b(
                model_output=model_output,
                step_index=i,
                x_t=img,
                target=target,
                clip_denoised=clip_denoised,
                generator=generator,
            )
            img = result.prev_sample
        return self._format_output(img, output_type)

    @staticmethod
    def _format_output(images: torch.Tensor, output_type: str) -> BDBMPipelineOutput:
        if output_type == "pt":
            return BDBMPipelineOutput(images=images)
        images_np = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return BDBMPipelineOutput(images=images_np)
        pil_images = []
        for arr in images_np:
            if arr.shape[2] == 1:
                arr = arr.squeeze(2)
            pil_images.append(Image.fromarray(arr))
        return BDBMPipelineOutput(images=pil_images)

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
    ) -> BDBMPipelineOutput:
        """Run bidirectional BDBM sampling."""
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)

        if self.scheduler.steps is None or self.scheduler.asc_steps is None:
            raise RuntimeError("Scheduler steps not initialised; call set_timesteps().")

        if direction == "b2a":
            return self._sample_b2a(
                source=source_image,
                steps=self.scheduler.steps,
                clip_denoised=clip_denoised,
                output_type=output_type,
                generator=generator,
                cfg_scale=cfg_scale,
            )
        if direction == "a2b":
            return self._sample_a2b(
                target=source_image,
                asc_steps=self.scheduler.asc_steps,
                clip_denoised=clip_denoised,
                output_type=output_type,
                generator=generator,
                cfg_scale=cfg_scale,
            )
        raise ValueError(f"Unknown direction: {direction!r}; expected 'b2a' or 'a2b'.")
