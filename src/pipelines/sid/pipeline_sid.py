# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Pipeline for conditional Simple Diffusion (SiD) image translation."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline, UNet2DModel
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from src.schedulers.scheduling_sid import SiDScheduler


@dataclass
class SIDPipelineOutput(BaseOutput):
    """Output class for SID pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class SIDPipeline(DiffusionPipeline):
    """Conditional Simple Diffusion pipeline for paired translation."""

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: UNet2DModel, scheduler: SiDScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert input image(s) to BCHW tensor in [-1, 1]."""
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and image and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img = img.convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                images.append(img_tensor)
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.max() > 1.0:
            image = image / 255.0
        if image.min() >= 0:
            image = image * 2 - 1

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 40,
        cfg_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        """Translate ``source_image`` with conditional SiD denoising."""
        device = self.device
        dtype = self.dtype
        source = self.prepare_inputs(source_image, device, dtype)
        batch_size = source.shape[0]
        target_channels = int(getattr(self.unet.config, "out_channels", source.shape[1]))
        # Determine if the model was built with concat conditioning by checking
        # whether in_channels exceeds out_channels.
        use_conditioning = self.unet.config.in_channels > target_channels
        use_cfg = abs(float(cfg_scale) - 1.0) > 1e-6
        null_condition = torch.zeros_like(source) if use_cfg else None
        nfe_per_denoise = 2 if use_cfg else 1
        model_device = self.device
        model_dtype = self.dtype

        xt = randn_tensor(
            (batch_size, target_channels, source.shape[-2], source.shape[-1]),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        nfe_count = 0
        progress_bar = tqdm(range(num_inference_steps), desc="SID Sampling")
        denom = max(num_inference_steps - 1, 1)
        max_train_t = float(self.scheduler.config.num_train_timesteps - 1)

        for i in progress_bar:
            t_cont = 1.0 - (float(i) / float(denom))
            t_batch = torch.full(
                (batch_size,),
                t_cont * max_train_t,
                device=device,
                dtype=dtype,
            )

            if use_cfg:
                model_input = torch.cat([xt, xt], dim=0).to(
                    device=model_device, dtype=model_dtype
                )
                timestep_input = torch.cat([t_batch, t_batch], dim=0)
                cond_input = torch.cat([source, null_condition], dim=0).to(
                    device=model_device, dtype=model_dtype
                )
                if use_conditioning:
                    model_input = torch.cat([model_input, cond_input], dim=1)
                model_output_batched = self.unet(model_input, timestep_input).sample
                model_output_cond, model_output_uncond = model_output_batched.chunk(2, dim=0)
                model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
            else:
                xt_dev = xt.to(device=model_device, dtype=model_dtype)
                if use_conditioning:
                    xt_dev = torch.cat(
                        [xt_dev, source.to(device=model_device, dtype=model_dtype)], dim=1
                    )
                model_output = self.unet(xt_dev, t_batch).sample
            model_output = model_output.to(device=xt.device, dtype=xt.dtype)
            nfe_count += nfe_per_denoise

            step_output = self.scheduler.step(
                model_output=model_output,
                timestep=i,
                sample=xt,
                generator=generator,
                return_dict=True,
            )
            xt = step_output.prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, num_inference_steps, xt)

        images = xt.clamp(-1, 1)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe_count)
        return SIDPipelineOutput(images=images, nfe=nfe_count)

    def _convert_to_pil(self, images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        pil_images = []
        for img in images:
            if img.shape[2] == 1:
                img = img.squeeze(2)
            pil_images.append(Image.fromarray(img))
        return pil_images

    def _convert_to_numpy(self, images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).numpy()
