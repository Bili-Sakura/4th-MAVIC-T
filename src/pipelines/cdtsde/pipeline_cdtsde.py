"""Inference pipeline for CDTSDE baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from src.models.unet_cdtsde import CDTSDEUNet
from src.schedulers.scheduling_cdtsde import CDTSDEScheduler


@dataclass
class CDTSDEPipelineOutput(BaseOutput):
    """Output class for CDTSDE pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class CDTSDEPipeline(DiffusionPipeline):
    """Image-to-image translation pipeline using CDTSDE."""

    model_cpu_offload_seq = "unet"

    def __init__(self, unet: CDTSDEUNet, scheduler: CDTSDEScheduler):
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
        """Convert image-like inputs to BCHW tensors in [-1, 1]."""
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and image and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                if img.mode not in ("L", "RGB"):
                    img = img.convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                if arr.ndim == 2:
                    tensor = torch.from_numpy(arr).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(arr).permute(2, 0, 1)
                images.append(tensor)
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.max() > 1.0:
            image = image / 255.0
        if image.min() >= 0.0:
            image = image * 2.0 - 1.0

        return image.to(device=device, dtype=dtype)

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img.squeeze(-1) if img.shape[-1] == 1 else img) for img in images]

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1.0) / 2.0
        images = images.clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).numpy()

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 50,
        stochastic: bool = True,
        apply_domain_shift: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        device = self.device
        dtype = self.dtype

        x_T = self.prepare_inputs(source_image, device=device, dtype=dtype)
        batch_size = x_T.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if self.scheduler.timesteps is None:
            raise ValueError("Scheduler timesteps not initialized.")

        sqrt_alpha_last = self.scheduler.sqrt_alphas_cumprod[-1].to(device=device, dtype=dtype)
        sigma_last = self.scheduler.sigmas[-1].to(device=device, dtype=dtype)
        noise = randn_tensor(
            x_T.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        x = sqrt_alpha_last * x_T + sigma_last * noise

        nfe = 0
        total_steps = len(self.scheduler.timesteps) - 1
        progress = tqdm(range(total_steps), desc="CDTSDE Sampling")

        for i in progress:
            # j descends over reduced schedule indices: ... 3,2,1
            j = total_steps - i
            t_model = self.scheduler.timesteps[j]
            t_batch = torch.full((batch_size,), t_model, device=device, dtype=torch.long)

            pred_noise = self.unet(x, t_batch, xT=x_T)
            nfe += 1

            idx_batch = torch.full((batch_size,), j, device=device, dtype=torch.long)
            pred_x0 = self.scheduler.predict_start_from_noise(
                sample=x,
                timesteps=idx_batch,
                noise=pred_noise,
                use_inference_schedule=True,
            )

            if apply_domain_shift:
                lam_linear = self.scheduler.etas[j].to(device=device, dtype=dtype)
                lam_batch = torch.full((batch_size,), lam_linear, device=device, dtype=dtype)
                lam_hat = self.unet.predict_lambda(lam_batch, pred_x0.shape)
                pred_x0 = lam_hat * x_T + (1.0 - lam_hat) * pred_x0

            step_out = self.scheduler.step(
                pred_original_sample=pred_x0,
                step_index=j,
                sample=x,
                reference_sample=x_T,
                generator=generator if isinstance(generator, torch.Generator) else None,
                stochastic=stochastic,
                return_dict=True,
            )
            x = step_out.prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, total_steps, x)

        images = x.clamp(-1.0, 1.0)
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        elif output_type != "pt":
            raise ValueError(f"Unsupported output_type: {output_type}")

        if not return_dict:
            return images, nfe
        return CDTSDEPipelineOutput(images=images, nfe=nfe)

