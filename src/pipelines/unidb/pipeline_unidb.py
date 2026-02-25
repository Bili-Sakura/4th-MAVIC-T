# Copyright 2024 The UniDB Authors (https://github.com/2769433owo/UniDB-plusplus).
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Pipeline for UniDB/UniDB++ compatible with the Hugging Face diffusers library.

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class UniDBPipelineOutput(BaseOutput):
    """Output class for UniDB pipelines.

    Args:
        images: Generated images.
        nfe: Number of function evaluations.
        sampler: Samver used.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0
    sampler: str = "unidb"


class UniDBPipeline(DiffusionPipeline):
    """Pipeline for UniDB/UniDB++ image-to-image translation.

    UniDB uses a noise-predicting model with (xt, mu, t) where mu is the
    condition (LQ/source image). The scheduler supports euler, noise-solver-*,
    and data-solver-* methods.
    """

    model_cpu_offload_seq = "unet"

    def __init__(
        self,
        unet,
        scheduler,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 100,
        cfg_scale: float = 1.0,
        method: Optional[str] = None,
        solver_type: Optional[str] = None,
        solver_step: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        """Run UniDB inference.

        Args:
            image: Condition image (LQ/source). Shape (B,C,H,W) or PIL.
            num_inference_steps: Number of denoising steps.
            method: Override scheduler method (euler, noise-solver-1, etc.).
            solver_type: Override solver_type (sde, mean-ode, pf-ode).
            solver_step: Override solver_step for UniDB++ (5, 10, 20, 25, 50, 100).
            generator: Random generator.
            output_type: "pil" or "np" or "pt".
            return_dict: Whether to return UniDBPipelineOutput.
            callback: Callback(i, total, x_t).
            callback_steps: Call callback every N steps.
        """
        if method is not None:
            self.scheduler.config.method = method
        if solver_type is not None:
            self.scheduler.config.solver_type = solver_type
        if solver_step is not None:
            self.scheduler.config.solver_step = solver_step

        # LQ (condition) = input image. UniDB test uses feed_data(LQ, LQ, GT):
        # state=LQ, condition=LQ. So we start from x = LQ at t=T and reverse to x0.
        x_T = self._preprocess_image(image)
        x_T = x_T.to(device=self.device, dtype=self.dtype)
        use_cfg = abs(float(cfg_scale) - 1.0) > 1e-6
        null_condition = torch.zeros_like(x_T) if use_cfg else None
        nfe_per_denoise = 2 if use_cfg else 1
        model_device = self.device
        model_dtype = self.dtype

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self.scheduler._mu = x_T  # condition (LQ)

        # Start from x = LQ (as in UniDB test). The reverse process denoises LQ -> clean.
        x = x_T.clone()

        nfe = 0
        method = self.scheduler.config.method
        solver_type = self.scheduler.config.solver_type

        # Euler: step through t = T, T-1, ..., 1
        # timesteps are [T, T-1, ..., 1] for euler, or reduced for solver_step
        step_size = (
            1
            if self.scheduler.config.method == "euler"
            else self.scheduler.T // self.scheduler.config.solver_step
        )

        for i in tqdm(range(len(self.scheduler.timesteps)), desc="UniDB"):
            t_val = int(self.scheduler.timesteps[i].item())

            t_tensor = torch.full(
                (x.shape[0],),
                t_val,
                device=x.device,
                dtype=torch.long,
            )

            if use_cfg:
                model_input = torch.cat([x, x], dim=0).to(device=model_device, dtype=model_dtype)
                cond_input = torch.cat([x_T, null_condition], dim=0).to(
                    device=model_device, dtype=model_dtype
                )
                timestep_input = torch.cat([t_tensor, t_tensor], dim=0)
                noise_pred_batched = self.unet(model_input, cond_input, timestep_input)
                noise_pred_cond, noise_pred_uncond = noise_pred_batched.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.unet(
                    x.to(device=model_device, dtype=model_dtype),
                    x_T.to(device=model_device, dtype=model_dtype),
                    t_tensor,
                )
            noise_pred = noise_pred.to(device=x.device, dtype=x.dtype)
            nfe += nfe_per_denoise

            scheduler_output = self.scheduler.step(
                model_output=noise_pred,
                timestep=i,
                sample=x,
                x_T=x_T,
                generator=generator,
                return_dict=True,
            )
            x = scheduler_output.prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, len(self.scheduler.timesteps), x)

        if output_type == "pil":
            images = self._tensor_to_pil(x)
        elif output_type == "np":
            images = x.cpu().numpy()
        else:
            images = x

        if not return_dict:
            return images

        return UniDBPipelineOutput(
            images=images,
            nfe=nfe,
            sampler=f"unidb-{method}-{solver_type}",
        )

    def _preprocess_image(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
    ) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = [image]
        if isinstance(image, list):
            images = []
            for img in image:
                if isinstance(img, Image.Image):
                    img = np.array(img)
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                images.append(img)
            x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
        elif isinstance(image, torch.Tensor):
            x = image
            if x.ndim == 3:
                x = x.unsqueeze(0)
            if x.max() > 1.0:
                x = x / 255.0
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        return x

    def _tensor_to_pil(self, x: torch.Tensor) -> List[Image.Image]:
        x = (x.clamp(0, 1) * 255).round().byte()
        x = x.cpu().permute(0, 2, 3, 1).numpy()
        return [Image.fromarray(xi) for xi in x]
