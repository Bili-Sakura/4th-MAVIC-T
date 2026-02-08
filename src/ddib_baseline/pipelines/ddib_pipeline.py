# Copyright 2024 The DDIB Authors and The Hugging Face Team.
# Licensed under the MIT License (the "License");
#
# Pipeline for Dual Diffusion Implicit Bridges (DDIB) compatible with
# the Hugging Face diffusers library.
#
# DDIB performs image-to-image translation by:
#   1. Encoding the source image to a shared latent via DDIM reverse sampling
#      with the source-domain model.
#   2. Decoding the latent to the target domain via DDIM forward sampling
#      with the target-domain model.

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from ..schedulers.ddib_scheduler import DDIBScheduler
from ..models import DDIBUNet


@dataclass
class DDIBPipelineOutput(BaseOutput):
    """
    Output class for the DDIB pipeline.

    Args:
        images: List of generated PIL images, numpy array, or torch tensor.
        latent: The shared latent representation (optional, for debugging).
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    latent: Optional[torch.Tensor] = None


class DDIBPipeline(DiffusionPipeline):
    r"""
    Pipeline for image-to-image translation using Dual Diffusion Implicit Bridges.

    DDIB (ICLR 2023) translates between two domains by concatenating a source-to-latent
    DDIM reverse ODE with a latent-to-target DDIM forward ODE, using two independently
    trained diffusion models.

    Args:
        source_unet: Diffusion model trained on the **source** domain.
        target_unet: Diffusion model trained on the **target** domain.
        scheduler: A :class:`DDIBScheduler` shared by both models.
    """

    model_cpu_offload_seq = "source_unet->target_unet"

    def __init__(
        self,
        source_unet: DDIBUNet,
        target_unet: DDIBUNet,
        scheduler: DDIBScheduler,
    ):
        super().__init__()
        self.register_modules(
            source_unet=source_unet,
            target_unet=target_unet,
            scheduler=scheduler,
        )

    @property
    def device(self) -> torch.device:
        return next(self.target_unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.target_unet.parameters()).dtype

    # ------------------------------------------------------------------
    # DDIM reverse loop (encode: x_0 → x_T using source model)
    # ------------------------------------------------------------------

    def _ddim_reverse_sample_loop(
        self,
        model: torch.nn.Module,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Run DDIM reverse sampling to encode ``x_0`` into the latent ``x_T``."""
        x = x_0
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = t_cur.expand(x.shape[0])
            scaled_t = self.scheduler._scale_timesteps(t_batch)
            model_output = model(x, scaled_t)

            out = self.scheduler.ddim_reverse_step(
                model_output=model_output,
                timestep=t_batch,
                timestep_next=t_next.expand(x.shape[0]),
                sample=x,
                clip_denoised=clip_denoised,
            )
            x = out.prev_sample
        return x

    # ------------------------------------------------------------------
    # DDIM forward loop (decode: x_T → x_0 using target model)
    # ------------------------------------------------------------------

    def _ddim_sample_loop(
        self,
        model: torch.nn.Module,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = True,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Run DDIM forward sampling to decode a latent ``x_T`` into ``x_0``."""
        x = noise
        reversed_timesteps = timesteps.flip(0)
        for i in range(len(reversed_timesteps) - 1):
            t_cur = reversed_timesteps[i]
            t_prev = reversed_timesteps[i + 1]

            t_batch = t_cur.expand(x.shape[0])
            scaled_t = self.scheduler._scale_timesteps(t_batch)
            model_output = model(x, scaled_t)

            out = self.scheduler.ddim_step(
                model_output=model_output,
                timestep=t_batch,
                timestep_prev=t_prev.expand(x.shape[0]),
                sample=x,
                eta=eta,
                clip_denoised=clip_denoised,
            )
            x = out.prev_sample
        return x

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert PIL / numpy / tensor inputs to normalised ``[-1, 1]`` tensors."""
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img_array = np.array(img).astype(np.float32) / 255.0
                if img_array.ndim == 2:
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                else:
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

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 250,
        clip_denoised: bool = True,
        eta: float = 0.0,
        output_type: str = "pil",
        return_dict: bool = True,
        return_latent: bool = False,
    ):
        """
        Translate *source_image* from the source domain to the target domain.

        Args:
            source_image: Source image(s) in ``[-1, 1]`` or ``[0, 1]`` or PIL.
            num_inference_steps: Number of DDIM steps for both encode and decode.
            clip_denoised: Clip predicted ``x_0`` to ``[-1, 1]``.
            eta: DDIM eta (0 = deterministic).
            output_type: ``"pil"`` | ``"np"`` | ``"pt"``.
            return_dict: If ``True`` return a :class:`DDIBPipelineOutput`.
            return_latent: If ``True`` include the shared latent in the output.

        Returns
        -------
        :class:`DDIBPipelineOutput` or tuple of images.
        """
        device = self.device
        dtype = self.dtype

        x_source = self.prepare_inputs(source_image, device, dtype)

        # Build timestep sequences
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # ascending: [0, step, 2*step, ...]

        # 1. Encode: source → latent (DDIM reverse with source model)
        latent = self._ddim_reverse_sample_loop(
            self.source_unet, x_source, timesteps, clip_denoised=clip_denoised,
        )

        # 2. Decode: latent → target (DDIM forward with target model)
        images = self._ddim_sample_loop(
            self.target_unet, latent, timesteps, clip_denoised=clip_denoised, eta=eta,
        )

        images = images.clamp(-1, 1)

        # Post-process
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images,)

        return DDIBPipelineOutput(
            images=images,
            latent=latent if return_latent else None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in images]

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).numpy()
