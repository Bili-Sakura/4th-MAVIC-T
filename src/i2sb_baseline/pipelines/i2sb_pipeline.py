# Copyright 2024 The I2SB Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Pipeline for Image-to-Image Schrödinger Bridge (I2SB) compatible with
# the Hugging Face diffusers library.

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from ..schedulers.i2sb_scheduler import I2SBScheduler
from ..models import I2SBUNet


@dataclass
class I2SBPipelineOutput(BaseOutput):
    """
    Output class for I2SB pipeline.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray` or `torch.Tensor`):
            List of denoised PIL images of length `batch_size` or NumPy array or torch tensor of shape
            `(batch_size, height, width, num_channels)`.
        nfe (`int`):
            Number of function evaluations (model forward passes) used during sampling.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class I2SBPipeline(DiffusionPipeline):
    r"""
    Pipeline for image-to-image generation using Image-to-Image Schrödinger Bridge.

    This pipeline implements the I2SB algorithm from the paper
    [I2SB: Image-to-Image Schrödinger Bridge](https://arxiv.org/abs/2302.05872). I2SB learns to
    transform between two data distributions using a Schrödinger bridge, enabling high-quality
    image-to-image translation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        unet ([`torch.nn.Module`]):
            An I2SB UNet model for denoising.
        scheduler ([`I2SBScheduler`]):
            An `I2SBScheduler` for the Schrödinger bridge diffusion process.
    """

    model_cpu_offload_seq = "unet"

    def __init__(
        self,
        unet: I2SBUNet,
        scheduler: I2SBScheduler,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
        )

    @property
    def device(self) -> torch.device:
        """Get the device of the pipeline."""
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the pipeline."""
        return next(self.unet.parameters()).dtype

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Prepare input images for the pipeline.

        Converts PIL images or numpy arrays to normalized tensors in [-1, 1] range.
        """
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            # Convert PIL images to tensor
            images = []
            for img in image:
                img = img.convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                images.append(img_tensor)
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Ensure image is in [-1, 1] range
        if image.max() > 1.0:
            image = image / 255.0

        if image.min() >= 0:
            image = image * 2 - 1  # Convert [0, 1] to [-1, 1]

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        nfe: int = 100,
        ot_ode: bool = False,
        clip_denoise: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        """
        Generate images using the I2SB Schrödinger bridge diffusion process.

        Args:
            source_image: The source/condition image(s) for the bridge.
                Can be a tensor of shape (B, C, H, W) in [-1, 1] range,
                or PIL Image(s).
            nfe: Number of function evaluations / sampling steps (default: 100).
            ot_ode: If True, use deterministic OT-ODE path (default: False).
            clip_denoise: If True, clamp predicted x0 to [-1, 1] (default: False).
            generator: Random number generator for reproducibility.
            output_type: Output format - "pil", "np", or "pt" (default: "pil").
            return_dict: Whether to return a dict with the output (default: True).
            callback: Callback function for progress updates.
            callback_steps: Frequency of callback calls.

        Returns:
            Images generated through the Schrödinger bridge diffusion process.
        """
        device = self.device
        dtype = self.dtype

        # Prepare source image (x1 in I2SB terminology)
        x1 = self.prepare_inputs(source_image, device, dtype)
        batch_size = x1.shape[0]

        # Set timesteps
        self.scheduler.set_timesteps(nfe, device=device)
        steps = self.scheduler.timesteps  # (nfe + 1,) evenly spaced indices

        # Start from the source image at the last timestep
        xt = x1.clone()

        # Build noise levels for timestep embedding
        interval = self.scheduler.config.interval
        t0_val = self.scheduler.config.t0
        T_val = self.scheduler.config.T
        noise_levels = torch.linspace(t0_val, T_val, interval, device=device, dtype=dtype)

        # Condition for concat mode
        has_condition = hasattr(self.unet, 'condition_mode') and self.unet.condition_mode == 'concat'
        cond = x1 if has_condition else None

        # Backward sampling loop: iterate from large timestep to small
        num_steps = len(steps) - 1
        progress_bar = tqdm(range(num_steps), desc="I2SB Sampling")
        nfe_count = 0

        for i in progress_bar:
            # steps goes from 0 to interval-1; we sample backward (large → small)
            step = steps[num_steps - i]      # current step (later in diffusion)
            prev_step = steps[num_steps - i - 1]  # previous step (earlier)

            # Compute timestep embedding
            step_int = step.item() if isinstance(step, torch.Tensor) else step
            t_emb = noise_levels[step_int] * interval
            t_batch = torch.full((batch_size,), t_emb, device=device, dtype=dtype)

            # Model prediction
            pred = self.unet(xt, t_batch, cond=cond)
            nfe_count += 1

            # Recover predicted x0 and sample posterior
            pred_x0 = self.scheduler.compute_pred_x0(step_int, xt, pred, clip_denoise=clip_denoise)
            xt = self.scheduler.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, num_steps, xt)

        # Post-process output
        images = xt.clamp(-1, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        # else: output_type == "pt", return tensor as-is

        if not return_dict:
            return (images, nfe_count)

        return I2SBPipelineOutput(images=images, nfe=nfe_count)

    def _convert_to_pil(self, images: torch.Tensor) -> List[Image.Image]:
        """Convert tensor to PIL images."""
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
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
        """Convert tensor to numpy array."""
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images
