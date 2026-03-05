# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2025 Chadebec et al. and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Pipeline for Latent Bridge Matching (LBM) compatible with
# the Hugging Face diffusers library.

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from src.schedulers.scheduling_lbm import LBMScheduler
from src.models import UNet2DWrapper as LBMUNet


@dataclass
class LBMPipelineOutput(BaseOutput):
    """
    Output class for LBM pipeline.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray` or `torch.Tensor`):
            List of denoised PIL images of length `batch_size` or NumPy array or torch tensor of shape
            `(batch_size, height, width, num_channels)`.
        nfe (`int`):
            Number of function evaluations (model forward passes) used during sampling.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class LBMPipeline(DiffusionPipeline):
    r"""
    Pipeline for image-to-image generation using Latent Bridge Matching.

    This pipeline implements the LBM algorithm from the paper
    `LBM: Latent Bridge Matching for Fast Image-to-Image Translation
    <https://arxiv.org/abs/2503.07535>`_.  LBM uses flow-matching to
    interpolate between source and target distributions, enabling single-step
    or few-step high-quality image-to-image translation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        unet ([`torch.nn.Module`]):
            A UNet model for denoising.
        scheduler ([`LBMScheduler`]):
            An `LBMScheduler` for the bridge flow-matching process.
    """

    model_cpu_offload_seq = "unet"

    def __init__(
        self,
        unet: LBMUNet,
        scheduler: LBMScheduler,
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
        num_inference_steps: int = 1,
        cfg_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        """
        Generate images using the LBM bridge flow-matching process.

        Args:
            source_image: The source/condition image(s) for the bridge.
                Can be a tensor of shape (B, C, H, W) in [-1, 1] range,
                or PIL Image(s).
            num_inference_steps: Number of Euler sampling steps (default: 1).
                LBM is designed for single-step inference but supports
                multi-step for improved quality.
            cfg_scale: Classifier-free guidance scale (default: 1.0, disables CFG).
            generator: Random number generator for reproducibility.
            output_type: Output format - "pil", "np", or "pt" (default: "pil").
            return_dict: Whether to return a dict with the output (default: True).
            callback: Callback function for progress updates.
            callback_steps: Frequency of callback calls.

        Returns:
            Images generated through the LBM bridge flow-matching process.
        """
        device = self.device
        dtype = self.dtype

        # Prepare source image
        x_source = self.prepare_inputs(source_image, device, dtype)
        batch_size = x_source.shape[0]

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Start from the source image
        sample = x_source.clone()

        # Condition for concat mode
        has_condition = hasattr(self.unet, 'condition_mode') and self.unet.condition_mode == 'concat'
        cond = x_source if has_condition else None
        use_cfg = has_condition and abs(float(cfg_scale) - 1.0) > 1e-6
        null_condition = torch.zeros_like(cond) if use_cfg else None
        nfe_per_step = 2 if use_cfg else 1
        nfe_count = 0

        # Sampling loop (Euler steps)
        progress_bar = tqdm(
            enumerate(self.scheduler.timesteps),
            total=len(self.scheduler.timesteps),
            desc="LBM Sampling",
        )

        for i, t in progress_bar:
            t_batch = t.to(device).repeat(batch_size) if isinstance(t, torch.Tensor) else torch.full(
                (batch_size,), t, device=device, dtype=torch.long,
            )

            # Model prediction
            if use_cfg:
                model_input = torch.cat([sample, sample], dim=0)
                timestep_input = torch.cat([t_batch, t_batch], dim=0)
                cond_input = torch.cat([cond, null_condition], dim=0)
                pred_batched = self.unet(model_input, timestep_input, cond=cond_input)
                pred_cond, pred_uncond = pred_batched.chunk(2, dim=0)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                cond_input = cond if cond is not None else None
                pred = self.unet(sample, t_batch, cond=cond_input)
            nfe_count += nfe_per_step

            # Euler step
            result = self.scheduler.step(pred, t, sample, return_dict=True)
            sample = result.prev_sample

            # Inject bridge noise for intermediate steps
            if i < len(self.scheduler.timesteps) - 1:
                next_t = self.scheduler.timesteps[i + 1]
                next_t_batch = next_t.to(device).repeat(batch_size)
                next_sigmas = self.scheduler.get_sigmas(
                    next_t_batch, n_dim=sample.ndim, device=device, dtype=dtype,
                )
                bridge_noise = self.scheduler.bridge_noise_sigma
                sample = sample + bridge_noise * (
                    next_sigmas * (1.0 - next_sigmas)
                ) ** 0.5 * torch.randn_like(sample)

            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, len(self.scheduler.timesteps), sample)

        # Post-process output
        images = sample.clamp(-1, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        # else: output_type == "pt", return tensor as-is

        if not return_dict:
            return (images, nfe_count)

        return LBMPipelineOutput(images=images, nfe=nfe_count)

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
