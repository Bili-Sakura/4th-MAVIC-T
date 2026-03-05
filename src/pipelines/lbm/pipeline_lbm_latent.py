# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2025 Chadebec et al. and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Latent-space inference pipeline for Latent Bridge Matching (LBM).

"""Latent-space inference pipeline for LBM.

Wraps the LBM bridge flow-matching process with a frozen VAE so that
the UNet operates entirely in latent space while the pipeline accepts
and produces pixel-space images.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput

from src.schedulers.scheduling_lbm import LBMScheduler
from src.models import UNet2DWrapper as LBMUNet
from src.utils.multidiffusion import (
    DEFAULT_LATENT_WINDOW_SIZE,
    get_views,
)


@dataclass
class LBMLatentPipelineOutput(BaseOutput):
    """Output class for the LBM latent pipeline.

    Attributes
    ----------
    images : list of PIL.Image or ndarray or Tensor
        Generated images in pixel space.
    nfe : int
        Number of function evaluations during sampling.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class LBMLatentPipeline(DiffusionPipeline):
    """LBM pipeline that operates in VAE latent space.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation
    for the generic methods implemented for all pipelines (downloading, saving, running
    on a particular device, etc.).

    The pipeline encodes pixel-space source images into the latent space of a frozen VAE,
    runs the LBM bridge flow-matching process in that latent space, and decodes the result
    back to pixel space.

    Parameters
    ----------
    unet : LBMUNet
        A UNet trained on VAE latents for LBM.
    scheduler : LBMScheduler
        The LBM scheduler.
    vae : AutoencoderKL
        A frozen pre-trained VAE for encoding/decoding.
    """

    model_cpu_offload_seq = "vae->unet"

    def __init__(
        self,
        unet: LBMUNet,
        scheduler: LBMScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

    # ------------------------------------------------------------------
    # VAE encoding/decoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        """Adapt single-channel images to 3-channel for VAE encoding."""
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Restore original channel count after VAE decoding."""
        if target_channels == 1 and images.shape[1] == 3:
            return images.mean(dim=1, keepdim=True)
        return images

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space images to VAE latent space."""
        adapted = self._adapt_channels(images)
        posterior = self.vae.encode(adapted).latent_dist
        return posterior.mean * self.vae.config.scaling_factor

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to pixel-space images."""
        scaled = latents / self.vae.config.scaling_factor
        return self.vae.decode(scaled).sample

    def _unet_tiled(
        self,
        zt: torch.Tensor,
        t_batch: torch.Tensor,
        cond: Optional[torch.Tensor],
        views: List[Tuple[int, int, int, int]],
        view_batch_size: int = 1,
    ) -> torch.Tensor:
        """MultiDiffusion: run UNet on overlapping crops and merge by averaging."""
        value = torch.zeros_like(zt)
        count = torch.zeros_like(zt)
        batch_size = zt.shape[0]
        view_batches = [
            views[i : i + view_batch_size]
            for i in range(0, len(views), view_batch_size)
        ]
        for batch_view in view_batches:
            vb_size = len(batch_view)
            crops_zt = torch.cat(
                [zt[:, :, h_start:h_end, w_start:w_end] for h_start, h_end, w_start, w_end in batch_view],
                dim=0,
            )
            crops_cond = (
                torch.cat(
                    [cond[:, :, h_start:h_end, w_start:w_end] for h_start, h_end, w_start, w_end in batch_view],
                    dim=0,
                )
                if cond is not None
                else None
            )
            t_crops = t_batch.repeat_interleave(vb_size, dim=0)
            pred_crops = self.unet(
                crops_zt.to(device=self._execution_device, dtype=next(self.unet.parameters()).dtype),
                t_crops,
                cond=(
                    crops_cond.to(device=self._execution_device, dtype=next(self.unet.parameters()).dtype)
                    if crops_cond is not None
                    else None
                ),
            )
            pred_crops = pred_crops.to(device=zt.device, dtype=zt.dtype)
            for b in range(batch_size):
                for k, (h_start, h_end, w_start, w_end) in enumerate(batch_view):
                    idx = b * vb_size + k
                    value[b : b + 1, :, h_start:h_end, w_start:w_end] += pred_crops[idx : idx + 1]
                    count[b : b + 1, :, h_start:h_end, w_start:w_end] += 1
        return torch.where(count > 0, value / count, value)

    def _resize_to_output_size(
        self,
        image: torch.Tensor,
        output_size: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        if output_size is None:
            return image
        h, w = output_size
        if image.shape[-2] == h and image.shape[-1] == w:
            return image
        return torch.nn.functional.interpolate(
            image, size=(h, w), mode="bilinear", align_corners=False
        )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Prepare input images for the pipeline."""
        if device is None:
            device = self._execution_device
        if dtype is None:
            dtype = next(self.unet.parameters()).dtype

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

        if image.max() > 1.0:
            image = image / 255.0

        if image.min() >= 0:
            image = image * 2 - 1

        return image.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Output conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        """Convert tensor in [-1, 1] to PIL images."""
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

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        """Convert tensor in [-1, 1] to numpy array."""
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

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
        target_channels: Optional[int] = None,
        output_size: Optional[Tuple[int, int]] = None,
        view_batch_size: int = 1,
        latent_window_size: int = DEFAULT_LATENT_WINDOW_SIZE,
    ):
        """Translate a source image via LBM in VAE latent space.

        MultiDiffusion tiling is used when latent size > trained size.

        Args:
            source_image: Source/condition image(s) as tensor (B, C, H, W) in [-1, 1], or PIL Image(s).
            num_inference_steps: Number of Euler sampling steps (default: 1).
            cfg_scale: Classifier-free guidance scale (default: 1.0, disables CFG).
            generator: Random number generator for reproducibility.
            output_type: Output format - "pil", "np", or "pt" (default: "pil").
            return_dict: Whether to return a dict with the output (default: True).
            callback: Callback function for progress updates.
            callback_steps: Frequency of callback calls.
            target_channels: Number of output channels (None = same as source).
            output_size: Target pixel resolution (H, W). Enables MultiDiffusion tiling.
            view_batch_size: Number of tiles to process in parallel.
            latent_window_size: MultiDiffusion tile size in latent space.
        """
        device = self._execution_device
        dtype = next(self.unet.parameters()).dtype

        # Prepare pixel inputs
        x_pixel = self.prepare_inputs(source_image, device, dtype)
        x_pixel = self._resize_to_output_size(x_pixel, output_size)
        orig_channels = x_pixel.shape[1]

        # Encode to latent space
        z_source = self._encode(x_pixel)
        batch_size = z_source.shape[0]
        _, _, lh, lw = z_source.shape
        views = None
        if output_size is not None:
            views = get_views(lh, lw, window_size=latent_window_size)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # Start from the encoded source
        sample = z_source.clone()

        # Condition for concat mode
        has_condition = hasattr(self.unet, "condition_mode") and self.unet.condition_mode == "concat"
        cond = z_source if has_condition else None
        use_cfg = has_condition and abs(float(cfg_scale) - 1.0) > 1e-6
        null_condition = torch.zeros_like(cond) if use_cfg else None
        nfe_per_step = 2 if use_cfg else 1
        nfe_count = 0

        # Sampling loop (Euler steps)
        progress_bar = tqdm(
            enumerate(self.scheduler.timesteps),
            total=len(self.scheduler.timesteps),
            desc="LBM Latent Sampling",
        )

        for i, t in progress_bar:
            t_batch = t.to(device).repeat(batch_size) if isinstance(t, torch.Tensor) else torch.full(
                (batch_size,), t, device=device, dtype=torch.long,
            )

            # Model prediction (with optional MultiDiffusion tiling)
            if use_cfg:
                model_input = torch.cat([sample, sample], dim=0)
                timestep_input = torch.cat([t_batch, t_batch], dim=0)
                cond_input = torch.cat([cond, null_condition], dim=0)
                if views is not None:
                    pred = self._unet_tiled(model_input, timestep_input, cond_input, views, view_batch_size)
                else:
                    pred = self.unet(model_input, timestep_input, cond=cond_input)
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                if views is not None:
                    pred = self._unet_tiled(sample, t_batch, cond, views, view_batch_size)
                else:
                    pred = self.unet(sample, t_batch, cond=cond)
            pred = pred.to(device=sample.device, dtype=sample.dtype)
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

            if callback is not None and i % callback_steps == 0:
                callback(i, len(self.scheduler.timesteps), sample)

        # Decode from latent to pixel space
        images = self._decode(sample)
        images = self._restore_channels(images, target_channels or orig_channels)
        images = images.clamp(-1, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images, nfe_count)

        return LBMLatentPipelineOutput(images=images, nfe=nfe_count)
