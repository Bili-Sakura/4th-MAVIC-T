"""Latent-space inference pipeline for I2SB.

Wraps the I2SB Schrödinger Bridge process with a frozen VAE so that
the UNet operates entirely in latent space while the pipeline accepts
and produces pixel-space images.

This pipeline follows the classic latent modeling pattern established
in diffusers pipelines like StableDiffusionPipeline, where pixel-space
images are encoded into VAE latents, processed in latent space, and
then decoded back to pixel space.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput

from src.schedulers.scheduling_i2sb import I2SBScheduler
from src.models.unet_i2sb import I2SBUNet


@dataclass
class I2SBLatentPipelineOutput(BaseOutput):
    """Output class for the I2SB latent pipeline.

    Attributes
    ----------
    images : list of PIL.Image or ndarray or Tensor
        Generated images in pixel space.
    nfe : int
        Number of function evaluations during sampling.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class I2SBLatentPipeline(DiffusionPipeline):
    """I2SB pipeline that operates in VAE latent space.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation
    for the generic methods implemented for all pipelines (downloading, saving, running
    on a particular device, etc.).

    The pipeline encodes pixel-space source images into the latent space of a frozen VAE,
    runs the I2SB Schrödinger Bridge process in that latent space, and decodes the result
    back to pixel space. This allows the UNet to operate entirely in latent space while
    maintaining a pixel-space API.

    Parameters
    ----------
    unet : I2SBUNet
        An I2SB UNet trained on VAE latents.
    scheduler : I2SBScheduler
        The I2SB scheduler.
    vae : AutoencoderKL
        A frozen pre-trained VAE for encoding/decoding.
    """

    model_cpu_offload_seq = "vae->unet"

    def __init__(
        self,
        unet: I2SBUNet,
        scheduler: I2SBScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

    # ------------------------------------------------------------------
    # VAE encoding/decoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        """Adapt single-channel images to 3-channel for VAE encoding.

        Args:
            images: Input tensor of shape (B, C, H, W).

        Returns:
            Tensor with 3 channels if input was 1 channel, otherwise unchanged.
        """
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Restore original channel count after VAE decoding.

        Args:
            images: Decoded tensor of shape (B, C, H, W).
            target_channels: Original number of channels before encoding.

        Returns:
            Tensor with restored channel count.
        """
        if target_channels == 1 and images.shape[1] == 3:
            return images.mean(dim=1, keepdim=True)
        return images

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space images to VAE latent space.

        Args:
            images: Pixel-space images in [-1, 1] range, shape (B, C, H, W).

        Returns:
            Latent representations scaled by VAE scaling factor.
        """
        adapted = self._adapt_channels(images)
        posterior = self.vae.encode(adapted).latent_dist
        return posterior.mean * self.vae.config.scaling_factor

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to pixel-space images.

        Args:
            latents: Latent representations scaled by VAE scaling factor.

        Returns:
            Pixel-space images in [-1, 1] range.
        """
        scaled = latents / self.vae.config.scaling_factor
        return self.vae.decode(scaled).sample

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Prepare input images for the pipeline.

        Converts PIL images or numpy arrays to normalized tensors in [-1, 1] range.

        Args:
            image: Input image(s) as PIL Image, list of PIL Images, numpy array, or tensor.
            device: Target device. If None, uses pipeline execution device.
            dtype: Target dtype. If None, uses pipeline dtype.

        Returns:
            Normalized tensor in [-1, 1] range, shape (B, C, H, W).
        """
        if device is None:
            device = self._execution_device
        if dtype is None:
            dtype = next(self.unet.parameters()).dtype

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

    # ------------------------------------------------------------------
    # Output conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        """Convert tensor in [-1, 1] to PIL images.

        Args:
            images: Tensor of shape (B, C, H, W) in [-1, 1] range.

        Returns:
            List of PIL Images.
        """
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

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        """Convert tensor in [-1, 1] to numpy array.

        Args:
            images: Tensor of shape (B, C, H, W) in [-1, 1] range.

        Returns:
            Numpy array of shape (B, H, W, C) in [0, 1] range.
        """
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
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
        nfe: int = 100,
        ot_ode: bool = False,
        clip_denoise: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        """Translate a source image via I2SB in VAE latent space.

        Args:
            source_image: The source/condition image(s) for the bridge.
                Can be a tensor of shape (B, C, H, W) in [-1, 1] range,
                or PIL Image(s).
            nfe: Number of function evaluations / sampling steps (default: 100).
            ot_ode: If True, use deterministic OT-ODE path (default: False).
            clip_denoise: If True, clamp predicted x0 to [-1, 1] (default: False).
                Note: This is ignored in latent space as clipping is handled by VAE.
            generator: Random number generator for reproducibility.
            output_type: Output format - "pil", "np", or "pt" (default: "pil").
            return_dict: Whether to return a dict with the output (default: True).
            callback: Callback function for progress updates.
            callback_steps: Frequency of callback calls.

        Returns:
            Images generated through the Schrödinger bridge diffusion process.
        """
        device = self._execution_device
        dtype = next(self.unet.parameters()).dtype

        # Prepare pixel inputs
        x_pixel = self.prepare_inputs(source_image, device, dtype)
        orig_channels = x_pixel.shape[1]

        # Encode to latent space
        z1 = self._encode(x_pixel)
        batch_size = z1.shape[0]

        # Set timesteps
        self.scheduler.set_timesteps(nfe, device=device)
        steps = self.scheduler.timesteps

        zt = z1.clone()

        # Build noise levels for timestep embedding
        interval = self.scheduler.config.interval
        t0_val = self.scheduler.config.t0
        T_val = self.scheduler.config.T
        noise_levels = torch.linspace(t0_val, T_val, interval, device=device, dtype=dtype)

        has_condition = hasattr(self.unet, "condition_mode") and self.unet.condition_mode == "concat"
        cond = z1 if has_condition else None

        # Backward sampling loop
        num_steps = len(steps) - 1
        progress_bar = tqdm(range(num_steps), desc="I2SB Latent Sampling")
        nfe_count = 0

        for i in progress_bar:
            step = steps[num_steps - i]
            prev_step = steps[num_steps - i - 1]

            step_int = step.item() if isinstance(step, torch.Tensor) else step
            t_emb = noise_levels[step_int] * interval
            t_batch = torch.full((batch_size,), t_emb, device=device, dtype=dtype)

            pred = self.unet(zt, t_batch, cond=cond)
            nfe_count += 1

            # No clip_denoise for latent space
            pred_x0 = self.scheduler.compute_pred_x0(step_int, zt, pred, clip_denoise=False)
            zt = self.scheduler.p_posterior(prev_step, step, zt, pred_x0, ot_ode=ot_ode)

            if callback is not None and i % callback_steps == 0:
                callback(i, num_steps, zt)

        # Decode from latent to pixel space
        images = self._decode(zt)
        images = self._restore_channels(images, orig_channels)
        images = images.clamp(-1, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        # else: output_type == "pt", return tensor as-is

        if not return_dict:
            return (images, nfe_count)

        return I2SBLatentPipelineOutput(images=images, nfe=nfe_count)
