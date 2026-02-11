"""Latent-space inference pipeline for DDIB.

Wraps the Dual Diffusion Implicit Bridges process with a frozen VAE
so that both UNets operate in latent space while the pipeline accepts
and produces pixel-space images.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput

from src.schedulers.scheduling_ddib import DDIBScheduler
from src.models.unet_ddib import DDIBUNet


@dataclass
class DDIBLatentPipelineOutput(BaseOutput):
    """Output class for the DDIB latent pipeline.

    Attributes
    ----------
    images : list of PIL.Image or ndarray or Tensor
        Generated images in pixel space.
    latent : Tensor or None
        Shared latent representation (optional, for debugging).
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    latent: Optional[torch.Tensor] = None


class DDIBLatentPipeline(DiffusionPipeline):
    """DDIB pipeline that operates in VAE latent space.

    The pipeline encodes pixel-space source images into the latent space
    of a frozen VAE, runs the DDIB encode→decode process in that latent
    space, and decodes the result back to pixel space.

    Parameters
    ----------
    source_unet : DDIBUNet
        Diffusion model trained on the source domain (in latent space).
    target_unet : DDIBUNet
        Diffusion model trained on the target domain (in latent space).
    scheduler : DDIBScheduler
        Shared DDIB scheduler.
    vae : AutoencoderKL
        A frozen pre-trained VAE for encoding/decoding.
    """

    model_cpu_offload_seq = "vae->source_unet->target_unet"

    def __init__(
        self,
        source_unet: DDIBUNet,
        target_unet: DDIBUNet,
        scheduler: DDIBScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()
        self.register_modules(
            source_unet=source_unet,
            target_unet=target_unet,
            scheduler=scheduler,
            vae=vae,
        )

    @property
    def device(self) -> torch.device:
        """Returns the device of the pipeline."""
        return next(self.target_unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Returns the dtype of the pipeline."""
        return next(self.target_unet.parameters()).dtype

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        """Adapt single-channel images to 3-channel for VAE encoding."""
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Restore channel count after VAE decoding."""
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
        """Decode VAE latents back to pixel space."""
        scaled = latents / self.vae.config.scaling_factor
        return self.vae.decode(scaled).sample

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
        """Run DDIM reverse sampling to encode ``x_0`` into the latent ``x_T``.
        
        Args:
            model: The UNet model to use for prediction.
            x_0: Starting sample in latent space.
            timesteps: Timestep schedule (ascending order).
            clip_denoised: Whether to clip predicted x_0 to [-1, 1].
            
        Returns:
            The encoded latent representation.
        """
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
        """Run DDIM forward sampling to decode a latent ``x_T`` into ``x_0``.
        
        Args:
            model: The UNet model to use for prediction.
            noise: Starting noise in latent space.
            timesteps: Timestep schedule (ascending order).
            clip_denoised: Whether to clip predicted x_0 to [-1, 1].
            eta: DDIM eta parameter (0 = deterministic).
            
        Returns:
            The decoded sample in latent space.
        """
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
    # Input preparation helpers
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert PIL / numpy / tensor inputs to normalised ``[-1, 1]`` tensors.
        
        Args:
            image: Input image(s) in various formats.
            device: Target device for the tensor.
            dtype: Target dtype for the tensor.
            
        Returns:
            Normalized tensor in [-1, 1] range.
        """
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
    # Output conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        """Convert tensor images to PIL Image list.
        
        Args:
            images: Tensor in [-1, 1] range, shape (B, C, H, W).
            
        Returns:
            List of PIL Images.
        """
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in images]

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        """Convert tensor images to numpy array.
        
        Args:
            images: Tensor in [-1, 1] range, shape (B, C, H, W).
            
        Returns:
            Numpy array in [0, 1] range, shape (B, H, W, C).
        """
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).numpy()

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
        target_channels: Optional[int] = None,
    ):
        """Translate a source image via DDIB in VAE latent space.

        The pipeline performs the following steps:
        1. Encodes the source image to VAE latent space
        2. Runs DDIM reverse sampling with the source UNet to encode to shared noise
        3. Runs DDIM forward sampling with the target UNet to decode to target latent
        4. Decodes the target latent back to pixel space

        Args:
            source_image: Source image(s) in ``[-1, 1]`` or ``[0, 1]`` or PIL format.
            num_inference_steps: Number of DDIM steps for both encode and decode.
            clip_denoised: Clip predicted ``x_0`` to ``[-1, 1]`` during sampling.
            eta: DDIM eta parameter (0 = deterministic, 1 = fully stochastic).
            output_type: Output format: ``"pil"`` | ``"np"`` | ``"pt"``.
            return_dict: If ``True`` return a :class:`DDIBLatentPipelineOutput`.
            return_latent: If ``True`` include the shared latent in the output.
            target_channels: Number of channels for the output image. If not 
                provided, defaults to the number of channels in source_image.

        Returns:
            :class:`DDIBLatentPipelineOutput` or tuple of images.
        """
        device = self.device
        dtype = self.dtype

        # Prepare pixel inputs
        x_pixel = self.prepare_inputs(source_image, device, dtype)
        orig_channels = x_pixel.shape[1]

        # Encode to latent space
        z_source = self._encode(x_pixel)

        # Build timestep sequences
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # DDIM reverse: source latent → shared noise
        z_latent = self._ddim_reverse_sample_loop(
            self.source_unet, z_source, timesteps, clip_denoised=clip_denoised,
        )

        # DDIM forward: shared noise → target latent
        z_target = self._ddim_sample_loop(
            self.target_unet, z_latent, timesteps, clip_denoised=clip_denoised, eta=eta,
        )

        # Decode from latent to pixel space
        images = self._decode(z_target)
        images = self._restore_channels(images, target_channels or orig_channels)
        images = images.clamp(-1, 1)

        # Post-process output
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images,)

        return DDIBLatentPipelineOutput(
            images=images,
            latent=z_latent if return_latent else None,
        )
