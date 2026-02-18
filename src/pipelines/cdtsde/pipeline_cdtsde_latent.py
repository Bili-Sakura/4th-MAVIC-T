"""Latent-space inference pipeline for CDTSDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from src.models.unet_cdtsde import CDTSDEUNet
from src.schedulers.scheduling_cdtsde import CDTSDEScheduler
from src.utils.multidiffusion import (
    DEFAULT_LATENT_WINDOW_SIZE,
    get_views,
)


@dataclass
class CDTSDELatentPipelineOutput(BaseOutput):
    """Output class for CDTSDE latent pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class CDTSDELatentPipeline(DiffusionPipeline):
    """CDTSDE pipeline operating in VAE latent space."""

    model_cpu_offload_seq = "vae->unet"

    def __init__(
        self,
        unet: CDTSDEUNet,
        scheduler: CDTSDEScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None) is not None
            else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

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

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        if target_channels == 1 and images.shape[1] == 3:
            return images.mean(dim=1, keepdim=True)
        return images

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        adapted = self._adapt_channels(images)
        posterior = self.vae.encode(adapted).latent_dist
        return posterior.mean * self.vae.config.scaling_factor

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        scaled = latents / self.vae.config.scaling_factor
        return self.vae.decode(scaled).sample

    def _unet_tiled(
        self,
        z: torch.Tensor,
        t_batch: torch.Tensor,
        z_T: torch.Tensor,
        views: List[Tuple[int, int, int, int]],
        view_batch_size: int = 1,
    ) -> torch.Tensor:
        """MultiDiffusion: run UNet on overlapping crops and merge pred_noise by averaging."""
        value = torch.zeros_like(z)
        count = torch.zeros_like(z)
        batch_size = z.shape[0]
        view_batches = [
            views[i : i + view_batch_size]
            for i in range(0, len(views), view_batch_size)
        ]
        for batch_view in view_batches:
            vb_size = len(batch_view)
            crops_z = torch.cat(
                [z[:, :, h_start:h_end, w_start:w_end] for h_start, h_end, w_start, w_end in batch_view],
                dim=0,
            )
            crops_z_T = torch.cat(
                [z_T[:, :, h_start:h_end, w_start:w_end] for h_start, h_end, w_start, w_end in batch_view],
                dim=0,
            )
            t_crops = t_batch.repeat_interleave(vb_size, dim=0)
            pred_noise_crops = self.unet(crops_z, t_crops, xT=crops_z_T)
            for b in range(batch_size):
                for k, (h_start, h_end, w_start, w_end) in enumerate(batch_view):
                    idx = b * vb_size + k
                    value[b : b + 1, :, h_start:h_end, w_start:w_end] += pred_noise_crops[idx : idx + 1]
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
        target_channels: Optional[int] = None,
        output_size: Optional[Tuple[int, int]] = None,
        view_batch_size: int = 1,
        latent_window_size: int = DEFAULT_LATENT_WINDOW_SIZE,
    ):
        device = self.device
        dtype = self.dtype

        x_pixel = self.prepare_inputs(source_image, device=device, dtype=dtype)
        x_pixel = self._resize_to_output_size(x_pixel, output_size)
        orig_channels = x_pixel.shape[1]
        z_T = self._encode(x_pixel)
        batch_size = z_T.shape[0]
        _, _, lh, lw = z_T.shape
        views = None
        # Only use MultiDiffusion tiling when output_size is explicitly set; otherwise run full-res
        if output_size is not None:
            views = get_views(lh, lw, window_size=latent_window_size)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if self.scheduler.timesteps is None:
            raise ValueError("Scheduler timesteps not initialized.")

        sqrt_alpha_last = self.scheduler.sqrt_alphas_cumprod[-1].to(device=device, dtype=dtype)
        sigma_last = self.scheduler.sigmas[-1].to(device=device, dtype=dtype)
        noise = randn_tensor(
            z_T.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        z = sqrt_alpha_last * z_T + sigma_last * noise

        nfe = 0
        total_steps = len(self.scheduler.timesteps) - 1
        progress = tqdm(range(total_steps), desc="CDTSDE Latent Sampling")

        for i in progress:
            j = total_steps - i
            t_model = self.scheduler.timesteps[j]
            t_batch = torch.full((batch_size,), t_model, device=device, dtype=torch.long)

            if views is not None:
                pred_noise = self._unet_tiled(z, t_batch, z_T, views, view_batch_size=view_batch_size)
            else:
                pred_noise = self.unet(z, t_batch, xT=z_T)
            nfe += 1

            idx_batch = torch.full((batch_size,), j, device=device, dtype=torch.long)
            pred_z0 = self.scheduler.predict_start_from_noise(
                sample=z,
                timesteps=idx_batch,
                noise=pred_noise,
                use_inference_schedule=True,
            )

            if apply_domain_shift:
                lam_linear = self.scheduler.etas[j].to(device=device, dtype=dtype)
                lam_batch = torch.full((batch_size,), lam_linear, device=device, dtype=dtype)
                lam_hat = self.unet.predict_lambda(lam_batch, pred_z0.shape)
                pred_z0 = lam_hat * z_T + (1.0 - lam_hat) * pred_z0

            step_out = self.scheduler.step(
                pred_original_sample=pred_z0,
                step_index=j,
                sample=z,
                reference_sample=z_T,
                generator=generator if isinstance(generator, torch.Generator) else None,
                stochastic=stochastic,
                return_dict=True,
            )
            z = step_out.prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, total_steps, z)

        images = self._decode(z)
        images = self._restore_channels(images, target_channels or orig_channels)
        images = images.clamp(-1.0, 1.0)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        elif output_type != "pt":
            raise ValueError(f"Unsupported output_type: {output_type}")

        if not return_dict:
            return images, nfe
        return CDTSDELatentPipelineOutput(images=images, nfe=nfe)

