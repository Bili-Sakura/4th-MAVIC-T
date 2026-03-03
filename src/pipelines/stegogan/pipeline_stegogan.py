# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""StegoGAN single-pass inference pipeline.

Provides :class:`StegoGANPipeline` for running inference with a trained
StegoGAN Generator A, following the diffusers-style pipeline pattern.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput

from src.models.stegogan_model import StegoGANGeneratorA


@dataclass
class StegoGANPipelineOutput(BaseOutput):
    """Output class for StegoGAN pipeline.

    Attributes
    ----------
    images : list of PIL.Image.Image, np.ndarray, or torch.Tensor
        Generated images after translation.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class StegoGANPipeline(DiffusionPipeline):
    """Single-pass inference pipeline for StegoGAN image translation.

    At inference time, only Generator A (source→target) is needed.
    This pipeline wraps the generator with a consistent API for loading,
    preprocessing, and postprocessing.

    Inherits from :class:`~diffusers.DiffusionPipeline` so that checkpoints
    can be loaded via ``from_pretrained`` following the HuggingFace
    *diffusers* convention.

    Parameters
    ----------
    generator : torch.nn.Module
        A trained :class:`~src.models.StegoGANGeneratorA`.
    """

    def __init__(self, generator: StegoGANGeneratorA) -> None:
        super().__init__()
        self.register_modules(generator=generator)

    @property
    def device(self) -> torch.device:
        return next(self.generator.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.generator.parameters()).dtype

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare input images for the pipeline."""
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img_array = np.array(img, dtype=np.float32)
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
                if img_array.ndim == 2:
                    img_array = img_array[:, :, np.newaxis]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                images.append(img_tensor)
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.min() >= 0 and image.max() <= 1.0:
            image = image * 2 - 1
        elif image.max() > 1.0:
            image = image / 255.0 * 2 - 1

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[StegoGANPipelineOutput, tuple]:
        """Generate translated images via a single forward pass through G_A.

        Parameters
        ----------
        source_image : torch.Tensor or PIL.Image.Image or list of PIL.Image.Image
            Source images for translation.
        output_type : str
            ``"pil"``, ``"np"``, or ``"pt"``.
        return_dict : bool
            If ``True``, return a :class:`StegoGANPipelineOutput`.
        """
        device = self.device
        dtype = self.dtype

        x = self.prepare_inputs(source_image, device, dtype)

        # Single forward pass — no extra feature at inference
        fake = self.generator(x)
        images = fake.clamp(-1, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return (images,)

        return StegoGANPipelineOutput(images=images)

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        pil_images = []
        for img in images:
            if img.shape[2] == 1:
                pil_images.append(Image.fromarray(img.squeeze(2), mode="L"))
            else:
                pil_images.append(Image.fromarray(img))
        return pil_images

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images
