# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Pipeline wrappers for Img2Image-Turbo models.

Follows the ``diffusers``-style pipeline pattern where the pipeline owns the
full inference workflow: text encoding → encode → denoise → decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from PIL import Image
from torchvision import transforms


@dataclass
class TurboPipelineOutput:
    """Output of a Turbo pipeline call.

    Attributes
    ----------
    images : list[PIL.Image.Image]
        Generated images.
    """
    images: List[Image.Image]


class Pix2PixTurboPipeline:
    """Inference pipeline for Pix2Pix-Turbo.

    Wraps the model to provide a clean ``__call__`` API following diffusers
    pipeline conventions.

    Parameters
    ----------
    model : Pix2PixTurbo
        The Pix2Pix-Turbo model.
    """

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        prompt: Optional[str] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
    ) -> TurboPipelineOutput:
        """Run paired image-to-image translation.

        Parameters
        ----------
        image : Tensor (B, 3, H, W)
            Source image in ``[-1, 1]``.
        prompt : str or None
            Text prompt (encoded on-the-fly).
        prompt_embeds : Tensor or None
            Pre-computed prompt embeddings.

        Returns
        -------
        TurboPipelineOutput
            Output containing PIL images.
        """
        assert (prompt is None) != (prompt_embeds is None), \
            "Provide either prompt or prompt_embeds, not both."

        device = image.device
        if prompt is not None:
            prompt_embeds = self.model.encode_prompt(prompt, device)

        prompt_embeds = prompt_embeds.expand(image.shape[0], -1, -1)
        output = self.model(image, prompt_embeds)

        # Convert to PIL
        images_pil = []
        for i in range(output.shape[0]):
            img_pil = transforms.ToPILImage()(output[i].cpu() * 0.5 + 0.5)
            images_pil.append(img_pil)

        return TurboPipelineOutput(images=images_pil)


class CycleGANTurboPipeline:
    """Inference pipeline for CycleGAN-Turbo.

    Wraps the model to provide a clean ``__call__`` API following diffusers
    pipeline conventions.

    Parameters
    ----------
    model : CycleGANTurbo
        The CycleGAN-Turbo model.
    """

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        direction: str = "a2b",
        caption: Optional[str] = None,
        caption_emb: Optional[torch.Tensor] = None,
    ) -> TurboPipelineOutput:
        """Run unpaired image-to-image translation.

        Parameters
        ----------
        image : Tensor (B, 3, H, W)
            Input image in ``[-1, 1]``.
        direction : str
            ``"a2b"`` or ``"b2a"``.
        caption : str or None
            Text prompt.
        caption_emb : Tensor or None
            Pre-computed text embeddings.

        Returns
        -------
        TurboPipelineOutput
            Output containing PIL images.
        """
        output = self.model(image, direction=direction, caption=caption, caption_emb=caption_emb)

        images_pil = []
        for i in range(output.shape[0]):
            img_pil = transforms.ToPILImage()(output[i].cpu() * 0.5 + 0.5)
            images_pil.append(img_pil)

        return TurboPipelineOutput(images=images_pil)
