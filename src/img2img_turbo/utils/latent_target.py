"""Latent-space target encoder for the RGB2IR ablation study.

Loads a pre-trained VAE (encoder only) from a local checkpoint and uses it
to encode target images into latent space.  A latent-space L2 loss between
the model's internal latent and the pre-trained VAE's latent encourages the
Pix2Pix-Turbo pipeline to produce outputs that are consistent in latent
space with targets encoded by the reference VAE.

The pre-trained VAE checkpoints are expected at
``models/BiliSakura/VAEs`` (a HuggingFace ``diffusers`` style directory).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class LatentTargetEncoder(nn.Module):
    """Frozen VAE encoder used to produce latent targets.

    Parameters
    ----------
    vae_path : str
        Local path (or HuggingFace hub id) to a ``diffusers``-style VAE
        checkpoint directory (e.g. ``./models/BiliSakura/VAEs``).
    """

    def __init__(self, vae_path: str) -> None:
        super().__init__()
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(vae_path)
        self.vae.requires_grad_(False)
        self.vae.eval()
        # Store scaling factor for consistency
        self.scaling_factor: float = self.vae.config.scaling_factor

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent means (no sampling noise).

        Parameters
        ----------
        images : Tensor (B, 3, H, W)
            Images in ``[-1, 1]``.

        Returns
        -------
        Tensor (B, C_latent, H_lat, W_lat)
            Scaled latent representation (deterministic mean).
        """
        posterior = self.vae.encode(images).latent_dist
        return posterior.mean * self.scaling_factor
