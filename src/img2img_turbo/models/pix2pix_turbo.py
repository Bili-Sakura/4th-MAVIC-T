"""Pix2Pix-Turbo model built on ``diffusers`` components.

This module re-implements the core Pix2Pix-Turbo architecture from
``vendor/Img2Image-Turbo`` using native Hugging Face *diffusers* building
blocks: :class:`~diffusers.AutoencoderKL`, :class:`~diffusers.UNet2DConditionModel`,
and :class:`~diffusers.DDPMScheduler`.

The key architectural innovations from the vendor code are preserved:

* **Enhanced VAE decoder** with skip connections from the encoder to the
  decoder via four learned 1×1 convolutions.
* **LoRA fine-tuning** of both UNet and VAE for parameter-efficient training.
* **Single-step denoising** using a DDPM scheduler at ``t = 999``.

The wrapper class :class:`Pix2PixTurbo` exposes a simple
``forward(source, prompt_embeds)`` that runs the full encode → denoise →
decode pipeline, producing an output image in ``[-1, 1]``.
"""

from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

from ..utils.vae_utils import vae_encoder_fwd, vae_decoder_fwd


class Pix2PixTurbo(nn.Module):
    """Pix2Pix-Turbo model using native HuggingFace diffusers components.

    This wraps a Stable-Diffusion-Turbo backbone with LoRA adapters and
    skip-connection-enhanced VAE decoder for one-step paired image translation.

    Parameters
    ----------
    pretrained_path : str or None
        Path to a previously saved ``.pkl`` checkpoint.  When ``None`` the
        model is initialised with fresh LoRA adapters on top of SD-Turbo.
    pretrained_model_name_or_path : str
        HuggingFace model identifier for the base SD-Turbo weights.
    lora_rank_unet : int
        LoRA rank for the UNet adapter layers.
    lora_rank_vae : int
        LoRA rank for the VAE adapter layers.
    """

    # Default LoRA target modules (same as vendor code)
    TARGET_MODULES_UNET: List[str] = [
        "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
        "conv_shortcut", "conv_out", "proj_in", "proj_out",
        "ff.net.2", "ff.net.0.proj",
    ]
    TARGET_MODULES_VAE: List[str] = [
        "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
        "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
        "to_k", "to_q", "to_v", "to_out.0",
    ]

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        pretrained_model_name_or_path: str = "stabilityai/sd-turbo",
        lora_rank_unet: int = 8,
        lora_rank_vae: int = 4,
    ) -> None:
        super().__init__()

        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_unet = list(self.TARGET_MODULES_UNET)
        self.target_modules_vae = list(self.TARGET_MODULES_VAE)

        # ---- Text encoder & tokenizer (frozen) ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder",
        )
        self.text_encoder.requires_grad_(False)

        # ---- Scheduler (single-step DDPM) ----
        self.sched = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler",
        )

        # ---- VAE with skip connections ----
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae",
        )
        vae.encoder.forward = vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        vae.decoder.ignore_skip = False
        vae.decoder.gamma = 1

        # ---- UNet ----
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet",
        )

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"],
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"],
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            self.lora_rank_unet = sd["rank_unet"]
            self.lora_rank_vae = sd["rank_vae"]
            self.target_modules_unet = sd["unet_lora_target_modules"]
            self.target_modules_vae = sd["vae_lora_target_modules"]
        else:
            # Fresh initialisation with LoRA adapters
            nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            vae_lora_config = LoraConfig(
                r=lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_vae,
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            unet_lora_config = LoraConfig(
                r=lora_rank_unet,
                init_lora_weights="gaussian",
                target_modules=self.target_modules_unet,
            )
            unet.add_adapter(unet_lora_config)

        self.unet = unet
        self.vae = vae
        self.timesteps = torch.tensor([999]).long()

    def set_eval(self) -> None:
        """Set the model to evaluation mode and freeze all parameters."""
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self) -> None:
        """Set the model to training mode, unfreezing only LoRA / skip layers."""
        self.unet.train()
        self.vae.train()
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def get_trainable_params(self) -> list:
        """Return the list of trainable parameter tensors."""
        layers = []
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                layers.append(p)
        layers += list(self.unet.conv_in.parameters())
        for n, p in self.vae.named_parameters():
            if "lora" in n and "vae_skip" in n:
                layers.append(p)
        layers += list(self.vae.decoder.skip_conv_1.parameters())
        layers += list(self.vae.decoder.skip_conv_2.parameters())
        layers += list(self.vae.decoder.skip_conv_3.parameters())
        layers += list(self.vae.decoder.skip_conv_4.parameters())
        return layers

    def encode_prompt(self, prompt: str, device: torch.device) -> torch.Tensor:
        """Tokenise and encode a text prompt, returning encoder hidden states."""
        tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        return self.text_encoder(tokens)[0]

    def forward(
        self,
        source: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full encode → denoise → decode pipeline.

        Parameters
        ----------
        source : Tensor (B, 3, H, W)
            Source/conditioning image in ``[-1, 1]``.
        prompt_embeds : Tensor (B, seq_len, hidden_dim)
            Pre-computed text encoder hidden states.

        Returns
        -------
        Tensor (B, 3, H, W)
            Generated image clamped to ``[-1, 1]``.
        """
        device = source.device
        timesteps = self.timesteps.to(device)

        # Encode source image to latent space
        encoded = self.vae.encode(source).latent_dist.sample() * self.vae.config.scaling_factor

        # Single-step UNet denoising
        model_pred = self.unet(encoded, timesteps, encoder_hidden_states=prompt_embeds).sample

        # Scheduler step
        self.sched.set_timesteps(1, device=device)
        self.sched.alphas_cumprod = self.sched.alphas_cumprod.to(device)
        x_denoised = self.sched.step(model_pred, timesteps, encoded, return_dict=True).prev_sample
        x_denoised = x_denoised.to(model_pred.dtype)

        # Decode with skip connections
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        return output.clamp(-1, 1)

    def save_model(self, path: str) -> None:
        """Save LoRA + skip-connection weights to a checkpoint file."""
        sd = {
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules": self.target_modules_vae,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {
                k: v for k, v in self.unet.state_dict().items()
                if "lora" in k or "conv_in" in k
            },
            "state_dict_vae": {
                k: v for k, v in self.vae.state_dict().items()
                if "lora" in k or "skip" in k
            },
        }
        torch.save(sd, path)
