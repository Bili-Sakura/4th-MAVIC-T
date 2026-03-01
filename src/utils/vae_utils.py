# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Custom VAE forward functions for skip-connection-enhanced Img2Image-Turbo.

These functions replace the default ``AutoencoderKL`` encoder/decoder forwards
so that intermediate encoder activations can be passed to the decoder via
learned 1×1 skip convolutions – the key architectural innovation from
`Img2Image-Turbo <https://github.com/GaParmar/img2img-turbo>`_.
"""

from __future__ import annotations

import torch


def vae_encoder_fwd(self, sample: torch.Tensor) -> torch.Tensor:
    """Custom VAE encoder forward that stores intermediate down-block activations.

    After running, ``self.current_down_blocks`` holds a list of tensors
    (one per encoder down-block) that the decoder can consume as skip inputs.
    """
    sample = self.conv_in(sample)
    l_blocks = []
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def vae_decoder_fwd(self, sample: torch.Tensor, latent_embeds=None) -> torch.Tensor:
    """Custom VAE decoder forward that accepts skip connections from the encoder.

    When ``self.ignore_skip`` is ``False`` the decoder reads
    ``self.incoming_skip_acts`` (set by the caller) and blends them in via
    ``skip_conv_{1..4}`` convolutions scaled by ``self.gamma``.
    """
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample
