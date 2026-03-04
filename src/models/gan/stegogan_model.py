# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""StegoGAN model components built on native PyTorch modules.

This module provides the networks required by the StegoGAN (Steganographic
GAN) method for non-bijective image-to-image translation:

* :class:`StegoGANGeneratorA` – ResNet-based generator (A→B) that can
  optionally accept a latent feature vector to embed steganographic
  information.  Based on ``ResnetMaskV1Generator`` in the original StegoGAN.
* :class:`StegoGANGeneratorB` – ResNet-based generator (B→A) that additionally
  outputs a latent feature vector and a latent mask for detecting semantic
  mismatches.  Based on ``ResnetMaskV3Generator`` in the original StegoGAN.
* :class:`StegoGANDiscriminator` – N-layer PatchGAN discriminator.

Loss module:

* :class:`StegoGANLoss` – Configurable GAN objective (LSGAN / Vanilla /
  WGAN-GP), matching the ``GANLoss`` class from CycleGAN / StegoGAN.

Reference: Wu, Sidi, et al. "StegoGAN: Leveraging Steganography for
Non-Bijective Image-to-Image Translation." CVPR 2024.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get_norm_layer(norm_type: str = "instance"):
    """Return a normalisation layer factory."""
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return lambda x: nn.Identity()
    raise NotImplementedError(f"Norm layer [{norm_type}] is not supported")


def _init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialise network weights."""

    def _init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"Init method [{init_type}] not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(_init_func)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ResnetBlock(nn.Module):
    """Single ResNet block with skip connection."""

    def __init__(self, dim: int, norm_layer, use_dropout: bool, use_bias: bool) -> None:
        super().__init__()
        layers: list = []
        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            layers += [nn.Dropout(0.5)]
        layers += [nn.ReflectionPad2d(1)]
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class _NetMatchability(nn.Module):
    """Matchability network that produces a sigmoid mask from latent features.

    Used inside :class:`StegoGANGeneratorB` to detect semantic mismatches.
    """

    def __init__(self, input_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(input_dim, eps=1e-05)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(input_dim, eps=1e-05)
        self.conv3 = nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm1(self.conv1(feat)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.sigmoid(self.conv3(x))
        return x


def _mask_generate(latent_feature: torch.Tensor, mask_net: _NetMatchability):
    """Apply mask network to latent features.

    Returns
    -------
    output : torch.Tensor
        Masked features (latent * mask).
    features_discarded : torch.Tensor
        Discarded features (latent * (1 - mask)).
    reverse_mask_sum : torch.Tensor
        Single-channel visualisation of the reverse mask.
    """
    latent_mask = mask_net(latent_feature)
    output = latent_feature * latent_mask
    reverse_mask = 1 - latent_mask
    reverse_mask_sum = torch.unsqueeze(torch.max(reverse_mask, dim=1)[0], dim=1)
    features_discarded = latent_feature * reverse_mask
    return output, features_discarded, reverse_mask_sum


# ---------------------------------------------------------------------------
# Generator A  (source → target, with optional extra-feature injection)
# ---------------------------------------------------------------------------

class StegoGANGeneratorA(ModelMixin, ConfigMixin):
    """ResNet generator A (source→target) with optional steganographic feature injection.

    Based on ``ResnetMaskV1Generator`` from the original StegoGAN.

    Parameters
    ----------
    input_nc : int
        Number of input channels.
    output_nc : int
        Number of output channels.
    ngf : int
        Base number of generator filters.
    n_blocks : int
        Number of ResNet blocks.
    norm_type : str
        Normalisation type (``"instance"`` or ``"batch"``).
    use_dropout : bool
        Whether to use dropout in ResNet blocks.
    resnet_layer : int
        Position where the extra-feature is injected. ``-1`` = before ResNet
        blocks, ``8`` = after all ResNet blocks, other values = after that
        many ResNet blocks.
    use_fusion_block : bool
        If True, apply a fusion ResNet block to the extra-feature before
        adding it.
    init_type : str
        Weight initialisation method.
    init_gain : float
        Gain for weight initialisation.
    """

    @register_to_config
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        norm_type: str = "instance",
        use_dropout: bool = False,
        resnet_layer: int = 8,
        use_fusion_block: bool = True,
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        self._resnet_layer = resnet_layer
        self._n_blocks = n_blocks

        norm_layer = _get_norm_layer(norm_type)
        use_bias = norm_type == "instance"

        # Encoder
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
        self.conv1 = nn.Sequential(*encoder)

        # ResNet blocks — split at resnet_layer
        mult = 2 ** n_downsampling
        blocks_before: list = []
        for _ in range(min(resnet_layer + 1, n_blocks) if resnet_layer >= 0 else 0):
            blocks_before.append(_ResnetBlock(ngf * mult, norm_layer, use_dropout, use_bias))
        self.conv2 = nn.Sequential(*blocks_before)

        blocks_after: list = []
        start = (resnet_layer + 1) if resnet_layer >= 0 else 0
        for _ in range(start, n_blocks):
            blocks_after.append(_ResnetBlock(ngf * mult, norm_layer, use_dropout, use_bias))
        self.conv3 = nn.Sequential(*blocks_after)

        # Decoder
        decoder: list = []
        for i in range(n_downsampling):
            m = 2 ** (n_downsampling - i)
            decoder += [
                nn.ConvTranspose2d(ngf * m, ngf * m // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(ngf * m // 2),
                nn.ReLU(True),
            ]
        decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

        # Optional fusion block
        self._use_fusion_block = use_fusion_block
        if use_fusion_block:
            self.fusion = nn.Sequential(
                _ResnetBlock(ngf * (2 ** n_downsampling), norm_layer, use_dropout, use_bias),
            )

        _init_weights(self, init_type, init_gain)

    def forward(self, x: torch.Tensor, extra_feature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image ``(B, C, H, W)``.
        extra_feature : torch.Tensor or None
            Optional latent feature from Generator B for steganographic
            embedding.
        """
        out = self.conv1(x)

        if extra_feature is not None and self._use_fusion_block:
            extra_feature = self.fusion(extra_feature)

        if self._resnet_layer == -1:
            if extra_feature is not None:
                out = out + extra_feature
            out = self.conv3(out)
        elif self._resnet_layer >= self._n_blocks:
            out = self.conv2(out)
            if extra_feature is not None:
                out = out + extra_feature
        else:
            out = self.conv2(out)
            if extra_feature is not None:
                out = out + extra_feature
            out = self.conv3(out)

        return self.decoder(out)


# ---------------------------------------------------------------------------
# Generator B  (target → source, with mask output)
# ---------------------------------------------------------------------------

class StegoGANGeneratorB(ModelMixin, ConfigMixin):
    """ResNet generator B (target→source) with latent mask output.

    Based on ``ResnetMaskV3Generator`` from the original StegoGAN.

    Parameters
    ----------
    input_nc : int
        Number of input channels (target domain).
    output_nc : int
        Number of output channels (source domain).
    ngf : int
        Base number of generator filters.
    n_blocks : int
        Number of ResNet blocks.
    norm_type : str
        Normalisation type (``"instance"`` or ``"batch"``).
    use_dropout : bool
        Whether to use dropout in ResNet blocks.
    mask_group : int
        Number of output channels for the matchability mask.
    resnet_layer : int
        Position where the mask is applied in the ResNet chain.
    init_type : str
        Weight initialisation method.
    init_gain : float
        Gain for weight initialisation.
    """

    @register_to_config
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        n_blocks: int = 9,
        norm_type: str = "instance",
        use_dropout: bool = False,
        mask_group: int = 256,
        resnet_layer: int = 8,
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        self._resnet_layer = resnet_layer
        self._n_blocks = n_blocks

        norm_layer = _get_norm_layer(norm_type)
        use_bias = norm_type == "instance"

        # Encoder
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
        self.conv1 = nn.Sequential(*encoder)

        # ResNet blocks — split at resnet_layer
        mult = 2 ** n_downsampling
        blocks_before: list = []
        for _ in range(min(resnet_layer + 1, n_blocks) if resnet_layer >= 0 else 0):
            blocks_before.append(_ResnetBlock(ngf * mult, norm_layer, use_dropout, use_bias))
        self.conv2 = nn.Sequential(*blocks_before)

        blocks_after: list = []
        start = (resnet_layer + 1) if resnet_layer >= 0 else 0
        for _ in range(start, n_blocks):
            blocks_after.append(_ResnetBlock(ngf * mult, norm_layer, use_dropout, use_bias))
        self.conv3 = nn.Sequential(*blocks_after)

        # Matchability mask network — out_dim must match bottleneck channels
        # for element-wise multiplication to work
        bottleneck_ch = mult * ngf
        self.mask = _NetMatchability(input_dim=bottleneck_ch, out_dim=bottleneck_ch)

        # Decoder
        decoder: list = []
        for i in range(n_downsampling):
            m = 2 ** (n_downsampling - i)
            decoder += [
                nn.ConvTranspose2d(ngf * m, ngf * m // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(ngf * m // 2),
                nn.ReLU(True),
            ]
        decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

        _init_weights(self, init_type, init_gain)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        output : torch.Tensor
            Translated image ``(B, output_nc, H, W)``.
        features_discarded : torch.Tensor
            Latent features identified as domain-mismatch.
        reverse_mask_sum : torch.Tensor
            Single-channel visualisation of the mismatch mask ``(B, 1, H/4, W/4)``.
        """
        out = self.conv1(x)

        if self._resnet_layer == -1:
            out, features_discarded, reverse_mask_sum = _mask_generate(out, self.mask)
            out = self.conv3(out)
        elif self._resnet_layer >= self._n_blocks:
            out = self.conv2(out)
            out, features_discarded, reverse_mask_sum = _mask_generate(out, self.mask)
        else:
            out = self.conv2(out)
            out, features_discarded, reverse_mask_sum = _mask_generate(out, self.mask)
            out = self.conv3(out)

        return self.decoder(out), features_discarded, reverse_mask_sum


# ---------------------------------------------------------------------------
# Discriminator (PatchGAN)
# ---------------------------------------------------------------------------

class StegoGANDiscriminator(ModelMixin, ConfigMixin):
    """N-layer PatchGAN discriminator for StegoGAN.

    Parameters
    ----------
    input_nc : int
        Number of input channels.
    ndf : int
        Base number of discriminator filters.
    n_layers : int
        Number of discriminator conv layers (default 3 → 70×70 receptive field).
    norm_type : str
        Normalisation type.
    init_type : str
        Weight initialisation method.
    init_gain : float
        Gain for weight initialisation.
    """

    @register_to_config
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_type: str = "instance",
        init_type: str = "normal",
        init_gain: float = 0.02,
    ) -> None:
        super().__init__()
        norm_layer = _get_norm_layer(norm_type)
        use_bias = norm_type == "instance"

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

        _init_weights(self, init_type, init_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# GAN loss
# ---------------------------------------------------------------------------

class StegoGANLoss(nn.Module):
    """GAN loss (LSGAN / Vanilla / WGAN-GP)."""

    def __init__(self, gan_mode: str = "lsgan") -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError(f"GAN mode [{gan_mode}] not implemented")

    def _get_target(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def __call__(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if self.gan_mode in ("lsgan", "vanilla"):
            target = self._get_target(prediction, target_is_real)
            return self.loss(prediction, target)
        # wgangp
        return -prediction.mean() if target_is_real else prediction.mean()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_generator_a(
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    n_blocks: int = 9,
    norm_type: str = "instance",
    use_dropout: bool = False,
    resnet_layer: int = 8,
    use_fusion_block: bool = True,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> StegoGANGeneratorA:
    """Create and initialise Generator A (source→target)."""
    return StegoGANGeneratorA(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_blocks=n_blocks,
        norm_type=norm_type,
        use_dropout=use_dropout,
        resnet_layer=resnet_layer,
        use_fusion_block=use_fusion_block,
        init_type=init_type,
        init_gain=init_gain,
    )


def create_generator_b(
    input_nc: int = 3,
    output_nc: int = 3,
    ngf: int = 64,
    n_blocks: int = 9,
    norm_type: str = "instance",
    use_dropout: bool = False,
    mask_group: int = 256,
    resnet_layer: int = 8,
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> StegoGANGeneratorB:
    """Create and initialise Generator B (target→source, with mask)."""
    return StegoGANGeneratorB(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_blocks=n_blocks,
        norm_type=norm_type,
        use_dropout=use_dropout,
        mask_group=mask_group,
        resnet_layer=resnet_layer,
        init_type=init_type,
        init_gain=init_gain,
    )


def create_discriminator(
    input_nc: int = 3,
    ndf: int = 64,
    n_layers: int = 3,
    norm_type: str = "instance",
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> StegoGANDiscriminator:
    """Create and initialise a PatchGAN discriminator."""
    return StegoGANDiscriminator(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=n_layers,
        norm_type=norm_type,
        init_type=init_type,
        init_gain=init_gain,
    )
