# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2024 The UniDB Authors (https://github.com/2769433owo/UniDB-plusplus).
# Licensed under the Apache License, Version 2.0 (the "License");
#
# UniDB ConditionalUNet architecture. The model predicts noise; it accepts
# (xt, mu, t) where mu is the condition (LQ/source image) and t is timestep 1..T.

from __future__ import annotations

import functools
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange


def _exists(x):
    return x is not None


def _default(val, d):
    if _exists(val):
        return val
    return d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def NonLinearity(inplace=False):
    return nn.SiLU(inplace)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, _default(dim_out, dim), 3, 1, 1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, _default(dim_out, dim), 4, 2, 1)


def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(
        dim_in, dim_out, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class Block(nn.Module):
    def __init__(self, conv, dim_in, dim_out, act=NonLinearity()):
        super().__init__()
        self.proj = conv(dim_in, dim_out)
        self.act = act

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        if _exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self, conv, dim_in, dim_out, time_emb_dim=None, act=NonLinearity()
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(act, nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim
            else None
        )
        self.block1 = Block(conv, dim_in, dim_out, act)
        self.block2 = Block(conv, dim_out, dim_out, act)
        self.res_conv = (
            conv(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None
        if _exists(self.mlp) and _exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        v = v / (h * w)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class UniDBConditionalUNet(ModelMixin, ConfigMixin):
    """UniDB ConditionalUNet: predicts noise given (xt, mu, t).

    Compatible with UniDB pretrained checkpoints from
    https://github.com/2769433owo/UniDB-plusplus.

    Args:
        in_channels: Input channels (e.g. 3 for RGB).
        out_channels: Output channels (same as in_channels for denoising).
        nf: Base channel count.
        depth: Number of down/up blocks (default 4).
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        nf: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        self.depth = depth
        block_class = functools.partial(
            ResBlock, conv=default_conv, act=NonLinearity()
        )

        self.init_conv = default_conv(in_channels * 2, nf, 7)
        time_dim = nf * 4
        sinu_pos_emb = SinusoidalPosEmb(nf)
        fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i + 1))
            self.downs.append(
                nn.ModuleList(
                    [
                        block_class(
                            dim_in=dim_in,
                            dim_out=dim_in,
                            time_emb_dim=time_dim,
                        ),
                        block_class(
                            dim_in=dim_in,
                            dim_out=dim_in,
                            time_emb_dim=time_dim,
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if i != (depth - 1)
                        else default_conv(dim_in, dim_out),
                    ]
                )
            )
            self.ups.insert(
                0,
                nn.ModuleList(
                    [
                        block_class(
                            dim_in=dim_out + dim_in,
                            dim_out=dim_out,
                            time_emb_dim=time_dim,
                        ),
                        block_class(
                            dim_in=dim_out + dim_in,
                            dim_out=dim_out,
                            time_emb_dim=time_dim,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if i != 0
                        else default_conv(dim_out, dim_in),
                    ]
                )
            )

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(
            dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim
        )
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(
            dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim
        )

        self.final_res_block = block_class(
            dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim
        )
        self.final_conv = nn.Conv2d(nf, out_channels, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def forward(
        self,
        xt: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Predict noise. t: timestep 1..T (int or tensor)."""
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=xt.device, dtype=xt.dtype)

        x = xt - mu
        x = torch.cat([x, mu], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t_emb = self.time_mlp(t.float())

        h = []
        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t_emb)
            h.append(x)
            x = b2(x, t_emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t_emb)
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t_emb)
            x = attn(x)
            x = upsample(x)

        x = torch.cat([x, x_], dim=1)
        x = self.final_res_block(x, t_emb)
        x = self.final_conv(x)

        x = x[..., :H, :W]
        return x
