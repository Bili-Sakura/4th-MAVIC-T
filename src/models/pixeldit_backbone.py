# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""PixelDiT backbone adapted for the DDBM calling convention.

PixelDiT (2025) is a pixel-space diffusion transformer with a dual-level
architecture: patch-level blocks for global semantics and pixel-level
transformer blocks for fine-grained texture detail.

Key innovations over prior pixel-space DiT work:
* **Pixel-wise AdaLN**: per-pixel modulation instead of patch-wise broadcasting.
* **Pixel Token Compaction**: linear maps that compress p² pixel tokens into
  one semantic-space token for global attention, then expand back.

This module provides the **backbone** (denoiser network) so it can serve as
an alternative architecture to the diffusers UNet in the DDBM bridge
framework.  The wrapper class :class:`PixelDiTBackbone` exposes the same
forward signature as :class:`DDBMUNet`::

    output = model(x, timestep, xT=source)

so the rest of the training / sampling code does not need any changes.

Note: The original PixelDiT class-conditioned model uses class label
embeddings, which are replaced here with a source-image conditioning
pathway (channel concatenation).  The text-to-image (PixelDiT_T2I) variant
is NOT included here; only the class-conditioned backbone is adapted.

Reference: https://github.com/Holasyb918/PixelDiT-Hack
Paper: https://arxiv.org/abs/2511.20645
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


# ---------------------------------------------------------------------------
# Positional Embedding Utilities (inlined from PixelDiT)
# ---------------------------------------------------------------------------


def _get_1d_sincos_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


# ---------------------------------------------------------------------------
# Basic components
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """RMS normalization (LlamaRMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * out).to(x.dtype)


def _modulate(
    x: torch.Tensor,
    shift: Optional[torch.Tensor],
    scale: torch.Tensor,
) -> torch.Tensor:
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


class _TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[..., None].float() * freqs[None, ...]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t, self.freq_dim))


# ---------------------------------------------------------------------------
# RoPE for patch-level attention
# ---------------------------------------------------------------------------


def _broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(len(t.shape) for t in tensors)
    assert len(shape_lens) == 1
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*(list(t.shape) for t in tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(len(set(t[1])) <= 2 for t in expandable_dims)
    max_dims = [(t[0], max(t[1])) for t in expandable_dims]
    expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
    tensors = [t.expand(*s) for t, s in zip(tensors, expandable_shapes)]
    return torch.cat(tensors, dim=dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


class _VisionRotaryEmbeddingFast(nn.Module):
    def __init__(self, dim: int, pt_seq_len: int = 16, theta: float = 10000.0) -> None:
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(pt_seq_len).float()
        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = freqs.repeat(1, 2)
        freqs = _broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.freqs_cos + _rotate_half(t) * self.freqs_sin


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    """Multi-head self-attention with optional RoPE and QK-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = _RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = _RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------


class _SwiGLUFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


# ---------------------------------------------------------------------------
# Patch-level DiT Block
# ---------------------------------------------------------------------------


class _PatchDiTBlock(nn.Module):
    """Patch-level DiT Block with AdaLN + RoPE + SwiGLU."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(hidden_size)
        self.norm2 = _RMSNorm(hidden_size)
        self.attn = _Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = _SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope=None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1)),
            rope=rope,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        )
        return x


# ---------------------------------------------------------------------------
# Pixel-wise AdaLN & Token Compaction
# ---------------------------------------------------------------------------


class _PixelwiseAdaLN(nn.Module):
    """Pixel-wise AdaLN: expand one semantic token into p² sets of modulation params."""

    def __init__(self, semantic_dim: int, pixel_dim: int, patch_size: int) -> None:
        super().__init__()
        self.num_pixels = patch_size * patch_size
        self.pixel_dim = pixel_dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(semantic_dim, self.num_pixels * 6 * pixel_dim),
        )

    def forward(self, s_cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        theta = self.mlp(s_cond)
        theta = theta.view(-1, self.num_pixels, 6 * self.pixel_dim)
        return theta.chunk(6, dim=-1)


class _PixelTokenCompaction(nn.Module):
    """Compress/expand pixel tokens to/from semantic space."""

    def __init__(self, pixel_dim: int, semantic_dim: int, patch_size: int) -> None:
        super().__init__()
        self.num_pixels = patch_size * patch_size
        self.pixel_dim = pixel_dim
        self.compress = nn.Linear(self.num_pixels * pixel_dim, semantic_dim)
        self.expand = nn.Linear(semantic_dim, self.num_pixels * pixel_dim)

    def compact(self, x: torch.Tensor) -> torch.Tensor:
        BL = x.shape[0]
        return self.compress(x.view(BL, -1)).unsqueeze(1)

    def decompact(self, x: torch.Tensor) -> torch.Tensor:
        BL = x.shape[0]
        return self.expand(x.squeeze(1)).view(BL, self.num_pixels, self.pixel_dim)


# ---------------------------------------------------------------------------
# Pixel Transformer Block
# ---------------------------------------------------------------------------


class _PixelTransformerBlock(nn.Module):
    """Pixel Transformer (PiT) block with pixel-wise AdaLN + token compaction."""

    def __init__(
        self,
        pixel_dim: int,
        semantic_dim: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.pixel_dim = pixel_dim
        self.semantic_dim = semantic_dim
        self.num_pixels = patch_size * patch_size
        self.pixelwise_adaln = _PixelwiseAdaLN(semantic_dim, pixel_dim, patch_size)
        self.compaction = _PixelTokenCompaction(pixel_dim, semantic_dim, patch_size)
        self.norm1 = _RMSNorm(pixel_dim)
        self.norm2 = _RMSNorm(pixel_dim)
        self.attn = _Attention(semantic_dim, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        mlp_hidden = int(pixel_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(pixel_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, pixel_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
        rope=None,
    ) -> torch.Tensor:
        B, L, p2, D_pix = x.shape
        D_sem = self.semantic_dim
        x = x.reshape(B * L, p2, D_pix)
        s_cond_flat = s_cond.reshape(B * L, -1)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.pixelwise_adaln(s_cond_flat)

        # Attention via compaction
        x_mod = self.norm1(x) * (1 + scale1) + shift1
        x_compact = self.compaction.compact(x_mod).reshape(B, L, D_sem)
        x_attn = self.attn(x_compact, rope).reshape(B * L, 1, D_sem)
        x_attn = self.compaction.decompact(x_attn)
        x = x + gate1 * x_attn

        # MLP
        x_mod = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(x_mod)

        return x.reshape(B, L, p2, D_pix)


# ---------------------------------------------------------------------------
# Final Layer
# ---------------------------------------------------------------------------


class _PixelFinalLayer(nn.Module):
    """Project pixel tokens to output channels with AdaLN modulation."""

    def __init__(self, semantic_dim: int, pixel_dim: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = _RMSNorm(pixel_dim)
        self.linear = nn.Linear(pixel_dim, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(semantic_dim, 2 * pixel_dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.linear(x)


# ---------------------------------------------------------------------------
# Main backbone
# ---------------------------------------------------------------------------


class PixelDiTBackbone(ModelMixin, ConfigMixin):
    """PixelDiT dual-level DiT backbone for the DDBM calling convention.

    This is a **backbone-only** integration.  The original PixelDiT is
    designed for class-conditioned generation; here class embeddings are
    removed and the source image is injected via channel concatenation
    (same as :class:`PixNerdBackbone`).

    Architecture:
    1. Patch-level pathway: N ``_PatchDiTBlock`` blocks with self-attention
       + RoPE + AdaLN + SwiGLU in the semantic space D.
    2. Pixel-level pathway: M ``_PixelTransformerBlock`` blocks operating in
       pixel space D_pix with pixel-wise AdaLN and token compaction.
    3. Final projection back to image space.

    Parameters
    ----------
    image_size : int
        Spatial resolution (H == W).
    in_channels : int
        Channels of the *target* image.  When ``condition_mode='concat'``
        the network internally doubles the input channels.
    hidden_size : int
        Semantic dimension D for patch-level pathway.
    pixel_dim : int
        Pixel dimension D_pix for pixel-level pathway (paper default 16).
    patch_depth : int
        N – number of patch-level DiT blocks.
    pixel_depth : int
        M – number of pixel-level transformer blocks.
    num_heads : int
        Attention heads for patch-level blocks.
    pixel_num_heads : int
        Attention heads for pixel-level (semantic-space) attention.
    patch_size : int
        Patch size for patchification.
    mlp_ratio : float
        FFN hidden-dim multiplier.
    condition_mode : str or None
        ``'concat'`` to concatenate source image along channels, or ``None``
        for unconditional mode.
    dropout : float
        Dropout probability.
    """

    @register_to_config
    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        hidden_size: int = 1152,
        pixel_dim: int = 16,
        patch_depth: int = 26,
        pixel_depth: int = 4,
        num_heads: int = 16,
        pixel_num_heads: int = 16,
        patch_size: int = 16,
        mlp_ratio: float = 4.0,
        condition_mode: Optional[str] = "concat",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.condition_mode = condition_mode

        embed_in = in_channels * 2 if condition_mode == "concat" else in_channels

        num_patches = (image_size // patch_size) ** 2

        # -- Patch-level pathway --
        self.patch_embed = nn.Conv2d(
            embed_in, hidden_size, kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False,
        )
        self.t_embedder = _TimestepEmbedder(hidden_size)

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = image_size // patch_size
        self.patch_rope = _VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)

        self.patch_blocks = nn.ModuleList([
            _PatchDiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(patch_depth)
        ])

        # -- Pixel-level pathway --
        self.pixel_embed = nn.Conv2d(embed_in, pixel_dim, kernel_size=1, stride=1)
        self.pixel_pos_embed = nn.Parameter(
            torch.zeros(1, image_size * image_size, pixel_dim), requires_grad=False,
        )

        self.pixel_blocks = nn.ModuleList([
            _PixelTransformerBlock(
                pixel_dim=pixel_dim,
                semantic_dim=hidden_size,
                num_heads=pixel_num_heads,
                patch_size=patch_size,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(pixel_depth)
        ])

        # -- Final layer --
        self.final_layer = _PixelFinalLayer(hidden_size, pixel_dim, in_channels)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._initialize_weights(num_patches)

    # ---- init helpers -------------------------------------------------------

    def _initialize_weights(self, num_patches: int) -> None:
        """Initialize following PixelDiT conventions."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Position embeddings (sincos)
        pos_embed = _get_2d_sincos_pos_embed(self.hidden_size, int(num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pixel_pos_embed = _get_2d_sincos_pos_embed(self.pixel_dim, self.image_size)
        self.pixel_pos_embed.data.copy_(torch.from_numpy(pixel_pos_embed).float().unsqueeze(0))

        # Patch / pixel embedding
        for embed in [self.patch_embed, self.pixel_embed]:
            w = embed.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            nn.init.constant_(embed.bias, 0)

        # Timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out AdaLN layers
        for block in self.patch_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.pixel_blocks:
            nn.init.constant_(block.pixelwise_adaln.mlp[-1].weight, 0)
            nn.init.constant_(block.pixelwise_adaln.mlp[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ---- patchify / unpatchify (pixel-level) --------------------------------

    def _patchify_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D_pix, H, W) → (B, L, p², D_pix)."""
        B, D, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p
        # (B, D, h, p, w, p) → (B, h*w, p*p, D)
        x = x.reshape(B, D, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, D)
        x = x.reshape(B, h * w, p * p, D)
        return x

    def _unpatchify_pixels(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, L, p², C) → (B, C, H, W)."""
        p = self.patch_size
        h, w = H // p, W // p
        B = x.shape[0]
        C = x.shape[-1]
        x = x.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        x = x.reshape(B, C, H, W)
        return x

    # ---- forward ------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        xT: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass matching the DDBM UNet calling convention.

        Parameters
        ----------
        x : Tensor  (B, C, H, W)
            Pre-conditioned noisy sample.
        timestep : Tensor  (B,)
            Rescaled log-sigma timestep.
        xT : Tensor or None  (B, C, H, W)
            Source/condition image.

        Returns
        -------
        Tensor  (B, C, H, W)
            Raw model output.
        """
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)

        B, C_in, H, W = x.shape
        p = self.patch_size
        L = (H // p) * (W // p)

        # -- Timestep conditioning --
        t_emb = self.t_embedder(timestep.view(-1))  # (B, D)

        # -- Patch-level pathway --
        s = self.patch_embed(x)              # (B, D, H/p, W/p)
        s = s.flatten(2).transpose(1, 2)     # (B, L, D)
        s = s + self.pos_embed
        s = self.dropout_layer(s)

        for block in self.patch_blocks:
            s = block(s, t_emb, self.patch_rope)

        # Semantic conditioning: add timestep
        s_cond = s + t_emb.unsqueeze(1)      # (B, L, D)

        # -- Pixel-level pathway --
        p_embed = self.pixel_embed(x)        # (B, D_pix, H, W)
        p_tokens = self._patchify_pixels(p_embed)  # (B, L, p², D_pix)
        pos_embed_patches = self.pixel_pos_embed.view(1, L, p * p, self.pixel_dim)
        p_tokens = p_tokens + pos_embed_patches

        for block in self.pixel_blocks:
            p_tokens = block(p_tokens, s_cond, self.patch_rope)

        # -- Final projection --
        p_flat = p_tokens.view(B, -1, self.pixel_dim)
        p_out = self.final_layer(p_flat, t_emb)  # (B, L*p², C_out)
        p_out = p_out.view(B, L, p * p, self.in_channels)
        output = self._unpatchify_pixels(p_out, H, W)

        return output
