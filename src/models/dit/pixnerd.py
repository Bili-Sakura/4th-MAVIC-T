# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""PixNerd backbone adapted for the DDBM calling convention.

PixNerd (Wang et al., 2025) is a pixel-space diffusion transformer that uses
*neural fields* (per-pixel hypernetwork MLP) for high-frequency modeling.
Unlike UNet backbones, PixNerd processes patches at the image level with
self-attention, then refines individual pixels within each patch using
dynamically generated MLP weights (NerfBlock).

This module provides the **backbone** (denoiser network) so it can serve as
an alternative architecture to the diffusers UNet in the DDBM bridge
framework.  The wrapper class :class:`PixNerdBackbone` exposes the same
forward signature as :class:`DDBMUNet`::

    output = model(x, timestep, xT=source)

so the rest of the training / sampling code does not need any changes.

Note: The original PixNerd uses MultiScaleDCN (deformable convolution via
Triton kernels) inside FlattenDiTBlock.  This adaptation replaces it with
standard self-attention for portability (no Triton/CUDA dependency).  The
NerfBlock (neural field decoder) is kept intact as it uses pure PyTorch.

Reference: https://github.com/MCG-NJU/PixNerd
Paper: https://arxiv.org/abs/2507.23268
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


# ---------------------------------------------------------------------------
# Building blocks – adapted from PixNerd src/models/transformer/pixnerd_c2i.py
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
    def _sinusoidal(t: torch.Tensor, dim: int, max_period: float = 10.0) -> torch.Tensor:
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


class _PatchEmbed(nn.Module):
    """Linear patch embedding: (B, num_patches, C*P*P) → (B, N, D)."""

    def __init__(self, in_features: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _precompute_freqs_cis_2d(
    dim: int, height: int, width: int, theta: float = 10000.0, scale: float = 16.0,
) -> torch.Tensor:
    """Precompute 2-D rotary position embedding frequencies (complex)."""
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
    return freqs_cis.reshape(height * width, -1)


def _apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2-D rotary positional embedding to Q and K.

    xq, xk: (B, N, H, head_dim)
    freqs_cis: (N, head_dim//2) complex
    """
    freqs_cis = freqs_cis[None, :, None, :]  # (1, N, 1, head_dim//2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# -- Conditioning blocks (replace MultiScaleDCN with standard attention) --


class _FlattenDiTBlock(nn.Module):
    """DiT block with self-attention + RoPE + AdaLN + SwiGLU FFN.

    Replaces PixNerd's FlattenDiTBlock that uses MultiScaleDCN (Triton) with
    standard scaled-dot-product attention for portability.
    """

    def __init__(self, hidden_size: int, num_groups: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        head_dim = hidden_size // num_groups

        # AdaLN modulation
        self.adaLN = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self-attention (replaces MultiScaleDCN)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.q_norm = _RMSNorm(head_dim)
        self.k_norm = _RMSNorm(head_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # SwiGLU FFN
        ffn_hidden = int(2 * hidden_size * 4 / 3)
        self.w1 = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.w3 = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.w2 = nn.Linear(ffn_hidden, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_groups
        head_dim = D // H

        # AdaLN modulation
        mod = self.adaLN(c)  # (B, 1, 6D) or (B, N, 6D)
        if mod.dim() == 2:
            mod = mod.unsqueeze(1)
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = mod.chunk(6, dim=-1)

        # Attention branch
        x_norm = self.norm1(x) * (1 + scale_attn) + shift_attn
        qkv = self.qkv(x_norm).reshape(B, N, 3, H, head_dim)
        q, k, v = qkv.unbind(2)  # each (B, N, H, head_dim)

        # Per-head QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q, k = _apply_rotary_emb(q, k, pos)

        # Standard attention
        q = q.transpose(1, 2)  # (B, H, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, N, D)
        attn_out = self.o_proj(attn_out)
        x = x + gate_attn * attn_out

        # FFN branch (SwiGLU)
        x_norm2 = self.norm2(x) * (1 + scale_ffn) + shift_ffn
        ffn_out = self.w2(F.silu(self.w1(x_norm2)) * self.w3(x_norm2))
        x = x + gate_ffn * ffn_out

        return x


# -- Neural field blocks (pure PyTorch, kept from PixNerd) --


class _NerfEmbedder(nn.Module):
    """Positional encoding for per-pixel features within a patch."""

    def __init__(self, in_channels: int, hidden_size_x: int, max_freqs: int = 8) -> None:
        super().__init__()
        self.max_freqs = max_freqs
        self.proj = nn.Linear(in_channels * (max_freqs * 2 + 1), hidden_size_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x]
        for freq in range(self.max_freqs):
            parts.append(torch.sin(x * (2 ** freq * math.pi)))
            parts.append(torch.cos(x * (2 ** freq * math.pi)))
        return self.proj(torch.cat(parts, dim=-1))


class _NerfBlock(nn.Module):
    """Hypernetwork MLP block: generates MLP weights from conditioning ``s``
    and applies them to per-pixel features ``x``.
    """

    def __init__(self, hidden_size_s: int, hidden_size_x: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.param_generator = nn.Linear(
            hidden_size_s, 2 * hidden_size_x ** 2 * mlp_ratio,
        )
        self.norm = _RMSNorm(hidden_size_x)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        batch_size, num_x, hidden_size_x = x.shape
        params = self.param_generator(s)
        fc1, fc2 = params.chunk(2, dim=-1)
        fc1 = fc1.view(batch_size, hidden_size_x, hidden_size_x * self.mlp_ratio)
        fc2 = fc2.view(batch_size, hidden_size_x * self.mlp_ratio, hidden_size_x)
        fc1 = F.normalize(fc1, dim=-2)
        fc2 = F.normalize(fc2, dim=-2)

        res = x
        x = self.norm(x)
        x = torch.bmm(x, fc1)
        x = F.silu(x)
        x = torch.bmm(x, fc2)
        return x + res


class _NerfFinalLayer(nn.Module):
    """Final projection for neural field output."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = _RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


# ---------------------------------------------------------------------------
# Main backbone
# ---------------------------------------------------------------------------


class PixNerdBackbone(ModelMixin, ConfigMixin):
    """PixNerd DiT + NerfBlock backbone for the DDBM calling convention.

    This is a **backbone-only** integration.  The original PixNerd uses flow
    matching which is not compatible with the DDBM diffusion-bridge scheduler.
    The transformer + neural-field architecture, however, can serve as a
    denoiser network within the bridge framework.

    Architecture:
    1. Patchify input → linear embedding (``s_embedder``)
    2. FlattenDiTBlocks (self-attention + RoPE + AdaLN) for patch-level reasoning
    3. NerfBlocks (hypernetwork MLP) for per-pixel refinement within each patch
    4. Unpatchify back to image space

    Parameters
    ----------
    image_size : int
        Spatial resolution (H == W).
    in_channels : int
        Channels of the *target* image.  When ``condition_mode='concat'``
        the network internally doubles the input channels.
    hidden_size : int
        Transformer hidden dimension.
    hidden_size_x : int
        Per-pixel hidden dimension for neural field blocks.
    nerf_mlp_ratio : int
        Hidden-dim multiplier for the NerfBlock hypernetwork MLP.
    num_blocks : int
        Total number of blocks (conditioning + nerf).
    num_cond_blocks : int
        Number of self-attention conditioning blocks (first ``num_cond_blocks``
        blocks are FlattenDiTBlocks; the rest are NerfBlocks).
    patch_size : int
        Patch size for patchification.
    num_groups : int
        Number of attention heads in conditioning blocks.
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
        hidden_size_x: int = 64,
        nerf_mlp_ratio: int = 4,
        num_blocks: int = 18,
        num_cond_blocks: int = 4,
        patch_size: int = 2,
        num_groups: int = 12,
        condition_mode: Optional[str] = "concat",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.num_blocks = num_blocks
        self.num_cond_blocks = num_cond_blocks
        self.patch_size = patch_size
        self.condition_mode = condition_mode

        embed_in = in_channels * 2 if condition_mode == "concat" else in_channels

        # Patch-level embedding
        self.s_embedder = _PatchEmbed(embed_in * patch_size ** 2, hidden_size)

        # Timestep embedding
        self.t_embedder = _TimestepEmbedder(hidden_size)

        # Per-pixel embedding (neural field)
        self.x_embedder = _NerfEmbedder(embed_in, hidden_size_x, max_freqs=8)

        # Conditioning blocks (self-attention + RoPE + AdaLN)
        blocks: list[nn.Module] = [
            _FlattenDiTBlock(hidden_size, num_groups)
            for _ in range(num_cond_blocks)
        ]
        # Neural field blocks (hypernetwork MLP)
        blocks.extend([
            _NerfBlock(hidden_size, hidden_size_x, nerf_mlp_ratio)
            for _ in range(num_cond_blocks, num_blocks)
        ])
        self.blocks = nn.ModuleList(blocks)

        # Output layer
        self.final_layer = _NerfFinalLayer(hidden_size_x, in_channels)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Positional embedding cache
        self._pos_cache: dict[tuple[int, int], torch.Tensor] = {}

        self._initialize_weights()

    # ---- init helpers -------------------------------------------------------

    def _initialize_weights(self) -> None:
        """Initialize following PixNerd conventions."""
        # Patch embed
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # NerfFinalLayer zero init
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    # ---- positional embedding -----------------------------------------------

    def _fetch_pos(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width)
        if key not in self._pos_cache:
            self._pos_cache[key] = _precompute_freqs_cis_2d(
                self.hidden_size // self.num_groups, height, width,
            )
        return self._pos_cache[key].to(device)

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
            Raw model output (before ``c_out / c_skip`` application by the
            pipeline scheduler).
        """
        if self.condition_mode == "concat" and xT is not None:
            x = torch.cat([x, xT], dim=1)

        B, C_in, H, W = x.shape
        P = self.patch_size

        # Unfold into patches: (B, C_in*P*P, num_patches) → (B, N, C_in*P*P)
        x_patches = F.unfold(x, kernel_size=P, stride=P).transpose(1, 2)

        # RoPE positional embedding for conditioning blocks
        pos = self._fetch_pos(H // P, W // P, x.device)

        # Timestep conditioning
        t_emb = self.t_embedder(timestep.view(-1)).view(B, 1, self.hidden_size)
        c = F.silu(t_emb)

        # Phase 1: Patch-level conditioning with self-attention
        s = self.s_embedder(x_patches)
        s = self.dropout_layer(s)
        for i in range(self.num_cond_blocks):
            s = self.blocks[i](s, c, pos)
        s = F.silu(t_emb + s)  # (B, N, D) per-patch conditioning

        # Phase 2: Per-pixel neural field refinement
        batch_size, length, _ = s.shape
        # Reshape patches to per-pixel: (B*N, P*P, C_in)
        x_pixels = x_patches.reshape(batch_size * length, C_in, P * P)
        x_pixels = x_pixels.transpose(1, 2)  # (B*N, P*P, C_in)
        s_flat = s.reshape(batch_size * length, self.hidden_size)  # (B*N, D)

        x_pixels = self.x_embedder(x_pixels)  # (B*N, P*P, hidden_size_x)
        for i in range(self.num_cond_blocks, self.num_blocks):
            x_pixels = self.blocks[i](x_pixels, s_flat)
        x_pixels = self.final_layer(x_pixels)  # (B*N, P*P, out_channels)

        # Fold back into image: (B, N, out_channels*P*P) → (B, out_channels, H, W)
        x_pixels = x_pixels.transpose(1, 2)  # (B*N, out_channels, P*P)
        x_pixels = x_pixels.reshape(batch_size, length, -1)  # (B, N, out_channels*P*P)
        output = F.fold(
            x_pixels.transpose(1, 2).contiguous(),
            (H, W),
            kernel_size=P,
            stride=P,
        )
        return output
