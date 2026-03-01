# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""MultiDiffusion-style tiled inference for high-res generation.

Shared utilities for running diffusion models trained at 512px (64x64 latent)
at 1024px (128x128 latent) by splitting into overlapping windows and
averaging predictions. See MultiDiffusion https://huggingface.co/papers/2302.08113.
"""

from typing import List, Tuple

# 512px → 64 latent; window and stride in latent grid
DEFAULT_LATENT_WINDOW_SIZE = 64
DEFAULT_LATENT_STRIDE = 8

# Pixel-space MultiDiffusion: trained at 512px
DEFAULT_PIXEL_WINDOW_SIZE = 512
DEFAULT_PIXEL_STRIDE = 64


def get_views(
    latent_height: int,
    latent_width: int,
    window_size: int = DEFAULT_LATENT_WINDOW_SIZE,
    stride: int = DEFAULT_LATENT_STRIDE,
) -> List[Tuple[int, int, int, int]]:
    """
    MultiDiffusion-style view layout for tiled high-res inference.

    Returns list of (h_start, h_end, w_start, w_end) in latent grid so the
    UNet only sees window_size x window_size crops (e.g. 64x64 = 512px).
    See Sec. 4.1 in MultiDiffusion https://huggingface.co/papers/2302.08113.

    Args:
        latent_height: Height of the latent grid.
        latent_width: Width of the latent grid.
        window_size: Size of each view in latent space (default 64).
        stride: Stride between views (default 8).

    Returns:
        List of (h_start, h_end, w_start, w_end) for each view.
    """
    num_blocks_h = (
        (latent_height - window_size) // stride + 1
        if latent_height > window_size
        else 1
    )
    num_blocks_w = (
        (latent_width - window_size) // stride + 1
        if latent_width > window_size
        else 1
    )
    views = []
    for i in range(num_blocks_h * num_blocks_w):
        h_start = (i // num_blocks_w) * stride
        h_end = h_start + window_size
        w_start = (i % num_blocks_w) * stride
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views
