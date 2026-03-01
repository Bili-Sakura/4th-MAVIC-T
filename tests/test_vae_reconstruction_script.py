# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.test_vae_reconstruction import (  # noqa: E402
    compute_reconstruction_metrics,
    load_image_to_three_channels,
    reduce_reconstruction_channels,
)


def test_load_image_expands_grayscale(tmp_path: Path):
    img = Image.fromarray(np.full((4, 4), 128, dtype=np.uint8), mode="L")
    path = tmp_path / "gray.png"
    img.save(path)

    tensor, orig_channels = load_image_to_three_channels(path, resolution=4, device=torch.device("cpu"))
    assert tensor.shape == (1, 3, 4, 4)
    assert orig_channels == 1
    assert torch.allclose(tensor[:, 0], tensor[:, 1])
    assert torch.allclose(tensor[:, 1], tensor[:, 2])


def test_reduce_reconstruction_channels_averages_single_channel():
    recon = torch.stack(
        [
            torch.zeros(2, 2),
            torch.ones(2, 2) * 0.5,
            torch.ones(2, 2),
        ],
        dim=0,
    ).unsqueeze(0)

    reduced = reduce_reconstruction_channels(recon, orig_channels=1)
    expected = torch.full((1, 1, 2, 2), 0.5)
    assert reduced.shape == (1, 1, 2, 2)
    assert torch.allclose(reduced, expected)


def test_reduce_reconstruction_channels_two_channels():
    recon = torch.stack(
        [
            torch.full((2, 2), 0.25),
            torch.full((2, 2), 0.75),
            torch.full((2, 2), 0.5),
        ],
        dim=0,
    ).unsqueeze(0)

    reduced = reduce_reconstruction_channels(recon, orig_channels=2)
    assert reduced.shape == (1, 2, 2, 2)
    assert torch.allclose(reduced[:, 0], recon[:, 0])
    assert torch.allclose(reduced[:, 1], recon[:, 1])


def test_compute_reconstruction_metrics_identical_images():
    img = torch.zeros(1, 1, 4, 4)
    metrics = compute_reconstruction_metrics(img, img)
    assert metrics["mae"] == pytest.approx(0.0)
    assert math.isinf(metrics["psnr"])
    assert metrics["ssim"] == pytest.approx(1.0)
