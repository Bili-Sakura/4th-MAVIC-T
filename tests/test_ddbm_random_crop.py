"""Tests for random crop in DDBM/DBIM dataset wrapper."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from examples.ddbm.dataset_wrapper import (
    MavicTDDBMDataset,
    _load_image_as_tensor,
)


class TestLoadImageRandomCrop:
    """Test _load_image_as_tensor with random crop modes."""

    def test_resize_only_default(self, tmp_path):
        """Without crop params, image is resized to resolution."""
        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.uint8(np.random.randint(0, 255, (64, 64, 3))))
        img.save(img_path)

        out = _load_image_as_tensor(str(img_path), channels=3, resolution=32)
        assert out.shape == (3, 32, 32)

    def test_direct_crop_from_large_image(self, tmp_path):
        """With use_random_crop and crop_pos, direct crop from original."""
        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.uint8(np.random.randint(0, 255, (64, 64, 3))))
        img.save(img_path)

        out = _load_image_as_tensor(
            str(img_path),
            channels=3,
            resolution=32,
            crop_pos=(16, 16),
            use_random_crop=True,
        )
        assert out.shape == (3, 32, 32)

    def test_resize_and_crop_mode(self, tmp_path):
        """With load_size > resolution, resize then crop."""
        img_path = tmp_path / "test.png"
        img = Image.fromarray(np.uint8(np.random.randint(0, 255, (32, 32, 3))))
        img.save(img_path)

        out = _load_image_as_tensor(
            str(img_path),
            channels=3,
            resolution=32,
            crop_pos=(0, 0),
            use_random_crop=True,
            load_size=64,
        )
        assert out.shape == (3, 32, 32)


class TestMavicTDDBMDatasetRandomCrop:
    """Test MavicTDDBMDataset exposes load_size and uses random crop when appropriate."""

    def test_dataset_accepts_load_size(self):
        """Dataset constructor accepts load_size parameter."""
        # Will fail at load() if dataset not found, but we can check the param exists
        import inspect
        sig = inspect.signature(MavicTDDBMDataset.__init__)
        assert "load_size" in sig.parameters

    def test_dataset_load_size_default_none(self):
        """load_size defaults to None for backward compatibility."""
        # MavicTDDBMDataset requires real dataset - we test the param is optional
        # by checking the constructor signature default
        import inspect
        sig = inspect.signature(MavicTDDBMDataset.__init__)
        assert sig.parameters["load_size"].default is None
