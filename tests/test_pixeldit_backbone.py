# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for the PixelDiT backbone integration.

Covers:
* Module imports and public API
* PixelDiTBackbone initialization and configuration
* Forward pass shapes (concat and unconditional modes)
* Factory integration (create_model with unet_type='pixeldit')
* Config save/load round-trip (ModelMixin / ConfigMixin)
* Gradient flow
"""

import torch
import pytest

from src.models.dit.pixeldit import PixelDiTBackbone
from src.models import (
    create_model,
    get_unet_type_config,
    SUPPORTED_DIT_TYPES,
    SUPPORTED_BACKBONE_TYPES,
    DIT_TYPE_PIXELDIT,
)


# ---------------------------------------------------------------------------
# Small model helper (fast tests)
# ---------------------------------------------------------------------------

def _make_small_model(**overrides):
    """Create a tiny PixelDiTBackbone for fast testing."""
    defaults = dict(
        image_size=32,
        in_channels=1,
        hidden_size=64,
        pixel_dim=8,
        patch_depth=2,
        pixel_depth=1,
        num_heads=4,
        pixel_num_heads=4,
        patch_size=4,
        mlp_ratio=2.0,
        condition_mode="concat",
        dropout=0.0,
    )
    defaults.update(overrides)
    return PixelDiTBackbone(**defaults)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """All public symbols are importable."""

    def test_pixeldit_backbone_class(self):
        assert PixelDiTBackbone is not None

    def test_unet_type_constant(self):
        assert DIT_TYPE_PIXELDIT == "pixeldit"

    def test_in_supported_types(self):
        assert DIT_TYPE_PIXELDIT in SUPPORTED_DIT_TYPES
        assert DIT_TYPE_PIXELDIT in SUPPORTED_BACKBONE_TYPES

    def test_get_unet_type_config(self):
        cfg = get_unet_type_config(DIT_TYPE_PIXELDIT)
        assert cfg["implemented"] is True
        assert "PixelDiT" in cfg["description"]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """PixelDiTBackbone initializes correctly."""

    def test_default_init(self):
        model = _make_small_model()
        assert isinstance(model, PixelDiTBackbone)

    def test_stores_config(self):
        model = _make_small_model()
        assert model.config["image_size"] == 32
        assert model.config["in_channels"] == 1
        assert model.config["condition_mode"] == "concat"

    def test_custom_params(self):
        model = _make_small_model(
            in_channels=3,
            hidden_size=128,
            num_heads=8,
            pixel_dim=16,
            patch_depth=4,
            pixel_depth=2,
        )
        assert model.hidden_size == 128
        assert model.pixel_dim == 16
        assert len(model.patch_blocks) == 4
        assert len(model.pixel_blocks) == 2

    def test_block_count(self):
        model = _make_small_model(patch_depth=3, pixel_depth=2)
        assert len(model.patch_blocks) == 3
        assert len(model.pixel_blocks) == 2


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForward:
    """Forward pass produces correct output shapes."""

    def test_concat_mode_1ch(self):
        model = _make_small_model(in_channels=1, condition_mode="concat")
        x = torch.randn(2, 1, 32, 32)
        t = torch.randn(2)
        xT = torch.randn(2, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == (2, 1, 32, 32)

    def test_concat_mode_3ch(self):
        model = _make_small_model(in_channels=3, condition_mode="concat")
        x = torch.randn(2, 3, 32, 32)
        t = torch.randn(2)
        xT = torch.randn(2, 3, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == (2, 3, 32, 32)

    def test_unconditional_mode(self):
        model = _make_small_model(condition_mode=None)
        x = torch.randn(2, 1, 32, 32)
        t = torch.randn(2)
        out = model(x, t)
        assert out.shape == (2, 1, 32, 32)

    def test_concat_with_xT(self):
        """With concat mode and xT provided, should work correctly."""
        model = _make_small_model(condition_mode="concat")
        x = torch.randn(1, 1, 32, 32)
        t = torch.randn(1)
        xT = torch.randn(1, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == (1, 1, 32, 32)

    def test_batch_size_one(self):
        model = _make_small_model()
        x = torch.randn(1, 1, 32, 32)
        t = torch.randn(1)
        xT = torch.randn(1, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == (1, 1, 32, 32)

    def test_no_nan(self):
        model = _make_small_model()
        x = torch.randn(2, 1, 32, 32)
        t = torch.randn(2)
        xT = torch.randn(2, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactory:
    """create_model factory correctly creates PixelDiT models."""

    def test_create_pixeldit(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            unet_type="pixeldit",
            condition_mode="concat",
            pixeldit_hidden_size=64,
            pixeldit_pixel_dim=8,
            pixeldit_patch_depth=2,
            pixeldit_pixel_depth=1,
            pixeldit_num_heads=4,
            pixeldit_pixel_num_heads=4,
            pixeldit_patch_size=4,
            pixeldit_mlp_ratio=2.0,
        )
        assert isinstance(model, PixelDiTBackbone)

    def test_factory_forward(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            unet_type="pixeldit",
            condition_mode="concat",
            pixeldit_hidden_size=64,
            pixeldit_pixel_dim=8,
            pixeldit_patch_depth=2,
            pixeldit_pixel_depth=1,
            pixeldit_num_heads=4,
            pixeldit_pixel_num_heads=4,
            pixeldit_patch_size=4,
            pixeldit_mlp_ratio=2.0,
        )
        x = torch.randn(2, 1, 32, 32)
        t = torch.randn(2)
        xT = torch.randn(2, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == (2, 1, 32, 32)

    def test_factory_defaults(self):
        """Factory with default pixeldit kwargs should still work."""
        model = create_model(
            image_size=256,
            in_channels=3,
            unet_type="pixeldit",
        )
        assert isinstance(model, PixelDiTBackbone)


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


class TestConfig:
    """Config save/load via ConfigMixin."""

    def test_config_roundtrip(self):
        model = _make_small_model(in_channels=3, hidden_size=128, num_heads=8)
        config = model.config
        model2 = PixelDiTBackbone.from_config(config)
        assert model2.config["in_channels"] == 3
        assert model2.config["hidden_size"] == 128
        assert model2.config["num_heads"] == 8


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradient:
    """Gradients flow through the model correctly."""

    def test_backward(self):
        model = _make_small_model()
        x = torch.randn(1, 1, 32, 32, requires_grad=True)
        t = torch.randn(1)
        xT = torch.randn(1, 1, 32, 32)
        out = model(x, t, xT=xT)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
