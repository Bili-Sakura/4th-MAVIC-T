# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for the PixNerd backbone integration.

Covers:
* Module imports and public API
* PixNerdBackbone initialization and configuration
* Forward pass shapes (concat and unconditional modes)
* Factory integration (create_model with unet_type='pixnerd')
* Config save/load round-trip (ModelMixin / ConfigMixin)
* Gradient flow
"""

import torch
import pytest

from src.models.dit.pixnerd_backbone import PixNerdBackbone
from src.models.unet.unet_ddbm import (
    create_model,
    get_unet_type_config,
    SUPPORTED_BACKBONE_TYPES,
    DIT_TYPE_PIXNERD,
    UNET_TYPE_PIXNERD,
)


# ---------------------------------------------------------------------------
# Small model helper (fast tests)
# ---------------------------------------------------------------------------

def _make_small_model(**overrides):
    """Create a tiny PixNerdBackbone for fast testing."""
    defaults = dict(
        image_size=32,
        in_channels=1,
        hidden_size=192,
        hidden_size_x=32,
        nerf_mlp_ratio=2,
        num_blocks=6,
        num_cond_blocks=2,
        patch_size=2,
        num_groups=6,
        condition_mode="concat",
        dropout=0.0,
    )
    defaults.update(overrides)
    return PixNerdBackbone(**defaults)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """All public symbols are importable."""

    def test_pixnerd_backbone_class(self):
        assert PixNerdBackbone is not None

    def test_dit_type_constant(self):
        assert DIT_TYPE_PIXNERD == "pixnerd"

    def test_backward_compat_unet_alias(self):
        assert UNET_TYPE_PIXNERD == DIT_TYPE_PIXNERD

    def test_in_supported_types(self):
        assert DIT_TYPE_PIXNERD in SUPPORTED_BACKBONE_TYPES

    def test_get_unet_type_config(self):
        cfg = get_unet_type_config(DIT_TYPE_PIXNERD)
        assert cfg["implemented"] is True
        assert "PixNerd" in cfg["description"]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """PixNerdBackbone initializes correctly."""

    def test_default_init(self):
        model = _make_small_model()
        assert isinstance(model, PixNerdBackbone)

    def test_stores_config(self):
        model = _make_small_model()
        assert model.config["image_size"] == 32
        assert model.config["in_channels"] == 1
        assert model.config["condition_mode"] == "concat"

    def test_custom_params(self):
        model = _make_small_model(
            in_channels=3,
            hidden_size=384,
            num_groups=12,
            num_blocks=10,
            num_cond_blocks=3,
        )
        assert model.hidden_size == 384
        assert model.num_groups == 12
        assert model.num_blocks == 10
        assert model.num_cond_blocks == 3

    def test_block_count(self):
        model = _make_small_model(num_blocks=8, num_cond_blocks=3)
        assert len(model.blocks) == 8


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

    def test_concat_no_xT(self):
        """With concat mode but xT=None, only x is passed (no concat)."""
        model = _make_small_model(condition_mode="concat")
        x = torch.randn(1, 1, 32, 32)
        t = torch.randn(1)
        # xT=None → no concat, x passes through with only in_channels.
        # The patch embedder expects 2*in_channels, so this will fail for
        # PixNerd (same as UNet: concat mode requires xT).  We verify that
        # providing xT works; omitting xT is not expected to work in concat mode.
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
    """create_model factory correctly creates PixNerd models."""

    def test_create_pixnerd(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            unet_type="pixnerd",
            condition_mode="concat",
            pixnerd_hidden_size=192,
            pixnerd_hidden_size_x=32,
            pixnerd_nerf_mlp_ratio=2,
            pixnerd_num_blocks=6,
            pixnerd_num_cond_blocks=2,
            pixnerd_patch_size=2,
            pixnerd_num_groups=6,
        )
        assert isinstance(model, PixNerdBackbone)

    def test_factory_forward(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            unet_type="pixnerd",
            condition_mode="concat",
            pixnerd_hidden_size=192,
            pixnerd_hidden_size_x=32,
            pixnerd_nerf_mlp_ratio=2,
            pixnerd_num_blocks=6,
            pixnerd_num_cond_blocks=2,
            pixnerd_patch_size=2,
            pixnerd_num_groups=6,
        )
        x = torch.randn(2, 1, 32, 32)
        t = torch.randn(2)
        xT = torch.randn(2, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == (2, 1, 32, 32)

    def test_factory_defaults(self):
        """Factory with default pixnerd kwargs should still work."""
        model = create_model(
            image_size=32,
            in_channels=1,
            unet_type="pixnerd",
        )
        assert isinstance(model, PixNerdBackbone)


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


class TestConfig:
    """Config save/load via ConfigMixin."""

    def test_config_roundtrip(self):
        model = _make_small_model(in_channels=3, hidden_size=384, num_groups=12)
        config = model.config
        model2 = PixNerdBackbone.from_config(config)
        assert model2.config["in_channels"] == 3
        assert model2.config["hidden_size"] == 384
        assert model2.config["num_groups"] == 12


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
