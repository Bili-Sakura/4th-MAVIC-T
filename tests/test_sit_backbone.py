# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for the SiT backbone integration.

Covers:
* Module imports and public API
* SiTBackbone initialization and configuration
* Forward pass shapes (concat and unconditional modes)
* Factory integration (create_model with unet_type='sit')
* Config save/load round-trip (ModelMixin / ConfigMixin)
* Gradient flow
"""

import torch
import pytest

from src.models.dit.sit_backbone import SiTBackbone
from src.models.unet.unet_ddbm import (
    create_model,
    get_unet_type_config,
    SUPPORTED_DIT_TYPES,
    SUPPORTED_BACKBONE_TYPES,
    DIT_TYPE_SIT,
)


# ---------------------------------------------------------------------------
# Small model helper (fast tests)
# ---------------------------------------------------------------------------

def _make_small_model(**overrides):
    """Create a tiny SiTBackbone for fast testing."""
    defaults = dict(
        image_size=32,
        patch_size=2,
        in_channels=1,
        hidden_size=192,
        depth=4,
        num_heads=6,
        mlp_ratio=4.0,
        condition_mode="concat",
        dropout=0.0,
    )
    defaults.update(overrides)
    return SiTBackbone(**defaults)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


class TestImports:
    """All public symbols are importable."""

    def test_sit_backbone_class(self):
        assert SiTBackbone is not None

    def test_unet_type_constant(self):
        assert DIT_TYPE_SIT == "sit"

    def test_in_supported_types(self):
        assert DIT_TYPE_SIT in SUPPORTED_DIT_TYPES
        assert DIT_TYPE_SIT in SUPPORTED_BACKBONE_TYPES

    def test_get_unet_type_config(self):
        cfg = get_unet_type_config(DIT_TYPE_SIT)
        assert cfg["implemented"] is True
        assert "SiT" in cfg["description"]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """SiTBackbone initializes correctly."""

    def test_default_init(self):
        model = _make_small_model()
        assert isinstance(model, SiTBackbone)

    def test_stores_config(self):
        model = _make_small_model()
        assert model.config["image_size"] == 32
        assert model.config["in_channels"] == 1
        assert model.config["condition_mode"] == "concat"
        assert model.config["depth"] == 4
        assert model.config["hidden_size"] == 192
        assert model.config["num_heads"] == 6

    def test_custom_params(self):
        model = _make_small_model(
            in_channels=3,
            hidden_size=384,
            num_heads=12,
            depth=6,
        )
        assert model.hidden_size == 384
        assert model.num_heads == 12
        assert model.depth == 6

    def test_block_count(self):
        model = _make_small_model(depth=8)
        assert len(model.blocks) == 8

    def test_pos_embed_not_trainable(self):
        model = _make_small_model()
        assert not model.pos_embed.requires_grad

    def test_num_patches(self):
        model = _make_small_model(image_size=32, patch_size=2)
        # (32 / 2) ** 2 = 256 patches
        assert model.x_embedder.num_patches == 256

    def test_patch_size_4(self):
        model = _make_small_model(image_size=32, patch_size=4)
        # (32 / 4) ** 2 = 64 patches
        assert model.x_embedder.num_patches == 64


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

    def test_patch_size_4(self):
        model = _make_small_model(image_size=32, patch_size=4)
        x = torch.randn(2, 1, 32, 32)
        t = torch.randn(2)
        xT = torch.randn(2, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == (2, 1, 32, 32)

    def test_output_channels_match_in_channels(self):
        """Output channel count should match in_channels (no learn_sigma)."""
        for ch in [1, 3]:
            model = _make_small_model(in_channels=ch)
            x = torch.randn(1, ch, 32, 32)
            t = torch.randn(1)
            xT = torch.randn(1, ch, 32, 32)
            out = model(x, t, xT=xT)
            assert out.shape[1] == ch


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestFactory:
    """create_model factory correctly creates SiT models."""

    def test_create_sit(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            unet_type="sit",
            condition_mode="concat",
            sit_hidden_size=192,
            sit_depth=4,
            sit_num_heads=6,
            sit_patch_size=2,
        )
        assert isinstance(model, SiTBackbone)

    def test_factory_forward(self):
        model = create_model(
            image_size=32,
            in_channels=1,
            unet_type="sit",
            condition_mode="concat",
            sit_hidden_size=192,
            sit_depth=4,
            sit_num_heads=6,
            sit_patch_size=2,
        )
        x = torch.randn(2, 1, 32, 32)
        t = torch.randn(2)
        xT = torch.randn(2, 1, 32, 32)
        out = model(x, t, xT=xT)
        assert out.shape == (2, 1, 32, 32)

    def test_factory_defaults(self):
        """Factory with default sit kwargs should still work (image_size must be divisible by patch_size=2)."""
        model = create_model(
            image_size=32,
            in_channels=1,
            unet_type="sit",
        )
        assert isinstance(model, SiTBackbone)


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


class TestConfig:
    """Config save/load via ConfigMixin."""

    def test_config_roundtrip(self):
        model = _make_small_model(in_channels=3, hidden_size=384, num_heads=12, depth=6)
        config = model.config
        model2 = SiTBackbone.from_config(config)
        assert model2.config["in_channels"] == 3
        assert model2.config["hidden_size"] == 384
        assert model2.config["num_heads"] == 12
        assert model2.config["depth"] == 6


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
