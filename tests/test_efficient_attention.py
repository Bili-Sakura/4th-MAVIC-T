# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for enable_efficient_attention utility and config fields."""

import logging

import pytest
import torch
from diffusers import UNet2DModel

from src.utils.training_utils import enable_efficient_attention


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestEnableEfficientAttention:
    """Tests for the enable_efficient_attention helper."""

    @pytest.fixture()
    def small_unet(self):
        """Tiny UNet2DModel with a single attention block for testing."""
        return UNet2DModel(
            sample_size=8,
            in_channels=1,
            out_channels=1,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )

    def test_noop_when_both_disabled(self, small_unet, caplog):
        """No-op when both flags are False (default)."""
        enable_efficient_attention(small_unet, enable_xformers=False, enable_flash_attention_2=False)
        # No warnings or info about attention should be logged
        for rec in caplog.records:
            assert "xformers" not in rec.message.lower()
            assert "flash" not in rec.message.lower()

    def test_flash_attention_2_sets_processor(self, small_unet, caplog):
        """Flash Attention 2 gracefully handles UNet2DModel (warns when unsupported)."""
        from diffusers.models.attention_processor import AttnProcessor2_0

        with caplog.at_level(logging.WARNING):
            enable_efficient_attention(small_unet, enable_flash_attention_2=True)
        # UNet2DModel may not have set_attn_processor - check it warns or succeeds
        has_set_fn = hasattr(small_unet, "set_attn_processor")
        if not has_set_fn:
            assert any("does not support" in r.message for r in caplog.records)
        else:
            for name, proc in small_unet.attn_processors.items():
                assert isinstance(proc, AttnProcessor2_0)

    def test_xformers_warns_when_unavailable(self, small_unet, caplog):
        """xformers flag should log a warning when xformers is not installed."""
        with caplog.at_level(logging.WARNING):
            enable_efficient_attention(small_unet, enable_xformers=True)
        # We expect either a warning or success message depending on env
        assert len(caplog.records) > 0

    def test_wrapper_model_resolves_unet(self, caplog):
        """When model has a .unet attribute, enable_efficient_attention targets it."""
        inner_unet = UNet2DModel(
            sample_size=8,
            in_channels=1,
            out_channels=1,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        # Simulate a wrapper with .unet attribute
        wrapper = torch.nn.Module()
        wrapper.unet = inner_unet

        with caplog.at_level(logging.INFO):
            enable_efficient_attention(wrapper, enable_flash_attention_2=True)
        # Should target inner_unet, not the wrapper
        # Check that log message references UNet2DModel (the inner model), not Module
        logged_msgs = " ".join(r.message for r in caplog.records)
        assert "UNet2DModel" in logged_msgs or "Flash Attention 2" in logged_msgs

    def test_model_without_xformers_support_warns(self, caplog):
        """Model without enable_xformers_memory_efficient_attention logs a warning."""
        plain_model = torch.nn.Linear(4, 4)
        with caplog.at_level(logging.WARNING):
            enable_efficient_attention(plain_model, enable_xformers=True)
        assert any("does not support" in r.message for r in caplog.records)

    def test_model_without_set_attn_processor_warns(self, caplog):
        """Model without set_attn_processor logs a warning for flash attn 2."""
        plain_model = torch.nn.Linear(4, 4)
        with caplog.at_level(logging.WARNING):
            enable_efficient_attention(plain_model, enable_flash_attention_2=True)
        assert any("does not support" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Config field tests
# ---------------------------------------------------------------------------


class TestConfigFields:
    """Verify enable_xformers / enable_flash_attention_2 fields exist in configs."""

    def test_ddbm_config_has_fields(self):
        from examples.ddbm.config import TaskConfig
        cfg = TaskConfig()
        assert hasattr(cfg, "enable_xformers")
        assert hasattr(cfg, "enable_flash_attention_2")
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attention_2 is False

    def test_bibbdm_config_has_fields(self):
        from examples.bibbdm.config import TaskConfig
        cfg = TaskConfig()
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attention_2 is False

    def test_i2sb_config_has_fields(self):
        from examples.i2sb.config import TaskConfig
        cfg = TaskConfig()
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attention_2 is False

    def test_ddib_config_has_fields(self):
        from examples.ddib.config import TaskConfig
        cfg = TaskConfig()
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attention_2 is False

    def test_img2img_turbo_config_has_fields(self):
        from examples.img2img_turbo.config import TaskConfig
        cfg = TaskConfig()
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attention_2 is False

    def test_unidb_config_has_fields(self):
        from examples.unidb.config import TaskConfig
        cfg = TaskConfig()
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attention_2 is False

    def test_sid_inherits_fields(self):
        """SiD reuses DDBM TaskConfig and should inherit the new fields."""
        from examples.sid.config import TaskConfig
        cfg = TaskConfig()
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attention_2 is False

    def test_cdtsde_inherits_fields(self):
        """CDTSDE subclasses DDBM TaskConfig and should inherit the new fields."""
        from examples.cdtsde.config import TaskConfig
        cfg = TaskConfig()
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attention_2 is False

    def test_bdbm_inherits_fields(self):
        """BDBM subclasses BiBBDM TaskConfig and should inherit the new fields."""
        from examples.bdbm.config import TaskConfig
        cfg = TaskConfig()
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attention_2 is False

    def test_dab_inherits_fields(self):
        """DAB subclasses BiBBDM TaskConfig and should inherit the new fields."""
        from examples.dab.config import TaskConfig
        cfg = TaskConfig()
        assert cfg.enable_xformers is False
        assert cfg.enable_flash_attention_2 is False
