# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for latent_channels config field across all baselines.

Verifies that each baseline's TaskConfig has the latent_channels field with
the correct default (32) and that it can be overridden. Also tests that
pre-built task configs inherit the default.
"""

import pytest

from examples.ddbm.config import TaskConfig as DdbmConfig
from examples.ddbm.config import sar2eo_config as ddbm_sar2eo
from examples.dbim.config import TaskConfig as DbimConfig
from examples.dbim.config import sar2eo_config as dbim_sar2eo
from examples.cdtsde.config import TaskConfig as CdtsdeConfig
from examples.cdtsde.config import sar2eo_config as cdtsde_sar2eo
from examples.bibbdm.config import TaskConfig as BibbdmConfig
from examples.bibbdm.config import sar2eo_config as bibbdm_sar2eo
from examples.bdbm.config import TaskConfig as BdbmConfig
from examples.bdbm.config import sar2eo_config as bdbm_sar2eo
from examples.i2sb.config import TaskConfig as I2sbConfig
from examples.i2sb.config import sar2eo_config as i2sb_sar2eo
from examples.ddib.config import TaskConfig as DdibConfig
from examples.ddib.config import sar2eo_config as ddib_sar2eo
from examples.cut.config import TaskConfig as CutConfig
from examples.cut.config import sar2eo_config as cut_sar2eo
from examples.img2img_turbo.config import TaskConfig as TurboConfig


# ---------------------------------------------------------------------------
# Default value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_cls", [DdbmConfig, DbimConfig, CdtsdeConfig, BibbdmConfig, BdbmConfig, I2sbConfig, DdibConfig, CutConfig, TurboConfig])
def test_latent_channels_default_is_32(cfg_cls):
    """All baselines default latent_channels to 32 (VAE latent dim)."""
    cfg = cfg_cls()
    assert cfg.latent_channels == 32


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_cls", [DdbmConfig, DbimConfig, CdtsdeConfig, BibbdmConfig, BdbmConfig, I2sbConfig, DdibConfig, CutConfig, TurboConfig])
def test_latent_channels_override(cfg_cls):
    """latent_channels can be overridden at construction time."""
    cfg = cfg_cls(latent_channels=16)
    assert cfg.latent_channels == 16


# ---------------------------------------------------------------------------
# Pre-built task configs inherit default
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg_fn", [ddbm_sar2eo, dbim_sar2eo, cdtsde_sar2eo, bibbdm_sar2eo, bdbm_sar2eo, i2sb_sar2eo, ddib_sar2eo, cut_sar2eo])
def test_task_config_has_latent_channels(cfg_fn):
    """Pre-built task configs carry the default latent_channels=32."""
    cfg = cfg_fn()
    assert cfg.latent_channels == 32


@pytest.mark.parametrize("cfg_fn", [ddbm_sar2eo, dbim_sar2eo, cdtsde_sar2eo, bibbdm_sar2eo, bdbm_sar2eo, i2sb_sar2eo, ddib_sar2eo, cut_sar2eo])
def test_task_config_latent_channels_overridable(cfg_fn):
    """Pre-built task configs allow latent_channels override via kwargs."""
    cfg = cfg_fn(latent_channels=4)
    assert cfg.latent_channels == 4


# ---------------------------------------------------------------------------
# Pixel-mode channels unchanged
# ---------------------------------------------------------------------------

def test_pixel_mode_model_channels_unchanged():
    """When use_latent_target is False, model_channels is still the pixel count."""
    cfg = DdbmConfig(model_channels=1, use_latent_target=False)
    assert cfg.model_channels == 1
    # latent_channels exists but is not used unless use_latent_target=True
    assert cfg.latent_channels == 32
