"""Tests for validation logging feature across all baseline trainers.

Validates:
- Config fields ``validation_epochs`` and ``validation_steps`` exist and default to None
- ``log_validation()`` method exists on all trainer classes
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

from examples.ddbm.config import TaskConfig as DdbmConfig
from examples.dbim.config import TaskConfig as DbimConfig
from examples.bibbdm.config import TaskConfig as BibbdmConfig
from examples.bdbm.config import TaskConfig as BdbmConfig
from examples.i2sb.config import TaskConfig as I2sbConfig
from examples.ddib.config import TaskConfig as DdibConfig
from examples.cut.config import TaskConfig as CutConfig
from examples.img2img_turbo.config import TaskConfig as TurboConfig


ALL_CONFIGS = [DdbmConfig, DbimConfig, BibbdmConfig, BdbmConfig, I2sbConfig, DdibConfig, CutConfig, TurboConfig]


@pytest.mark.parametrize("cfg_cls", ALL_CONFIGS)
class TestValidationConfigDefaults:
    def test_validation_epochs_default_none(self, cfg_cls):
        cfg = cfg_cls()
        assert cfg.validation_epochs is None

    def test_validation_steps_default_none(self, cfg_cls):
        cfg = cfg_cls()
        assert cfg.validation_steps is None

    def test_validation_epochs_overridable(self, cfg_cls):
        cfg = cfg_cls()
        cfg.validation_epochs = 5
        assert cfg.validation_epochs == 5

    def test_validation_steps_overridable(self, cfg_cls):
        cfg = cfg_cls()
        cfg.validation_steps = 100
        assert cfg.validation_steps == 100


# ---------------------------------------------------------------------------
# Trainer has log_validation method
# ---------------------------------------------------------------------------

from examples.ddbm.trainer import DDBMTrainer
from examples.dbim.trainer import DBIMTrainer
from examples.bibbdm.trainer import BiBBDMTrainer
from examples.bdbm.trainer import BDBMTrainer
from examples.i2sb.trainer import I2SBTrainer
from examples.ddib.trainer import DDIBTrainer
from examples.cut.trainer import CUTTrainer
from examples.img2img_turbo.trainer import Pix2PixTurboTrainer


class TestLogValidationMethodExists:
    """Verify each trainer class exposes a ``log_validation`` method."""

    def test_ddbm_has_log_validation(self):
        assert callable(getattr(DDBMTrainer, "log_validation", None))

    def test_dbim_has_log_validation(self):
        assert callable(getattr(DBIMTrainer, "log_validation", None))

    def test_bibbdm_has_log_validation(self):
        assert callable(getattr(BiBBDMTrainer, "log_validation", None))

    def test_bdbm_has_log_validation(self):
        assert callable(getattr(BDBMTrainer, "log_validation", None))

    def test_i2sb_has_log_validation(self):
        assert callable(getattr(I2SBTrainer, "log_validation", None))

    def test_ddib_has_log_validation(self):
        assert callable(getattr(DDIBTrainer, "log_validation", None))

    def test_cut_has_log_validation(self):
        assert callable(getattr(CUTTrainer, "log_validation", None))

    def test_turbo_has_log_validation(self):
        assert callable(getattr(Pix2PixTurboTrainer, "log_validation", None))
