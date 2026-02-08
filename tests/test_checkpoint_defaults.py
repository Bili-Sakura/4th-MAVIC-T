import pytest

from src.cut_baseline.config import TaskConfig as CutConfig
from src.ddbm_baseline.config import TaskConfig as DdbmConfig
from src.i2sb_baseline.config import TaskConfig as I2sbConfig
from src.img2img_turbo.config import TaskConfig as TurboConfig


@pytest.mark.parametrize("cfg_cls", [TurboConfig, DdbmConfig, CutConfig, I2sbConfig])
from src.ddib_baseline.config import TaskConfig as DdibConfig
from src.img2img_turbo.config import TaskConfig as TurboConfig


@pytest.mark.parametrize("cfg_cls", [TurboConfig, DdbmConfig, CutConfig, DdibConfig])
def test_default_checkpoint_settings(cfg_cls):
    cfg = cfg_cls()
    assert cfg.save_model_epochs == 1
    assert cfg.checkpointing_steps is None
    assert cfg.checkpoints_total_limit == 1
    assert cfg.push_to_hub is True
