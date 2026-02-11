import pytest

from examples.cut.config import TaskConfig as CutConfig
from examples.ddbm.config import TaskConfig as DdbmConfig
from examples.i2sb.config import TaskConfig as I2sbConfig
from examples.ddib.config import TaskConfig as DdibConfig
from examples.img2img_turbo.config import TaskConfig as TurboConfig


@pytest.mark.parametrize("cfg_cls", [TurboConfig, DdbmConfig, CutConfig, I2sbConfig, DdibConfig])
def test_default_checkpoint_settings(cfg_cls):
    cfg = cfg_cls()
    assert cfg.save_model_epochs == 1
    assert cfg.checkpointing_steps is None
    assert cfg.checkpoints_total_limit == 1
    assert cfg.push_to_hub is True
