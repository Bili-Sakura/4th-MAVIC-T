"""Tests for runtime random-crop training behavior across baselines."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from examples.ddbm.dataset_wrapper import MavicTDDBMDataset
from examples.dbim.dataset_wrapper import MavicTDBIMDataset
from examples.ddib.dataset_wrapper import MavicTDDIBDataset


def _write_rgb_pattern(path: Path, size: int = 1024) -> None:
    yy, xx = np.indices((size, size), dtype=np.int32)
    arr = np.stack(
        [
            (xx % 256).astype(np.uint8),
            (yy % 256).astype(np.uint8),
            ((xx + yy) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    Image.fromarray(arr, mode="RGB").save(path)


def test_ddbm_runtime_random_crop_is_reproducible_and_aligned(tmp_path: Path):
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    _write_rgb_pattern(src)
    _write_rgb_pattern(tgt)

    class DummyLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, split, task, with_target=True, load_images=False):
            return [{"input_path": str(src), "target_path": str(tgt)}]

    with patch("examples.ddbm.dataset_wrapper.MavicTImageToImageDataset", DummyLoader):
        ds = MavicTDDBMDataset(
            task="rgb2ir",
            split="train",
            resolution=512,
            source_channels=3,
            target_channels=3,
            use_random_crop=True,
        )

        torch.manual_seed(2026)
        tgt_a, src_a = ds[0]
        torch.manual_seed(2026)
        tgt_b, src_b = ds[0]
        torch.manual_seed(2027)
        _, src_c = ds[0]

    assert src_a.shape == (3, 512, 512)
    assert tgt_a.shape == (3, 512, 512)
    assert torch.allclose(src_a, src_b)
    assert torch.allclose(tgt_a, tgt_b)
    # Same random crop should be applied to source and target.
    assert torch.allclose(src_a, tgt_a)
    # Different RNG seed should very likely produce a different crop.
    assert not torch.allclose(src_a, src_c)


def test_dbim_dataset_reuses_ddbm_runtime_crop(tmp_path: Path):
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    _write_rgb_pattern(src)
    _write_rgb_pattern(tgt)

    class DummyLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, split, task, with_target=True, load_images=False):
            return [{"input_path": str(src), "target_path": str(tgt)}]

    with patch("examples.ddbm.dataset_wrapper.MavicTImageToImageDataset", DummyLoader):
        ds = MavicTDBIMDataset(
            task="sar2ir",
            split="train",
            resolution=512,
            source_channels=3,
            target_channels=3,
            use_random_crop=True,
        )
        torch.manual_seed(77)
        tgt_img, src_img = ds[0]

    assert src_img.shape == (3, 512, 512)
    assert tgt_img.shape == (3, 512, 512)
    assert torch.allclose(src_img, tgt_img)


def test_ddib_runtime_random_crop_single_domain(tmp_path: Path):
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    _write_rgb_pattern(src)
    _write_rgb_pattern(tgt)

    class DummyLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, split, task, with_target=True, load_images=False):
            return [{"input_path": str(src), "target_path": str(tgt)}]

    with patch("examples.ddib.dataset_wrapper.MavicTImageToImageDataset", DummyLoader):
        ds = MavicTDDIBDataset(
            task="sar2eo",
            domain="source",
            split="train",
            resolution=512,
            model_channels=3,
            use_random_crop=True,
        )
        torch.manual_seed(9)
        img_a = ds[0]
        torch.manual_seed(9)
        img_b = ds[0]

    assert img_a.shape == (3, 512, 512)
    assert torch.allclose(img_a, img_b)


def test_example_configs_default_to_512_runtime_crop():
    from examples.ddbm.config import sar2eo_config as ddbm_sar2eo, rgb2ir_config as ddbm_rgb2ir
    from examples.dbim.config import sar2eo_config as dbim_sar2eo
    from examples.ddib.config import sar2eo_config as ddib_sar2eo
    from examples.bibbdm.config import sar2eo_config as bibbdm_sar2eo
    from examples.i2sb.config import sar2eo_config as i2sb_sar2eo
    from examples.cut.config import sar2eo_config as cut_sar2eo
    from examples.unidb.config import sar2eo_config as unidb_sar2eo
    from examples.img2img_turbo.config import sar2eo_config as turbo_sar2eo

    cfgs = [
        ddbm_sar2eo(),
        ddbm_rgb2ir(),
        dbim_sar2eo(),
        ddib_sar2eo(),
        bibbdm_sar2eo(),
        i2sb_sar2eo(),
        cut_sar2eo(),
        unidb_sar2eo(),
        turbo_sar2eo(),
    ]
    for cfg in cfgs:
        assert cfg.resolution == 512
        assert getattr(cfg, "use_random_crop", True) is True
