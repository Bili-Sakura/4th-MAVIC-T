"""Tests for scripts/filter_bad_samples.py and the exclude_file filtering in dataset wrappers."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# filter_bad_samples.is_bad_image
# ---------------------------------------------------------------------------

class TestIsBadImage:
    """Unit tests for the core ``is_bad_image`` check."""

    def _make_image(self, arr: np.ndarray, suffix: str = ".png") -> str:
        """Save *arr* as an image to a temp file and return its path."""
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        img = Image.fromarray(arr)
        img.save(tmp.name)
        tmp.close()
        return tmp.name

    def test_all_black_is_bad(self):
        from scripts.filter_bad_samples import is_bad_image
        path = self._make_image(np.zeros((32, 32), dtype=np.uint8))
        assert is_bad_image(path) is True
        os.unlink(path)

    def test_normal_image_is_good(self):
        from scripts.filter_bad_samples import is_bad_image
        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        arr[0, 0, 0] = 128  # guarantee at least one non-zero
        path = self._make_image(arr)
        assert is_bad_image(path) is False
        os.unlink(path)

    def test_single_bright_pixel_is_bad_with_black_patch(self):
        """A 32×32 image with one bright pixel still contains all-black 16×16 patches."""
        from scripts.filter_bad_samples import is_bad_image
        arr = np.zeros((32, 32), dtype=np.uint8)
        arr[16, 16] = 1
        path = self._make_image(arr)
        assert is_bad_image(path) is True
        os.unlink(path)

    def test_threshold(self):
        from scripts.filter_bad_samples import is_bad_image
        arr = np.ones((32, 32), dtype=np.uint8)  # max pixel = 1
        path = self._make_image(arr)
        # With default thresh=0.0 it's good (max=1 > 0)
        assert is_bad_image(path, black_thresh=0.0) is False
        # With thresh=1.0 it's bad (max=1 <= 1)
        assert is_bad_image(path, black_thresh=1.0) is True
        os.unlink(path)

    def test_nonexistent_file_is_bad(self):
        from scripts.filter_bad_samples import is_bad_image
        assert is_bad_image("/tmp/does_not_exist_12345.png") is True

    def test_all_nan_float_tiff_is_bad(self):
        from scripts.filter_bad_samples import is_bad_image
        arr = np.full((32, 32), np.nan, dtype=np.float32)
        path = self._make_image(arr, suffix=".tiff")
        assert is_bad_image(path) is True
        os.unlink(path)

    def test_one_black_patch_among_good_patches(self):
        """A 32×32 image where one 16×16 quadrant is all-black is bad."""
        from scripts.filter_bad_samples import is_bad_image
        arr = np.random.randint(10, 256, (32, 32, 3), dtype=np.uint8)
        arr[:16, :16, :] = 0  # top-left 16×16 patch is all-black
        path = self._make_image(arr)
        assert is_bad_image(path) is True
        os.unlink(path)

    def test_no_full_black_patch_is_good(self):
        """Every 16×16 patch has at least one non-zero pixel → good."""
        from scripts.filter_bad_samples import is_bad_image
        arr = np.zeros((32, 32), dtype=np.uint8)
        # Place one bright pixel in each of the four 16×16 patches.
        arr[0, 0] = 1     # patch (0:16, 0:16)
        arr[0, 16] = 1    # patch (0:16, 16:32)
        arr[16, 0] = 1    # patch (16:32, 0:16)
        arr[16, 16] = 1   # patch (16:32, 16:32)
        path = self._make_image(arr)
        assert is_bad_image(path) is False
        os.unlink(path)

    def test_nan_patch_in_float_image(self):
        """A float image with one 16×16 all-NaN patch is bad."""
        from scripts.filter_bad_samples import is_bad_image
        arr = np.random.rand(32, 32).astype(np.float32) + 1.0
        arr[16:32, 0:16] = np.nan  # bottom-left 16×16 patch is all-NaN
        path = self._make_image(arr, suffix=".tiff")
        assert is_bad_image(path) is True
        os.unlink(path)

    def test_custom_patch_size(self):
        """Patch size can be configured; a smaller patch triggers detection."""
        from scripts.filter_bad_samples import is_bad_image
        arr = np.random.randint(10, 256, (32, 32), dtype=np.uint8)
        arr[0:8, 0:8] = 0  # 8×8 black region
        path = self._make_image(arr)
        # With default 16×16 it should be good (8×8 is too small)
        assert is_bad_image(path, patch_size=16) is False
        # With 8×8 patch size it should be bad
        assert is_bad_image(path, patch_size=8) is True
        os.unlink(path)

    def test_image_smaller_than_patch_all_black(self):
        """An image smaller than patch_size that is all-black is bad."""
        from scripts.filter_bad_samples import is_bad_image
        arr = np.zeros((8, 8), dtype=np.uint8)
        path = self._make_image(arr)
        assert is_bad_image(path, patch_size=16) is True
        os.unlink(path)

    def test_image_smaller_than_patch_not_black(self):
        """An image smaller than patch_size with non-zero pixels is good."""
        from scripts.filter_bad_samples import is_bad_image
        arr = np.ones((8, 8), dtype=np.uint8) * 128
        path = self._make_image(arr)
        assert is_bad_image(path, patch_size=16) is False
        os.unlink(path)


# ---------------------------------------------------------------------------
# _load_exclude_set (shared by all 3 wrappers; test via the turbo one)
# ---------------------------------------------------------------------------

class TestLoadExcludeSet:
    """Tests for _load_exclude_set. Requires torch (heavy deps in dataset_wrapper)."""

    @pytest.fixture(autouse=True)
    def _skip_without_torch(self):
        pytest.importorskip("torch")

    def _load_exclude_set(self, exclude_file):
        """Import _load_exclude_set avoiding heavy __init__.py imports."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "dataset_wrapper",
            str(Path(__file__).resolve().parent.parent / "src" / "img2img_turbo" / "dataset_wrapper.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._load_exclude_set(exclude_file)

    def test_none_returns_empty(self):
        assert self._load_exclude_set(None) == set()

    def test_missing_file_returns_empty(self):
        assert self._load_exclude_set("/tmp/nonexistent_exclude_424242.txt") == set()

    def test_loads_paths(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("/some/path/a.tif\n")
            f.write("/some/path/b.tif\n")
            f.write("\n")  # blank line should be skipped
            fname = f.name
        try:
            result = self._load_exclude_set(fname)
            assert len(result) == 2
            # Paths are resolved, so check they end with the expected names
            assert any(p.endswith("a.tif") for p in result)
            assert any(p.endswith("b.tif") for p in result)
        finally:
            os.unlink(fname)
