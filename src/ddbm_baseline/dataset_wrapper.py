"""PyTorch Dataset wrapper around ``MavicTImageToImageDataset``.

This module bridges the HuggingFace ``datasets.Dataset`` returned by
:class:`src.mavic_t_dataset.MavicTImageToImageDataset` and the PyTorch
:class:`torch.utils.data.Dataset` interface expected by DDBM training.

It handles:
* Reading images lazily from disk (via the path columns).
* Resizing to the model resolution.
* Channel adaptation (expanding/repeating channels to ``model_channels``).
* Normalising pixel values to [0, 1] as float32 tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Ensure the project root is importable so we can import from ``src``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.mavic_t_dataset import MavicTImageToImageDataset  # noqa: E402


def _load_image_as_tensor(path: str, channels: int, resolution: int) -> torch.Tensor:
    """Load an image from *path*, resize, and return a ``(C, H, W)`` float32 tensor in [0, 1]."""
    img = Image.open(path)
    img = img.resize((resolution, resolution), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    # Normalise to [0, 1]
    if arr.max() > 1.0:
        if arr.dtype == np.float32 and arr.max() > 255.0:
            arr = arr / 65535.0  # uint16 TIFF
        else:
            arr = arr / 255.0

    # Ensure 3-D: (H, W, C)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    # Channel adaptation
    c = arr.shape[2]
    if c < channels:
        arr = np.repeat(arr, channels // c + 1, axis=2)[:, :, :channels]
    elif c > channels:
        arr = arr[:, :, :channels]

    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)
    return tensor


class MavicTDDBMDataset(Dataset):
    """A PyTorch :class:`Dataset` that loads MAVIC-T image pairs for DDBM.

    Each ``__getitem__`` returns ``(target_tensor, source_tensor)`` following the
    DDBM convention where ``x0 = target`` and ``x_T = source``.  Both tensors
    are in ``[0, 1]`` range with shape ``(model_channels, resolution, resolution)``.

    Parameters
    ----------
    task : str
        One of ``sar2eo``, ``rgb2ir``, ``sar2ir``, ``sar2rgb``.
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    resolution : int
        Spatial resolution to resize images to.
    model_channels : int
        Number of channels the model operates in.
    with_target : bool or None
        Whether to load target images.  Defaults to ``True`` for train, ``False``
        for val/test.
    use_augmented : bool
        If ``True`` and ``split == "train"``, also include the ``*_crop_aug``
        variant as additional samples.
    refined_root, eval_root : str or Path or None
        Forwarded to :class:`MavicTImageToImageDataset`.
    """

    def __init__(
        self,
        task: str,
        split: str = "train",
        resolution: int = 256,
        model_channels: int = 3,
        with_target: Optional[bool] = None,
        use_augmented: bool = False,
        refined_root: Optional[str] = None,
        eval_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.split = split
        self.resolution = resolution
        self.model_channels = model_channels

        if with_target is None:
            with_target = split == "train"
        self.with_target = with_target

        kwargs = {}
        if refined_root is not None:
            kwargs["refined_root"] = refined_root
        if eval_root is not None:
            kwargs["eval_root"] = eval_root
        loader = MavicTImageToImageDataset(**kwargs)

        # Load the base dataset
        ds = loader.load(split=split, task=task, with_target=with_target)
        self._records = list(ds)

        # Optionally add augmented samples for training
        if use_augmented and split == "train" and not task.endswith("_crop_aug"):
            aug_task = f"{task}_crop_aug"
            try:
                ds_aug = loader.load(split="train", task=aug_task, with_target=with_target)
                self._records.extend(list(ds_aug))
            except (ValueError, FileNotFoundError):
                pass  # augmented variant may not exist for all tasks

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(target, source)`` tensors in [0, 1].

        For val/test splits where no target is available the first element is a
        zero tensor with the correct shape.
        """
        rec = self._records[idx]

        source = _load_image_as_tensor(rec["input_path"], self.model_channels, self.resolution)

        if self.with_target:
            target = _load_image_as_tensor(rec["target_path"], self.model_channels, self.resolution)
        else:
            target = torch.zeros_like(source)

        return target, source
