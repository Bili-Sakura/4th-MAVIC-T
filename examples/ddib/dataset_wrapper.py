"""PyTorch Dataset wrapper for DDIB single-domain training.

DDIB trains *unconditional* diffusion models on each domain independently.
This module bridges the HuggingFace ``datasets.Dataset`` returned by
:class:`src.utils.mavic_t_dataset.MavicTImageToImageDataset` and the PyTorch
:class:`torch.utils.data.Dataset` interface expected by the DDIB trainer.

It handles:
* Loading images from a *single* domain (source **or** target).
* Resizing to the model resolution.
* Channel adaptation (expanding/repeating channels to ``model_channels``).
* Normalising pixel values to [0, 1] as float32 tensors.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Set

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.mavic_t_dataset import MavicTImageToImageDataset  # noqa: E402


def _load_image_as_tensor(path: str, channels: int, resolution: int) -> torch.Tensor:
    """Load an image from *path*, resize, and return a ``(C, H, W)`` float32 tensor in [0, 1]."""
    img = Image.open(path)
    img = img.resize((resolution, resolution), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)

    if arr.max() > 1.0:
        if arr.dtype == np.float32 and arr.max() > 255.0:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    c = arr.shape[2]
    if c < channels:
        arr = np.repeat(arr, channels // c + 1, axis=2)[:, :, :channels]
    elif c > channels:
        arr = arr[:, :, :channels]

    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def _load_exclude_set(exclude_file: Optional[str]) -> Set[str]:
    """Load a set of absolute paths to exclude from training."""
    if not exclude_file:
        return set()
    path = Path(exclude_file)
    if not path.is_file():
        return set()
    out: Set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.add(str(Path(line).resolve()))
    return out


class MavicTDDIBDataset(Dataset):
    """A PyTorch :class:`Dataset` that loads MAVIC-T images for DDIB single-domain training.

    Unlike :class:`MavicTDDBMDataset` which returns ``(target, source)`` pairs,
    this dataset returns **single images** from one domain – either the source or
    the target – for unconditional diffusion training.

    Parameters
    ----------
    task : str
        One of ``sar2eo``, ``rgb2ir``, ``sar2ir``, ``sar2rgb``.
    domain : str
        ``"source"`` or ``"target"`` — which domain to load.
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    resolution : int
        Spatial resolution to resize images to.
    model_channels : int
        Number of channels the model operates in.
    use_augmented : bool
        If ``True`` and ``split == "train"``, also include the ``*_crop_aug``
        variant.
    use_horizontal_flip : bool
        Random horizontal flip augmentation for training.
    use_vertical_flip : bool
        Random vertical flip augmentation for training.
    exclude_file : str or None
        Path to a text file listing bad sample paths to skip.
    """

    def __init__(
        self,
        task: str,
        domain: str = "source",
        split: str = "train",
        resolution: int = 256,
        model_channels: int = 3,
        use_augmented: bool = False,
        use_horizontal_flip: bool = False,
        use_vertical_flip: bool = False,
        refined_root: Optional[str] = None,
        eval_root: Optional[str] = None,
        exclude_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.domain = domain
        self.split = split
        self.resolution = resolution
        self.model_channels = model_channels
        self.use_horizontal_flip = use_horizontal_flip and split == "train"
        self.use_vertical_flip = use_vertical_flip and split == "train"

        with_target = domain == "target" or split == "train"

        kwargs = {}
        if refined_root is not None:
            kwargs["refined_root"] = refined_root
        if eval_root is not None:
            kwargs["eval_root"] = eval_root
        loader = MavicTImageToImageDataset(**kwargs)

        ds = loader.load(split=split, task=task, with_target=with_target)
        self._records = list(ds)

        if use_augmented and split == "train" and not task.endswith("_crop_aug"):
            aug_task = f"{task}_crop_aug"
            try:
                ds_aug = loader.load(split="train", task=aug_task, with_target=with_target)
                self._records.extend(list(ds_aug))
            except (ValueError, FileNotFoundError):
                pass

        # Determine which path column to use
        self._path_key = "target_path" if domain == "target" else "input_path"

        # Filter out excluded samples
        exclude = _load_exclude_set(exclude_file)
        if exclude:
            before = len(self._records)
            self._records = [
                r for r in self._records
                if str(Path(r[self._path_key]).resolve()) not in exclude
            ]
            after = len(self._records)
            if before != after:
                logging.getLogger(__name__).info(
                    f"Excluded {before - after} samples via {exclude_file} "
                    f"({after} remaining)"
                )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single image tensor in [0, 1]."""
        rec = self._records[idx]
        img = _load_image_as_tensor(rec[self._path_key], self.model_channels, self.resolution)

        if self.use_horizontal_flip and torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
        if self.use_vertical_flip and torch.rand(1).item() > 0.5:
            img = TF.vflip(img)

        return img
