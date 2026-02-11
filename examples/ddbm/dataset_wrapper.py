"""PyTorch Dataset wrapper around ``MavicTImageToImageDataset``.

This module bridges the HuggingFace ``datasets.Dataset`` returned by
:class:`src.utils.mavic_t_dataset.MavicTImageToImageDataset` and the PyTorch
:class:`torch.utils.data.Dataset` interface expected by DDBM training.

It handles:
* Reading images lazily from disk (via the path columns).
* Resizing to the model resolution.
* Channel adaptation (expanding/repeating channels to ``model_channels``).
* Normalising pixel values to [0, 1] as float32 tensors.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

from typing import Set

# Ensure the project root is importable so we can import from ``src``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.mavic_t_dataset import MavicTImageToImageDataset  # noqa: E402


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
    use_horizontal_flip : bool
        If ``True`` and ``split == "train"``, randomly apply horizontal flip
        (applied consistently to both source and target).
    use_vertical_flip : bool
        If ``True`` and ``split == "train"``, randomly apply vertical flip
        (applied consistently to both source and target).
    refined_root, eval_root : str or Path or None
        Forwarded to :class:`MavicTImageToImageDataset`.
    """

    def __init__(
        self,
        task: str,
        split: str = "train",
        resolution: int = 256,
        source_channels: Optional[int] = None,
        target_channels: Optional[int] = None,
        model_channels: int = 3,
        with_target: Optional[bool] = None,
        use_augmented: bool = False,
        use_horizontal_flip: bool = False,
        use_vertical_flip: bool = False,
        refined_root: Optional[str] = None,
        eval_root: Optional[str] = None,
        exclude_file: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.split = split
        self.resolution = resolution
        self.source_channels = source_channels or model_channels
        self.target_channels = target_channels or model_channels
        self.use_horizontal_flip = use_horizontal_flip and split == "train"
        self.use_vertical_flip = use_vertical_flip and split == "train"

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
        ds = loader.load(split=split, task=task, with_target=with_target, load_images=False)
        self._records = list(ds)

        # Optionally add augmented samples for training
        if use_augmented and split == "train" and not task.endswith("_crop_aug"):
            aug_task = f"{task}_crop_aug"
            try:
                ds_aug = loader.load(split="train", task=aug_task, with_target=with_target, load_images=False)
                self._records.extend(list(ds_aug))
            except (ValueError, FileNotFoundError):
                pass  # augmented variant may not exist for all tasks

        # Filter out excluded samples
        exclude = _load_exclude_set(exclude_file)
        if exclude:
            before = len(self._records)
            self._records = [
                r for r in self._records
                if str(Path(r["input_path"]).resolve()) not in exclude
                and (not with_target or str(Path(r["target_path"]).resolve()) not in exclude)
            ]
            after = len(self._records)
            if before != after:
                logging.getLogger(__name__).info(
                    f"Excluded {before - after} samples via {exclude_file} "
                    f"({after} remaining)"
                )

    def __len__(self) -> int:
        return len(self._records)

    def get_output_name(self, idx: int) -> str:
        """Return the output filename for the given index (submission format: stem.png)."""
        rec = self._records[idx]
        stem = Path(rec["input_path"]).stem
        return f"{stem}.png"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(target, source)`` tensors in [0, 1].

        For val/test splits where no target is available the first element is a
        zero tensor with the correct shape.
        """
        rec = self._records[idx]

        source = _load_image_as_tensor(rec["input_path"], self.source_channels, self.resolution)

        if self.with_target:
            target = _load_image_as_tensor(rec["target_path"], self.target_channels, self.resolution)
        else:
            target = torch.zeros(self.target_channels, self.resolution, self.resolution)

        # Apply random flip augmentations consistently to both source and target
        if self.use_horizontal_flip and torch.rand(1).item() > 0.5:
            source = TF.hflip(source)
            target = TF.hflip(target)
        if self.use_vertical_flip and torch.rand(1).item() > 0.5:
            source = TF.vflip(source)
            target = TF.vflip(target)

        return target, source


class PairedValDataset(Dataset):
    """Dataset that loads (source, target) pairs from a paired validation manifest.

    Manifest format: one line per pair, ``input_path\\ttarget_path`` (tab-separated).
    Returns ``(target, source)`` tensors in [0, 1] to match MavicTDDBMDataset.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        resolution: int,
        source_channels: int,
        target_channels: int,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.source_channels = source_channels
        self.target_channels = target_channels
        self._pairs: list[tuple[str, str]] = []
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError(f"Paired val manifest not found: {path}")
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                inp, tgt = parts[0].strip(), parts[1].strip()
                if inp and tgt:
                    self._pairs.append((inp, tgt))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp_path, tgt_path = self._pairs[idx]
        source = _load_image_as_tensor(inp_path, self.source_channels, self.resolution)
        target = _load_image_as_tensor(tgt_path, self.target_channels, self.resolution)
        return target, source
