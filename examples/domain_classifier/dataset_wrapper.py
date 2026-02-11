"""Dataset helpers for domain real/fake classifier training and scoring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
import math
import random
from pathlib import Path
import sys
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

# Ensure project root is importable when executed as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.mavic_t_dataset import MavicTImageToImageDataset  # noqa: E402
from .config import DomainClassifierConfig  # noqa: E402


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinaryImageRecord:
    """Single image record for binary real/fake classification."""

    image_path: str
    label: int  # 1 = real, 0 = fake
    task: str
    role: str  # e.g. "target", "source", "target_negative"


def _resolve_optional_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return _PROJECT_ROOT / path


def _load_exclude_set(exclude_file: Optional[str]) -> set[str]:
    path = _resolve_optional_path(exclude_file)
    if path is None or not path.is_file():
        return set()

    out: set[str] = set()
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.add(str(Path(line).resolve()))
    return out


def _append_record(
    records: list[BinaryImageRecord],
    seen: set[tuple[str, int]],
    *,
    image_path: str,
    label: int,
    task: str,
    role: str,
    exclude_set: set[str],
) -> None:
    resolved = str(Path(image_path).resolve())
    if resolved in exclude_set:
        return
    key = (resolved, int(label))
    if key in seen:
        return
    seen.add(key)
    records.append(
        BinaryImageRecord(
            image_path=resolved,
            label=int(label),
            task=task,
            role=role,
        )
    )


def build_binary_records(cfg: DomainClassifierConfig) -> list[BinaryImageRecord]:
    """Build binary train records from MAVIC-T refined train + crop_aug manifests.

    Positive class (label=1):
        target-domain images from ``target_path``.

    Negative class (label=0), configurable:
        - source images from ``input_path`` of the same tasks, and/or
        - target images from additional tasks listed in ``negative_target_tasks_csv``.
    """

    positive_tasks = cfg.resolved_positive_tasks()
    if not positive_tasks:
        raise ValueError(
            f"No positive tasks found for target domain '{cfg.resolved_target_domain()}'. "
            "Provide --positive_tasks_csv to define tasks explicitly."
        )

    if cfg.negative_to_positive_ratio <= 0:
        raise ValueError(
            f"negative_to_positive_ratio must be > 0, got {cfg.negative_to_positive_ratio}"
        )

    negative_target_tasks = cfg.resolved_negative_target_tasks()
    if not cfg.use_source_as_fake and not negative_target_tasks:
        raise ValueError(
            "No fake samples configured: set use_source_as_fake=True or provide "
            "--negative_target_tasks_csv."
        )

    loader_kwargs = {}
    if cfg.refined_root:
        loader_kwargs["refined_root"] = cfg.refined_root
    loader = MavicTImageToImageDataset(**loader_kwargs)

    exclude_set = _load_exclude_set(cfg.exclude_file)
    records: list[BinaryImageRecord] = []
    seen: set[tuple[str, int]] = set()

    for task in positive_tasks:
        dataset = loader.load(
            split="train",
            task=task,
            with_target=True,
            load_images=False,
        )
        for row in dataset:
            _append_record(
                records,
                seen,
                image_path=row["target_path"],
                label=1,
                task=task,
                role="target",
                exclude_set=exclude_set,
            )
            if cfg.use_source_as_fake:
                _append_record(
                    records,
                    seen,
                    image_path=row["input_path"],
                    label=0,
                    task=task,
                    role="source",
                    exclude_set=exclude_set,
                )

    for task in negative_target_tasks:
        dataset = loader.load(
            split="train",
            task=task,
            with_target=True,
            load_images=False,
        )
        for row in dataset:
            _append_record(
                records,
                seen,
                image_path=row["target_path"],
                label=0,
                task=task,
                role="target_negative",
                exclude_set=exclude_set,
            )

    positives = [r for r in records if r.label == 1]
    negatives = [r for r in records if r.label == 0]
    if not positives:
        raise RuntimeError("No positive records were collected.")
    if not negatives:
        raise RuntimeError("No negative records were collected.")

    max_negatives = max(1, int(len(positives) * cfg.negative_to_positive_ratio))
    if len(negatives) > max_negatives:
        rng = random.Random(cfg.seed)
        negatives = rng.sample(negatives, max_negatives)

    merged = positives + negatives
    random.Random(cfg.seed).shuffle(merged)
    return merged


def split_binary_records(
    records: list[BinaryImageRecord],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[BinaryImageRecord], list[BinaryImageRecord]]:
    """Split into train/val with per-class stratification."""

    if val_ratio <= 0:
        return records, []
    if val_ratio >= 1:
        raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")

    rng = random.Random(seed)
    positives = [r for r in records if r.label == 1]
    negatives = [r for r in records if r.label == 0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    val_pos = max(1, int(len(positives) * val_ratio)) if len(positives) > 1 else 0
    val_neg = max(1, int(len(negatives) * val_ratio)) if len(negatives) > 1 else 0

    val_records = positives[:val_pos] + negatives[:val_neg]
    train_records = positives[val_pos:] + negatives[val_neg:]
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def summarize_binary_records(records: list[BinaryImageRecord]) -> dict[str, object]:
    labels = Counter(r.label for r in records)
    tasks = Counter(r.task for r in records)
    roles = Counter(r.role for r in records)
    return {
        "num_records": len(records),
        "num_real": labels.get(1, 0),
        "num_fake": labels.get(0, 0),
        "tasks": dict(tasks),
        "roles": dict(roles),
    }


def _to_float01(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint16:
        out = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    else:
        out = arr.astype(np.float32)
        max_val = float(np.max(out)) if out.size else 0.0
        if max_val > 255.0:
            out = out / 65535.0
        elif max_val > 1.0:
            out = out / 255.0
    return out


def _adapt_channels(arr: np.ndarray, num_channels: int) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    cur_channels = arr.shape[2]
    if cur_channels == num_channels:
        return arr

    if num_channels == 1:
        return arr.mean(axis=2, keepdims=True)

    if cur_channels == 1:
        return np.repeat(arr, num_channels, axis=2)

    if cur_channels > num_channels:
        return arr[:, :, :num_channels]

    repeat = math.ceil(num_channels / cur_channels)
    return np.repeat(arr, repeat, axis=2)[:, :, :num_channels]


def _load_image_tensor(
    image_path: str,
    *,
    resolution: int,
    num_channels: int,
    mean_tensor: Optional[torch.Tensor],
    std_tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    with Image.open(image_path) as image:
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.BILINEAR)
        arr = np.asarray(image)

    arr = _to_float01(arr)
    arr = _adapt_channels(arr, num_channels=num_channels)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()

    if mean_tensor is not None and std_tensor is not None:
        tensor = (tensor - mean_tensor) / std_tensor
    return tensor


class BinaryDomainImageDataset(Dataset):
    """PyTorch dataset for binary domain real/fake classification."""

    def __init__(
        self,
        records: list[BinaryImageRecord],
        *,
        resolution: int,
        num_channels: int,
        normalize_mean: Optional[tuple[float, ...]] = None,
        normalize_std: Optional[tuple[float, ...]] = None,
        use_horizontal_flip: bool = False,
        use_vertical_flip: bool = False,
    ) -> None:
        self.records = records
        self.resolution = int(resolution)
        self.num_channels = int(num_channels)
        self.use_horizontal_flip = bool(use_horizontal_flip)
        self.use_vertical_flip = bool(use_vertical_flip)

        if normalize_mean is None or normalize_std is None:
            self._mean = None
            self._std = None
        else:
            self._mean = (
                torch.tensor(normalize_mean, dtype=torch.float32)
                .view(self.num_channels, 1, 1)
            )
            self._std = (
                torch.tensor(normalize_std, dtype=torch.float32)
                .view(self.num_channels, 1, 1)
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        record = self.records[idx]
        pixel_values = _load_image_tensor(
            record.image_path,
            resolution=self.resolution,
            num_channels=self.num_channels,
            mean_tensor=self._mean,
            std_tensor=self._std,
        )

        if self.use_horizontal_flip and torch.rand(1).item() > 0.5:
            pixel_values = TF.hflip(pixel_values)
        if self.use_vertical_flip and torch.rand(1).item() > 0.5:
            pixel_values = TF.vflip(pixel_values)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(record.label, dtype=torch.long),
            "path": record.image_path,
        }


class InferenceImageDataset(Dataset):
    """Dataset for classifier scoring over arbitrary image files."""

    def __init__(
        self,
        image_paths: list[str],
        *,
        resolution: int,
        num_channels: int,
        normalize_mean: Optional[tuple[float, ...]] = None,
        normalize_std: Optional[tuple[float, ...]] = None,
    ) -> None:
        self.image_paths = image_paths
        self.resolution = int(resolution)
        self.num_channels = int(num_channels)
        if normalize_mean is None or normalize_std is None:
            self._mean = None
            self._std = None
        else:
            self._mean = (
                torch.tensor(normalize_mean, dtype=torch.float32)
                .view(self.num_channels, 1, 1)
            )
            self._std = (
                torch.tensor(normalize_std, dtype=torch.float32)
                .view(self.num_channels, 1, 1)
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[idx]
        pixel_values = _load_image_tensor(
            image_path,
            resolution=self.resolution,
            num_channels=self.num_channels,
            mean_tensor=self._mean,
            std_tensor=self._std,
        )
        return {"pixel_values": pixel_values, "path": image_path}

