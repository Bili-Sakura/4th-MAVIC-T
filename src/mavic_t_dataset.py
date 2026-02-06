from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import datasets

TASKS = (
    "sar2eo",
    "rgb2ir",
    "sar2ir",
    "sar2rgb",
    "rgb2ir_crop_aug",
    "sar2ir_crop_aug",
    "sar2rgb_crop_aug",
)
IMAGE_EXTS = (".png", ".tif", ".tiff")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_REFINED_ROOT = _repo_root() / "datasets/BiliSakura/MACIV-T-2025-Structure-Refined"
DEFAULT_ORIGINAL_ROOT = _repo_root() / "datasets/BiliSakura/MAVIC-T-2025"
REFINED_MANIFEST_FILES = (
    DEFAULT_REFINED_ROOT / "manifests" / "refined_manifest.csv",
    DEFAULT_REFINED_ROOT / "manifests" / "refined_manifest_crop_aug.csv",
)


def _is_hidden(path: Path) -> bool:
    name = path.name
    return name.startswith(".") or name.startswith("._")


def _normalize_task(task: Optional[str]) -> Optional[str]:
    if task is None:
        return None
    task = task.lower()
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {', '.join(TASKS)}")
    return task


class MavicTImageToImageDataset:
    """HuggingFace Datasets wrapper for MAVIC-T train/val/test splits."""

    def __init__(
        self,
        refined_root: str | Path = DEFAULT_REFINED_ROOT,
        original_root: str | Path = DEFAULT_ORIGINAL_ROOT,
    ) -> None:
        self.refined_root = Path(refined_root)
        self.original_root = Path(original_root)

    def load(
        self,
        split: str,
        task: Optional[str] = None,
        *,
        with_target: Optional[bool] = None,
        diffusers_format: bool = False,
    ) -> datasets.Dataset:
        split = split.lower()
        task = _normalize_task(task)
        if with_target is None:
            with_target = split == "train"

        if split == "train":
            return self._load_refined_train(task, with_target, diffusers_format)
        if split in ("val", "test"):
            if task is None:
                raise ValueError("task is required for val/test splits.")
            return self._load_original_eval(split, task, with_target, diffusers_format)

        raise ValueError("split must be one of: train, val, test.")

    def load_all_tasks(self, split: str, *, diffusers_format: bool = False) -> datasets.DatasetDict:
        split = split.lower()
        if split == "train":
            tasks = TASKS
        elif split in ("val", "test"):
            tasks = tuple(
                task for task in TASKS if (self.original_root / split / task).is_dir()
            )
            if not tasks:
                raise FileNotFoundError(f"No task folders found under {self.original_root}/{split}")
        else:
            raise ValueError("split must be one of: train, val, test.")

        return datasets.DatasetDict(
            {task: self.load(split, task=task, diffusers_format=diffusers_format) for task in tasks}
        )

    def _load_refined_train(
        self,
        task: Optional[str],
        with_target: bool,
        diffusers_format: bool,
    ) -> datasets.Dataset:
        records = []
        manifest_paths = [
            self.refined_root / "manifests" / path.name if isinstance(path, Path) else path
            for path in REFINED_MANIFEST_FILES
        ]
        manifest_paths = [p for p in manifest_paths if p.is_file()]
        if not manifest_paths:
            raise FileNotFoundError(
                f"Missing refined manifest files under {self.refined_root}/manifests "
                f"(expected one of: {', '.join(p.name for p in REFINED_MANIFEST_FILES)})"
            )

        for manifest_path in manifest_paths:
            with manifest_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    row_split = (row.get("split") or "").lower()
                    row_task = (row.get("task") or "").lower()
                    if row_split != "train":
                        continue
                    if task is not None and row_task != task:
                        continue

                    input_path = (self.refined_root / row["input"]).resolve()
                    target_path = (self.refined_root / row["target"]).resolve()

                    record = {
                        "id": row["input"],
                        "task": row_task,
                        "split": row_split,
                        "input": str(input_path),
                        "input_path": str(input_path),
                        "tile": row.get("tile", ""),
                        "source_city": row.get("source_city", ""),
                    }

                    if with_target:
                        record["target"] = str(target_path)
                        record["target_path"] = str(target_path)

                    if diffusers_format:
                        record["conditioning_image"] = record["input"]
                        if with_target:
                            record["image"] = record["target"]

                    records.append(record)

        dataset = datasets.Dataset.from_list(records)
        features = self._build_features(
            with_target=with_target,
            diffusers_format=diffusers_format,
            include_train_meta=True,
        )
        return dataset.cast(features)

    def _load_original_eval(
        self,
        split: str,
        task: str,
        with_target: bool,
        diffusers_format: bool,
    ) -> datasets.Dataset:
        if with_target:
            raise ValueError("Targets are not available for val/test splits.")

        task_dir = self.original_root / split / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Missing task directory: {task_dir}")

        files = sorted(
            f
            for f in task_dir.iterdir()
            if f.is_file() and not _is_hidden(f) and f.suffix.lower() in IMAGE_EXTS
        )
        records = []
        for f in files:
            input_path = f.resolve()
            record = {
                "id": f"{split}/{task}/{f.name}",
                "task": task,
                "split": split,
                "input": str(input_path),
                "input_path": str(input_path),
            }
            if diffusers_format:
                record["conditioning_image"] = record["input"]
            records.append(record)

        dataset = datasets.Dataset.from_list(records)
        features = self._build_features(
            with_target=False,
            diffusers_format=diffusers_format,
            include_train_meta=False,
        )
        return dataset.cast(features)

    def _build_features(
        self,
        *,
        with_target: bool,
        diffusers_format: bool,
        include_train_meta: bool,
    ) -> datasets.Features:
        features = {
            "id": datasets.Value("string"),
            "task": datasets.Value("string"),
            "split": datasets.Value("string"),
            "input": datasets.Image(),
            "input_path": datasets.Value("string"),
        }
        if include_train_meta:
            features["tile"] = datasets.Value("string")
            features["source_city"] = datasets.Value("string")
        if with_target:
            features["target"] = datasets.Image()
            features["target_path"] = datasets.Value("string")
        if diffusers_format:
            features["conditioning_image"] = datasets.Image()
            if with_target:
                features["image"] = datasets.Image()
        return datasets.Features(features)
