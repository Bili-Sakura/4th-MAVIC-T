# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Configuration helpers for domain real/fake classifier pre-training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is importable when executed as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.mavic_t_dataset import TASKS  # noqa: E402


SUPPORTED_DOMAINS = ("sar", "eo", "rgb", "ir")
DOMAIN_CHANNELS = {
    "sar": 1,
    "eo": 1,
    "rgb": 3,
    "ir": 1,
}
DOMAIN_NATIVE_RESOLUTION = {
    "eo": 256,
    "ir": 1024,
    "rgb": 1024,
    # No current target tasks use SAR; keep a sensible default for future use.
    "sar": 1024,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_BAD_SAMPLES_FILE = (
    "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
)


def normalize_domain_name(domain: str) -> str:
    out = domain.strip().lower()
    if out not in SUPPORTED_DOMAINS:
        raise ValueError(
            f"Unsupported domain '{domain}'. Expected one of: {', '.join(SUPPORTED_DOMAINS)}"
        )
    return out


def strip_crop_aug_suffix(task: str) -> str:
    return task[:-9] if task.endswith("_crop_aug") else task


def parse_task_modalities(task: str) -> tuple[str, str]:
    base = strip_crop_aug_suffix(task.strip().lower())
    if "2" not in base:
        raise ValueError(
            f"Task '{task}' does not match expected '<source>2<target>' format."
        )
    source_domain, target_domain = base.split("2", 1)
    return normalize_domain_name(source_domain), normalize_domain_name(target_domain)


def target_domain_for_task(task: str) -> str:
    return parse_task_modalities(task)[1]


def parse_csv_tasks(tasks_csv: Optional[str]) -> tuple[str, ...]:
    if tasks_csv is None:
        return ()
    tasks = tuple(item.strip().lower() for item in tasks_csv.split(",") if item.strip())
    return tasks


def expand_tasks_with_crop_aug(tasks: tuple[str, ...], include_crop_aug: bool) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()

    for task in tasks:
        task_l = task.lower()
        if task_l not in seen:
            out.append(task_l)
            seen.add(task_l)

        if include_crop_aug and not task_l.endswith("_crop_aug"):
            aug_task = f"{task_l}_crop_aug"
            if aug_task in TASKS and aug_task not in seen:
                out.append(aug_task)
                seen.add(aug_task)

    return tuple(out)


def default_positive_tasks_for_domain(
    target_domain: str,
    *,
    include_crop_aug: bool = True,
) -> tuple[str, ...]:
    domain = normalize_domain_name(target_domain)
    base_tasks = tuple(
        task
        for task in TASKS
        if not task.endswith("_crop_aug") and target_domain_for_task(task) == domain
    )
    return expand_tasks_with_crop_aug(base_tasks, include_crop_aug=include_crop_aug)


def default_num_channels_for_domain(domain: str) -> int:
    return DOMAIN_CHANNELS[normalize_domain_name(domain)]


def default_resolution_for_domain(domain: str) -> int:
    return DOMAIN_NATIVE_RESOLUTION[normalize_domain_name(domain)]


def normalization_stats_for_channels(num_channels: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ImageNet-based mean/std for the given channel count (fallback)."""
    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")

    if num_channels == 1:
        mean_1 = sum(IMAGENET_MEAN) / len(IMAGENET_MEAN)
        std_1 = sum(IMAGENET_STD) / len(IMAGENET_STD)
        return (mean_1,), (std_1,)

    if num_channels <= 3:
        return IMAGENET_MEAN[:num_channels], IMAGENET_STD[:num_channels]

    mean = list(IMAGENET_MEAN)
    std = list(IMAGENET_STD)
    while len(mean) < num_channels:
        mean.append(IMAGENET_MEAN[-1])
        std.append(IMAGENET_STD[-1])
    return tuple(mean[:num_channels]), tuple(std[:num_channels])


def load_dataset_stats(stats_path: str | Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Load pre-computed mean/std from JSON file.

    Expected format: {"mean": [...], "std": [...]}
    """
    path = Path(stats_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset stats file not found: {path}")
    with path.open() as f:
        data = json.load(f)
    mean = tuple(float(x) for x in data["mean"])
    std = tuple(float(x) for x in data["std"])
    return mean, std


@dataclass
class DomainClassifierConfig:
    """Configuration for pre-training a target-domain real/fake classifier."""

    # ---- data identity ----
    target_domain: str = "ir"
    positive_tasks_csv: Optional[str] = None
    negative_target_tasks_csv: Optional[str] = None
    include_crop_aug: bool = True
    use_source_as_fake: bool = True
    negative_to_positive_ratio: float = 1.0

    # ---- data I/O ----
    refined_root: Optional[str] = None
    exclude_file: Optional[str] = DEFAULT_BAD_SAMPLES_FILE

    # ---- image shape / channels ----
    resolution: Optional[int] = None
    num_channels: Optional[int] = None
    use_horizontal_flip: bool = True
    use_vertical_flip: bool = False

    # ---- model ----
    pretrained_model_name_or_path: str = "Francesco/resnet18-224-1k"

    # ---- optimization ----
    output_dir: str = "./ckpt/domain_classifier"
    run_name: Optional[str] = None
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_epochs: int = 5
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 0

    # ---- runtime ----
    mixed_precision: str = "bf16"
    dataloader_num_workers: int = 4
    seed: int = 42
    val_split_ratio: float = 0.0

    # ---- checkpointing / logging ----
    log_every_steps: int = 50
    checkpoint_every_epochs: int = 1
    checkpoints_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None

    # ---- dataset normalization (domain-specific) ----
    dataset_stats_path: Optional[str] = None

    def resolved_target_domain(self) -> str:
        return normalize_domain_name(self.target_domain)

    def resolved_num_channels(self) -> int:
        if self.num_channels is not None:
            return int(self.num_channels)
        return default_num_channels_for_domain(self.resolved_target_domain())

    def resolved_resolution(self) -> int:
        if self.resolution is not None:
            return int(self.resolution)
        return default_resolution_for_domain(self.resolved_target_domain())

    def resolved_positive_tasks(self) -> tuple[str, ...]:
        requested = parse_csv_tasks(self.positive_tasks_csv)
        if requested:
            return expand_tasks_with_crop_aug(
                requested, include_crop_aug=self.include_crop_aug
            )
        return default_positive_tasks_for_domain(
            self.resolved_target_domain(),
            include_crop_aug=self.include_crop_aug,
        )

    def resolved_negative_target_tasks(self) -> tuple[str, ...]:
        requested = parse_csv_tasks(self.negative_target_tasks_csv)
        if not requested:
            return ()
        return expand_tasks_with_crop_aug(
            requested, include_crop_aug=self.include_crop_aug
        )

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"{self.resolved_target_domain()}-resnet18-real-fake"

    def resolved_normalization_stats(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return mean/std: from dataset_stats_path if set, else ImageNet fallback."""
        if self.dataset_stats_path:
            return load_dataset_stats(self.dataset_stats_path)
        return normalization_stats_for_channels(self.resolved_num_channels())

