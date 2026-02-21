"""Task-specific configurations for UniDB baseline training.

UniDB uses a noise-predicting ConditionalUNet with lambda_square, gamma,
and cosine/linear schedule. Based on https://github.com/2769433owo/UniDB-plusplus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _default_paired_val_manifest(task_name: str) -> str:
    return f"datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_{task_name}.txt"


def _default_sar2rgb_sup_manifest() -> str:
    return "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_sar2rgb_sup.txt"


@dataclass
class TaskConfig:
    """Configuration for UniDB image-to-image translation."""

    # ---- task identity ----
    task_name: str = ""

    # ---- data ----
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 3  # UniDB ConditionalUNet uses 3-ch (RGB)
    resolution: int = 512
    use_augmented: bool = True
    use_random_crop: bool = True  # random runtime crop to resolution during train
    use_horizontal_flip: bool = False
    use_vertical_flip: bool = False
    exclude_file: Optional[str] = "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"

    # ---- model (UniDB ConditionalUNet) ----
    nf: int = 64
    depth: int = 4

    # ---- scheduler (UniDB SDE) ----
    lambda_square: float = 30.0
    gamma: float = 1e7
    num_train_timesteps: int = 100
    schedule: str = "cosine"
    eps: float = 0.005

    # ---- training ----
    output_dir: str = "./ckpt"
    train_batch_size: int = 8
    num_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    weight_decay: float = 0.0
    use_ema: bool = True
    ema_decay: float = 0.995
    # Per-sample conditioning dropout probability for CFG fine-tuning.
    conditioning_dropout_prob: float = 0.0

    # ---- logging / checkpointing ----
    log_with: str = "tensorboard"
    swanlab_experiment_name: Optional[str] = None
    swanlab_description: Optional[str] = None
    swanlab_tags: Optional[str] = None
    swanlab_init_kwargs_json: Optional[str] = None
    save_model_epochs: Optional[int] = 1
    checkpointing_steps: Optional[int] = None
    checkpoints_total_limit: int = 1
    resume_from_checkpoint: Optional[str] = None

    # ---- validation ----
    validation_epochs: Optional[int] = None
    validation_steps: Optional[int] = None
    paired_val_manifest: Optional[str] = None
    sar2rgb_sup_manifest: Optional[str] = None  # path to paired_sar2rgb_sup.txt (extra supervised SAR→RGB train data)
    use_sar2rgb_sup: bool = False  # when True and task is sar2rgb, add sar2rgb_sup pairs

    # ---- hub ----
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    # ---- hardware ----
    mixed_precision: str = "bf16"
    dataloader_num_workers: int = 4
    seed: int = 42

    # ---- sampling (evaluation) ----
    num_inference_steps: int = 100
    # Classifier-Free Guidance scale. 1.0 disables CFG.
    cfg_scale: float = 1.0
    method: str = "euler"
    solver_type: str = "mean-ode"
    solver_step: int = 100
    output_resolution: Optional[int] = None


def sar2eo_config(**overrides) -> TaskConfig:
    """SAR-to-EO: native 1024×1024, runtime random-crop to 512×512."""
    cfg = TaskConfig(
        task_name="sar2eo",
        paired_val_manifest=_default_paired_val_manifest("sar2eo"),
        source_channels=1,
        target_channels=1,
        model_channels=1,
        resolution=512,
        output_dir="./ckpt",
        train_batch_size=32,
        validation_steps=1000,
        checkpointing_steps=5000,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    """RGB-to-IR: 3-band → 1-band (native 1024×1024, train at 512×512 crop)."""
    cfg = TaskConfig(
        task_name="rgb2ir",
        paired_val_manifest=_default_paired_val_manifest("rgb2ir"),
        source_channels=3,
        target_channels=1,
        model_channels=3,
        resolution=512,
        output_dir="./ckpt",
        train_batch_size=8,
        validation_steps=1000,
        checkpointing_steps=5000,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    """SAR-to-IR: 1-band → 1-band (native 1024×1024, train at 512×512 crop)."""
    cfg = TaskConfig(
        task_name="sar2ir",
        paired_val_manifest=_default_paired_val_manifest("sar2ir"),
        source_channels=1,
        target_channels=1,
        model_channels=1,
        resolution=512,
        output_dir="./ckpt",
        train_batch_size=8,
        validation_steps=1000,
        checkpointing_steps=5000,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    """SAR-to-RGB: 1-band → 3-band (native 1024×1024, train at 512×512 crop)."""
    cfg = TaskConfig(
        task_name="sar2rgb",
        paired_val_manifest=_default_paired_val_manifest("sar2rgb"),
        sar2rgb_sup_manifest=_default_sar2rgb_sup_manifest(),
        use_sar2rgb_sup=True,
        source_channels=1,
        target_channels=3,
        model_channels=3,
        resolution=512,
        output_dir="./ckpt",
        train_batch_size=8,
        validation_steps=1000,
        checkpointing_steps=5000,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
