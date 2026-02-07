"""Task-specific configurations for DDBM baseline training.

Each task (sar2eo, rgb2ir, sar2ir, sar2rgb) defines its own config with
resolution, channel layout, model architecture, and training hyper-parameters.
Configs are plain dataclasses so per-task scripts can override any field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TaskConfig:
    """Configuration for a single DDBM image-to-image translation task."""

    # ---- task identity ----
    task_name: str = ""

    # ---- data ----
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 1  # channels the DDBM UNet operates in
    resolution: int = 256
    use_augmented: bool = False  # also load *_crop_aug training split
    use_horizontal_flip: bool = False  # random horizontal flip augmentation
    use_vertical_flip: bool = False  # random vertical flip augmentation

    # ---- sample filtering ----
    exclude_file: Optional[str] = None  # path to txt of bad image paths to skip

    # ---- model ----
    unet_type: str = "adm"
    num_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: str = "32,16,8"
    dropout: float = 0.0
    condition_mode: str = "concat"
    channel_mult: str = ""  # auto-detected from resolution when empty

    # ---- scheduler ----
    pred_mode: str = "vp"
    sigma_max: float = 1.0
    sigma_min: float = 0.002
    sigma_data: float = 0.5
    beta_d: float = 2.0
    beta_min: float = 0.1

    # ---- training ----
    output_dir: str = "./outputs/ddbm"
    train_batch_size: int = 8
    eval_batch_size: int = 4
    num_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "prodigy"  # "prodigy" | "adamw"
    learning_rate: float = 1.0  # Prodigy adapts lr; set to 1.0 by default
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    weight_decay: float = 0.0
    use_ema: bool = True
    ema_decay: float = 0.9999

    # ---- logging / checkpointing ----
    log_with: str = "tensorboard"
    save_model_epochs: int = 10
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 1
    resume_from_checkpoint: Optional[str] = None

    # ---- hub ----
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    # ---- hardware ----
    mixed_precision: str = "bf16"
    dataloader_num_workers: int = 4
    seed: int = 42

    # ---- metric-based loss (MAVIC-T evaluation objective) ----
    use_mavic_loss: bool = False   # add LPIPS + L1 loss alongside denoising loss
    mavic_lpips_weight: float = 1.0
    mavic_l1_weight: float = 1.0
    mavic_loss_weight: float = 0.1  # relative weight vs. the denoising loss

    # ---- sampling (evaluation) ----
    num_inference_steps: int = 40
    guidance: float = 1.0
    churn_step_ratio: float = 0.33


# ---------------------------------------------------------------------------
# Pre-built configs for the four core tasks
# ---------------------------------------------------------------------------

def sar2eo_config(**overrides) -> TaskConfig:
    """SAR-to-EO: 1-band 256x256 → 1-band 256x256."""
    cfg = TaskConfig(
        task_name="sar2eo",
        source_channels=1,
        target_channels=1,
        model_channels=1,
        resolution=256,
        output_dir="./outputs/ddbm_sar2eo",
        train_batch_size=32,
        eval_batch_size=16,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    """RGB-to-IR: 3-band → 1-band (native 1024×1024)."""
    cfg = TaskConfig(
        task_name="rgb2ir",
        source_channels=3,
        target_channels=1,
        model_channels=3,  # operate in 3-ch space; 1-ch target is expanded
        resolution=1024,
        use_augmented=True,
        output_dir="./outputs/ddbm_rgb2ir",
        train_batch_size=8,
        eval_batch_size=4,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    """SAR-to-IR: 1-band → 1-band (native 1024×1024)."""
    cfg = TaskConfig(
        task_name="sar2ir",
        source_channels=1,
        target_channels=1,
        model_channels=1,
        resolution=1024,
        use_augmented=True,
        output_dir="./outputs/ddbm_sar2ir",
        train_batch_size=8,
        eval_batch_size=4,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    """SAR-to-RGB: 1-band → 3-band (native 1024×1024)."""
    cfg = TaskConfig(
        task_name="sar2rgb",
        source_channels=1,
        target_channels=3,
        model_channels=3,  # operate in 3-ch space; 1-ch source is expanded
        resolution=1024,
        use_augmented=True,
        output_dir="./outputs/ddbm_sar2rgb",
        train_batch_size=8,
        eval_batch_size=4,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
