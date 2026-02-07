"""Task-specific configurations for Img2Image-Turbo (Pix2Pix-Turbo) training.

Each task (sar2eo, rgb2ir, sar2ir, sar2rgb) defines its own config with
resolution, channel layout, model architecture, and training hyper-parameters.
Configs are plain dataclasses so per-task scripts can override any field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskConfig:
    """Configuration for a single Pix2Pix-Turbo image-to-image translation task."""

    # ---- task identity ----
    task_name: str = ""
    prompt: str = "translate the image"

    # ---- data ----
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 3  # SD-Turbo VAE expects 3-ch RGB
    resolution: int = 512
    use_augmented: bool = False
    use_horizontal_flip: bool = False
    use_vertical_flip: bool = False

    # ---- sample filtering ----
    exclude_file: Optional[str] = None  # path to txt of bad image paths to skip

    # ---- model (Pix2Pix-Turbo) ----
    pretrained_model_name_or_path: str = "stabilityai/sd-turbo"
    lora_rank_unet: int = 8
    lora_rank_vae: int = 4

    # ---- training ----
    output_dir: str = "./outputs/img2img_turbo"
    train_batch_size: int = 4
    eval_batch_size: int = 1
    num_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "prodigy"  # "prodigy" | "adamw"
    learning_rate: float = 1.0  # Prodigy adapts lr; set to 1.0 by default
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0

    # ---- loss weights ----
    lambda_l2: float = 1.0
    lambda_lpips: float = 5.0

    # ---- logging / checkpointing ----
    log_with: str = "tensorboard"
    save_model_epochs: Optional[int] = 1
    checkpointing_steps: Optional[int] = None
    checkpoints_total_limit: int = 1
    resume_from_checkpoint: Optional[str] = None

    # ---- hub ----
    push_to_hub: bool = True
    hub_model_id: Optional[str] = None

    # ---- hardware ----
    mixed_precision: str = "bf16"
    dataloader_num_workers: int = 4
    seed: int = 42
    gradient_checkpointing: bool = False

    # ---- metric-based loss (MAVIC-T evaluation objective) ----
    use_mavic_loss: bool = False
    mavic_lpips_weight: float = 1.0
    mavic_l1_weight: float = 1.0
    mavic_loss_weight: float = 0.1


# ---------------------------------------------------------------------------
# Pre-built configs for the four core tasks
# ---------------------------------------------------------------------------

def sar2eo_config(**overrides) -> TaskConfig:
    """SAR-to-EO: 1-band 256×256 → 1-band 256×256."""
    cfg = TaskConfig(
        task_name="sar2eo",
        prompt="convert SAR image to electro-optical image",
        source_channels=1,
        target_channels=1,
        model_channels=3,
        resolution=512,
        output_dir="./outputs/turbo_sar2eo",
        train_batch_size=4,
        eval_batch_size=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    """RGB-to-IR: 3-band → 1-band."""
    cfg = TaskConfig(
        task_name="rgb2ir",
        prompt="convert RGB image to infrared image",
        source_channels=3,
        target_channels=1,
        model_channels=3,
        resolution=1024,
        use_augmented=True,
        output_dir="./outputs/turbo_rgb2ir",
        train_batch_size=4,
        eval_batch_size=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    """SAR-to-IR: 1-band → 1-band."""
    cfg = TaskConfig(
        task_name="sar2ir",
        prompt="convert SAR image to infrared image",
        source_channels=1,
        target_channels=1,
        model_channels=3,
        resolution=1024,
        use_augmented=True,
        output_dir="./outputs/turbo_sar2ir",
        train_batch_size=4,
        eval_batch_size=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    """SAR-to-RGB: 1-band → 3-band."""
    cfg = TaskConfig(
        task_name="sar2rgb",
        prompt="convert SAR image to RGB image",
        source_channels=1,
        target_channels=3,
        model_channels=3,
        resolution=1024,
        use_augmented=True,
        output_dir="./outputs/turbo_sar2rgb",
        train_batch_size=4,
        eval_batch_size=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
