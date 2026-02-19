"""Task-specific configurations for Img2Image-Turbo (Pix2Pix-Turbo) training.

.. note::
   **Lower priority**: the Img2Image-Turbo / Pix2Pix-Turbo method has been
   found less suitable for the MAVIC-T task compared to other baselines
   (CUT, DDBM).  Its code is retained for reference and future
   experimentation, but further implementation effort should focus on the
   other baselines first.

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
    use_augmented: bool = True  # also load *_crop_aug training split
    use_horizontal_flip: bool = False
    use_vertical_flip: bool = False

    # ---- sample filtering ----
    exclude_file: Optional[str] = "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"  # path to txt of bad image paths to skip

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
    optimizer_type: str = "prodigy"  # "prodigy" | "adamw" | "muon"
    learning_rate: float = 1.0  # Prodigy adapts lr; set to 1.0 by default
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-5  # Prodigy d0 parameter (initial estimate of D)
    max_grad_norm: float = 1.0

    # ---- loss weights ----
    lambda_l2: float = 1.0
    lambda_lpips: float = 5.0

    # ---- logging / checkpointing ----
    # Accelerate log_with: "tensorboard" | "wandb" | "swanlab" | "all" | comma-separated
    log_with: str = "tensorboard"
    # SwanLab init kwargs (used only when log_with contains "swanlab")
    swanlab_experiment_name: Optional[str] = None
    swanlab_description: Optional[str] = None
    swanlab_tags: Optional[str] = None  # comma-separated tags
    swanlab_init_kwargs_json: Optional[str] = None  # JSON object merged into init_kwargs["swanlab"]
    save_model_epochs: Optional[int] = 1
    checkpointing_steps: Optional[int] = None
    checkpoints_total_limit: int = 1
    resume_from_checkpoint: Optional[str] = None

    # ---- validation ----
    validation_epochs: Optional[int] = None  # run validation every N epochs
    validation_steps: Optional[int] = None   # run validation every N steps
    paired_val_manifest: Optional[str] = None  # path to paired_val_<task>.txt for golden val (log validation)
    sar2rgb_sup_manifest: Optional[str] = None  # path to paired_sar2rgb_sup.txt (extra supervised SAR→RGB train data)
    use_sar2rgb_sup: bool = False  # when True and task is sar2rgb, add sar2rgb_sup pairs

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

    # ---- latent modeling ablation ----
    use_latent_target: bool = False
    latent_vae_path: Optional[str] = None  # path to pre-trained VAE checkpoint
    lambda_latent: float = 1.0  # weight for latent-space L2 loss
    latent_channels: int = 32  # VAE latent dimension

    # ---- representation alignment ----
    use_rep_alignment: bool = False
    rep_alignment_model_path: Optional[str] = None  # path to encoder checkpoint
    lambda_rep_alignment: float = 0.1  # weight for alignment loss (constant, or start when schedule used)
    lambda_rep_alignment_decay_steps: int = 0
    lambda_rep_alignment_end: float = 0.0


# ---------------------------------------------------------------------------
# Pre-built configs for the four core tasks
# ---------------------------------------------------------------------------

def _default_paired_val_manifest(task_name: str) -> str:
    return f"datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_{task_name}.txt"


def _default_sar2rgb_sup_manifest() -> str:
    return "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_sar2rgb_sup.txt"


def sar2eo_config(**overrides) -> TaskConfig:
    """SAR-to-EO: 1-band 256×256 → 1-band 256×256."""
    cfg = TaskConfig(
        task_name="sar2eo",
        paired_val_manifest=_default_paired_val_manifest("sar2eo"),
        prompt="convert SAR image to electro-optical image",
        source_channels=1,
        target_channels=1,
        model_channels=3,
        resolution=512,
        output_dir="./outputs/turbo_sar2eo",
        train_batch_size=4,
        eval_batch_size=1,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment not applicable – no pre-trained EO encoder
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    """RGB-to-IR: 3-band → 1-band."""
    cfg = TaskConfig(
        task_name="rgb2ir",
        paired_val_manifest=_default_paired_val_manifest("rgb2ir"),
        prompt="convert RGB image to infrared image",
        source_channels=3,
        target_channels=1,
        model_channels=3,
        resolution=1024,
        use_augmented=True,
        output_dir="./outputs/turbo_rgb2ir",
        train_batch_size=4,
        eval_batch_size=1,
        # latent modeling ablation (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment not applicable – no pre-trained IR encoder
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    """SAR-to-IR: 1-band → 1-band."""
    cfg = TaskConfig(
        task_name="sar2ir",
        paired_val_manifest=_default_paired_val_manifest("sar2ir"),
        prompt="convert SAR image to infrared image",
        source_channels=1,
        target_channels=1,
        model_channels=3,
        resolution=1024,
        use_augmented=True,
        output_dir="./outputs/turbo_sar2ir",
        train_batch_size=4,
        eval_batch_size=1,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment not applicable – no pre-trained IR encoder
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    """SAR-to-RGB: 1-band → 3-band."""
    cfg = TaskConfig(
        task_name="sar2rgb",
        paired_val_manifest=_default_paired_val_manifest("sar2rgb"),
        sar2rgb_sup_manifest=_default_sar2rgb_sup_manifest(),
        use_sar2rgb_sup=True,
        prompt="convert SAR image to RGB image",
        source_channels=1,
        target_channels=3,
        model_channels=3,
        resolution=1024,
        use_augmented=True,
        output_dir="./outputs/turbo_sar2rgb",
        train_batch_size=4,
        eval_batch_size=1,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment via MaRS-RGB (encode the RGB target)
        rep_alignment_model_path="./models/BiliSakura/MaRS-Base-RGB",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
