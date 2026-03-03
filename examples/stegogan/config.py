# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Task-specific configurations for StegoGAN baseline training.

Each task (sar2eo, rgb2ir, sar2ir, sar2rgb) defines its own config with
resolution, channel layout, model architecture, and training hyper-parameters.
Configs are plain dataclasses so per-task scripts can override any field.

Reference: Wu, Sidi, et al. "StegoGAN: Leveraging Steganography for
Non-Bijective Image-to-Image Translation." CVPR 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskConfig:
    """Configuration for a single StegoGAN image-to-image translation task."""

    # ---- task identity ----
    task_name: str = ""

    # ---- data ----
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 3
    resolution: int = 512
    load_size: Optional[int] = None
    use_augmented: bool = True
    use_random_crop: bool = True
    use_horizontal_flip: bool = True
    use_vertical_flip: bool = False

    # ---- sample filtering ----
    exclude_file: Optional[str] = "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"

    # ---- generator A (source → target) ----
    ngf: int = 64
    n_blocks: int = 9
    normG: str = "instance"
    no_dropout: bool = True
    resnet_layer: int = 8
    use_fusion_block: bool = True
    init_type: str = "normal"
    init_gain: float = 0.02

    # ---- generator B (target → source, with mask) ----
    mask_group: int = 256

    # ---- discriminator ----
    ndf: int = 64
    n_layers_D: int = 3
    normD: str = "instance"

    # ---- StegoGAN loss hyper-parameters ----
    gan_mode: str = "lsgan"
    lambda_A: float = 10.0       # forward cycle loss weight
    lambda_B: float = 10.0       # backward cycle loss weight
    lambda_identity: float = 0.5  # identity mapping loss
    lambda_reg: float = 0.2       # mask sparsity regularisation
    lambda_consistency: float = 3.0  # steganographic consistency loss

    # ---- training ----
    output_dir: str = "./outputs/stegogan"
    train_batch_size: int = 1
    max_grad_norm: float = 1.0
    eval_batch_size: int = 4
    n_epochs: int = 100
    n_epochs_decay: int = 0
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "adam"
    learning_rate_G: float = 2e-4
    learning_rate_D: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lr_policy: str = "linear"

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
    validation_resolution: Optional[int] = None
    paired_val_manifest: Optional[str] = None
    sar2rgb_sup_manifest: Optional[str] = None
    use_sar2rgb_sup: bool = False

    # ---- hub ----
    push_to_hub: bool = True
    hub_model_id: Optional[str] = None
    hub_path_tier: Optional[str] = None

    # ---- hardware ----
    mixed_precision: str = "bf16"
    dataloader_num_workers: int = 4
    seed: int = 42

    # ---- metric-based loss (MAVIC-T evaluation objective) ----
    use_mavic_loss: bool = False
    mavic_lpips_weight: float = 1.0
    mavic_l1_weight: float = 1.0
    mavic_loss_weight: float = 0.1

    # ---- sampling (evaluation) ----
    num_inference_steps: int = 1

    # ---- SAR-specific preprocessing ----
    use_sar_despeckle: bool = False
    sar_despeckle_kernel_size: int = 5
    sar_despeckle_strength: float = 0.6


# ---------------------------------------------------------------------------
# Pre-built configs for the four core tasks
# ---------------------------------------------------------------------------

def _default_paired_val_manifest(task_name: str) -> str:
    return f"datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_{task_name}.txt"


def _default_sar2rgb_sup_manifest() -> str:
    return "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_sar2rgb_sup.txt"


def sar2eo_config(**overrides) -> TaskConfig:
    """SAR-to-EO: native 1024×1024, runtime random-crop to 512×512."""
    cfg = TaskConfig(
        task_name="sar2eo",
        paired_val_manifest=_default_paired_val_manifest("sar2eo"),
        source_channels=1,
        target_channels=1,
        model_channels=1,
        resolution=512,
        output_dir="./outputs/stegogan_sar2eo",
        train_batch_size=1,
        eval_batch_size=4,
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
        use_augmented=True,
        output_dir="./outputs/stegogan_rgb2ir",
        train_batch_size=1,
        eval_batch_size=4,
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
        use_augmented=True,
        use_sar_despeckle=True,
        output_dir="./outputs/stegogan_sar2ir",
        train_batch_size=1,
        eval_batch_size=4,
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
        use_augmented=True,
        use_sar_despeckle=True,
        output_dir="./outputs/stegogan_sar2rgb",
        train_batch_size=1,
        eval_batch_size=4,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
