# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Task-specific configurations for LBM baseline training.

Each task (sar2eo, rgb2ir, sar2ir, sar2rgb) defines its own config with
resolution, channel layout, model architecture, and training hyper-parameters.
Configs are plain dataclasses so per-task scripts can override any field.

Reference: Chadebec, Clément, Onur Tasar, Sanjeev Sreetharan, and Benjamin Aubin.
"LBM: Latent Bridge Matching for Fast Image-to-Image Translation."
ICCV 2025 (Highlight). https://arxiv.org/abs/2503.07535
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TaskConfig:
    """Configuration for a single LBM image-to-image translation task."""

    # ---- task identity ----
    task_name: str = ""

    # ---- data ----
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 1  # channels the UNet operates in
    resolution: int = 512
    use_augmented: bool = True
    use_random_crop: bool = True
    use_horizontal_flip: bool = False
    use_vertical_flip: bool = False

    # ---- sample filtering ----
    exclude_file: Optional[str] = "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"

    # ---- model ----
    backbone_type: str = "adm"
    num_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: str = "32,16,8"
    dropout: float = 0.0
    condition_mode: str = "concat"
    channel_mult: str = ""
    enable_xformers: bool = False
    enable_flash_attention_2: bool = False

    # ---- scheduler (LBM-specific) ----
    num_train_timesteps: int = 1000
    bridge_noise_sigma: float = 0.001
    timestep_sampling: str = "uniform"  # "uniform" | "log_normal" | "custom_timesteps"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    selected_timesteps: Optional[List[float]] = None
    prob: Optional[List[float]] = None

    # ---- loss ----
    latent_loss_type: str = "l2"  # "l2" | "l1"
    latent_loss_weight: float = 1.0
    pixel_loss_type: str = "l2"   # "l2" | "l1" | "lpips"
    pixel_loss_weight: float = 0.0
    pixel_loss_max_size: int = 512

    # ---- training ----
    output_dir: str = "./ckpt"
    train_batch_size: int = 8
    eval_batch_size: int = 4
    num_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "prodigy"  # "prodigy" | "adamw" | "muon"
    learning_rate: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-5
    use_ema: bool = True
    ema_decay: float = 0.9999
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
    max_validation_batches: Optional[int] = None
    paired_val_manifest: Optional[str] = None
    sar2rgb_sup_manifest: Optional[str] = None
    use_sar2rgb_sup: bool = False

    # ---- hub ----
    push_to_hub: bool = True
    hub_model_id: Optional[str] = None

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
    num_inference_steps: int = 1  # LBM is designed for single-step inference
    cfg_scale: float = 1.0
    output_resolution: Optional[int] = None

    # ---- latent modeling ablation ----
    use_latent_target: bool = False
    latent_vae_path: Optional[str] = None
    lambda_latent: float = 1.0
    latent_channels: int = 32

    # ---- representation alignment ----
    use_rep_alignment: bool = False
    rep_alignment_model_path: Optional[str] = None
    lambda_rep_alignment: float = 0.1
    lambda_rep_alignment_decay_steps: int = 0
    lambda_rep_alignment_end: float = 0.0

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
        output_dir="./ckpt",
        train_batch_size=32,
        eval_batch_size=16,
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
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
        output_dir="./ckpt",
        train_batch_size=8,
        eval_batch_size=4,
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
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
        output_dir="./ckpt",
        train_batch_size=8,
        eval_batch_size=4,
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
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
        output_dir="./ckpt",
        train_batch_size=8,
        eval_batch_size=4,
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        rep_alignment_model_path="./models/BiliSakura/MaRS-Base-RGB",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
