"""Task-specific configurations for I2SB baseline training.

Each task (sar2eo, rgb2ir, sar2ir, sar2rgb) defines its own config with
resolution, channel layout, model architecture, and training hyper-parameters.
Configs are plain dataclasses so per-task scripts can override any field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TaskConfig:
    """Configuration for a single I2SB image-to-image translation task."""

    # ---- task identity ----
    task_name: str = ""

    # ---- data ----
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 1  # channels the I2SB UNet operates in
    resolution: int = 256
    use_augmented: bool = True  # also load *_crop_aug training split
    use_horizontal_flip: bool = False  # random horizontal flip augmentation
    use_vertical_flip: bool = False  # random vertical flip augmentation

    # ---- sample filtering ----
    exclude_file: Optional[str] = "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"  # path to txt of bad image paths to skip

    # ---- model ----
    unet_type: str = "adm"
    num_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: str = "32,16,8"
    dropout: float = 0.0
    condition_mode: str = "concat"
    channel_mult: str = ""  # auto-detected from resolution when empty

    # ---- scheduler (I2SB-specific) ----
    interval: int = 1000    # number of diffusion timesteps
    beta_max: float = 0.3   # max diffusion rate
    t0: float = 1e-4        # start time
    T: float = 1.0          # end time
    ot_ode: bool = False     # use OT-ODE path (deterministic sampling)
    clip_denoise: bool = False

    # ---- training ----
    output_dir: str = "./ckpt"
    train_batch_size: int = 8
    eval_batch_size: int = 4
    num_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "prodigy"  # "prodigy" | "adamw" | "muon"
    learning_rate: float = 1.0  # Prodigy adapts lr; set to 1.0 by default
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-5  # Prodigy d0 parameter (initial estimate of D)
    use_ema: bool = True
    ema_decay: float = 0.9999

    # ---- logging / checkpointing ----
    log_with: str = "tensorboard"
    save_model_epochs: Optional[int] = 1
    checkpointing_steps: Optional[int] = None
    checkpoints_total_limit: int = 1
    resume_from_checkpoint: Optional[str] = None

    # ---- validation ----
    validation_epochs: Optional[int] = None  # run validation every N epochs
    validation_steps: Optional[int] = None   # run validation every N steps

    # ---- hub ----
    push_to_hub: bool = True
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
    nfe: int = 100  # number of function evaluations during sampling

    # ---- latent modeling ablation ----
    use_latent_target: bool = False
    latent_vae_path: Optional[str] = None  # path to pre-trained VAE checkpoint
    lambda_latent: float = 1.0  # weight for latent-space L2 loss
    latent_channels: int = 32  # VAE latent dimension (overrides model_channels when use_latent_target=True)

    # ---- representation alignment ----
    use_rep_alignment: bool = False
    rep_alignment_model_path: Optional[str] = None  # path to encoder checkpoint
    lambda_rep_alignment: float = 0.1  # weight for alignment loss


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
        output_dir="./ckpt",
        train_batch_size=32,
        eval_batch_size=16,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment via MaRS-SAR
        rep_alignment_model_path="./models/BiliSakura/MaRS-Base-SAR",
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
        output_dir="./ckpt",
        train_batch_size=8,
        eval_batch_size=4,
        # latent modeling ablation (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment via MaRS-RGB
        rep_alignment_model_path="./models/BiliSakura/MaRS-Base-RGB",
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
        output_dir="./ckpt",
        train_batch_size=8,
        eval_batch_size=4,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment via MaRS-SAR
        rep_alignment_model_path="./models/BiliSakura/MaRS-Base-SAR",
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
        output_dir="./ckpt",
        train_batch_size=8,
        eval_batch_size=4,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment via MaRS-SAR
        rep_alignment_model_path="./models/BiliSakura/MaRS-Base-SAR",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
