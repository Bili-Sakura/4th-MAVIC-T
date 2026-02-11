"""Task-specific configurations for DDIB baseline training.

DDIB (Dual Diffusion Implicit Bridges) trains *two independent unconditional*
diffusion models – one per domain – and translates images via a deterministic
DDIM encode→decode ODE.  Each task (sar2eo, rgb2ir, sar2ir, sar2rgb) defines
its own config with resolution, channel layout, model architecture, and
training hyper-parameters.  Configs are plain dataclasses so per-task scripts
can override any field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskConfig:
    """Configuration for a single DDIB image-to-image translation task."""

    # ---- task identity ----
    task_name: str = ""

    # ---- data ----
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 1  # channels each unconditional UNet operates in
    resolution: int = 256
    use_augmented: bool = True  # also load *_crop_aug training split
    use_horizontal_flip: bool = False
    use_vertical_flip: bool = False

    # ---- sample filtering ----
    exclude_file: Optional[str] = "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"

    # ---- model ----
    num_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: str = "32,16,8"
    dropout: float = 0.0
    learn_sigma: bool = False
    channel_mult: str = ""  # auto-detected from resolution when empty

    # ---- diffusion ----
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    predict_xstart: bool = False
    rescale_timesteps: bool = False

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
    prodigy_d0: float = 1e-5  # Prodigy d0 parameter (initial estimate of D)
    use_ema: bool = True
    ema_decay: float = 0.9999

    # ---- logging / checkpointing ----
    # Accelerate log_with: "tensorboard" | "wandb" | "swanlab" | "all" | comma-separated
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
    use_mavic_loss: bool = False
    mavic_lpips_weight: float = 1.0
    mavic_l1_weight: float = 1.0
    mavic_loss_weight: float = 0.1

    # ---- sampling (evaluation) ----
    num_inference_steps: int = 250  # DDIM steps for encode/decode
    clip_denoised: bool = True
    eta: float = 0.0  # DDIM eta (0 = deterministic)

    # ---- latent modeling ablation ----
    use_latent_target: bool = False
    latent_vae_path: Optional[str] = None
    lambda_latent: float = 1.0
    latent_channels: int = 32  # VAE latent dimension (overrides model_channels when use_latent_target=True)

    # ---- representation alignment ----
    use_rep_alignment: bool = False
    rep_alignment_model_path: Optional[str] = None
    lambda_rep_alignment: float = 0.1
    lambda_rep_alignment_decay_steps: int = 0
    lambda_rep_alignment_end: float = 0.0


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
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment not applicable – no pre-trained EO encoder
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
        model_channels=3,
        resolution=1024,
        use_augmented=True,
        output_dir="./ckpt",
        train_batch_size=8,
        eval_batch_size=4,
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment not applicable – no pre-trained IR encoder
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
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment not applicable – no pre-trained IR encoder
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
        model_channels=3,
        resolution=1024,
        use_augmented=True,
        output_dir="./ckpt",
        train_batch_size=8,
        eval_batch_size=4,
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment via MaRS-RGB (encode the RGB target)
        rep_alignment_model_path="./models/BiliSakura/MaRS-Base-RGB",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
