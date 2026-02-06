"""Task-specific configurations for CUT baseline training.

Each task (sar2eo, rgb2ir, sar2ir, sar2rgb) defines its own config with
resolution, channel layout, model architecture, and training hyper-parameters.
Configs are plain dataclasses so per-task scripts can override any field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskConfig:
    """Configuration for a single CUT image-to-image translation task."""

    # ---- task identity ----
    task_name: str = ""

    # ---- data ----
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 3  # channels the CUT generator operates in
    resolution: int = 256
    load_size: int = 286  # resize before cropping (CUT default)
    use_augmented: bool = False
    use_horizontal_flip: bool = True
    use_vertical_flip: bool = False

    # ---- generator ----
    netG: str = "resnet_9blocks"  # resnet_9blocks | resnet_6blocks
    ngf: int = 64
    normG: str = "instance"
    no_dropout: bool = True
    no_antialias: bool = False
    no_antialias_up: bool = False
    init_type: str = "xavier"
    init_gain: float = 0.02

    # ---- discriminator ----
    netD: str = "basic"  # basic (PatchGAN 70x70)
    ndf: int = 64
    n_layers_D: int = 3
    normD: str = "instance"

    # ---- feature network (PatchSampleF) ----
    netF: str = "mlp_sample"
    netF_nc: int = 256

    # ---- CUT loss ----
    CUT_mode: str = "CUT"  # CUT | FastCUT
    lambda_GAN: float = 1.0
    lambda_NCE: float = 1.0
    nce_idt: bool = True  # identity NCE loss
    nce_layers: str = "0,4,8,12,16"  # layers for NCE feature extraction
    nce_T: float = 0.07  # temperature for contrastive loss
    num_patches: int = 256
    nce_includes_all_negatives_from_minibatch: bool = False
    flip_equivariance: bool = False
    gan_mode: str = "lsgan"  # lsgan | vanilla | wgangp

    # ---- training ----
    output_dir: str = "./outputs/cut"
    train_batch_size: int = 1
    eval_batch_size: int = 4
    n_epochs: int = 200  # epochs with initial lr
    n_epochs_decay: int = 200  # epochs to linearly decay lr to zero
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lr_policy: str = "linear"  # linear | step | cosine

    # ---- logging / checkpointing ----
    log_with: str = "tensorboard"
    save_model_epochs: int = 10
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 5
    resume_from_checkpoint: Optional[str] = None

    # ---- hardware ----
    mixed_precision: str = "fp16"
    dataloader_num_workers: int = 4
    seed: int = 42

    # ---- metric-based loss (MAVIC-T evaluation objective) ----
    use_mavic_loss: bool = False
    mavic_lpips_weight: float = 1.0
    mavic_l1_weight: float = 1.0
    mavic_loss_weight: float = 0.1

    # ---- sampling (evaluation) ----
    num_inference_steps: int = 1  # CUT is single-pass (no iterative denoising)


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
        output_dir="./outputs/cut_sar2eo",
        train_batch_size=4,
        eval_batch_size=16,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    """RGB-to-IR: 3-band → 1-band (native 1024×1024, default training at 256)."""
    cfg = TaskConfig(
        task_name="rgb2ir",
        source_channels=3,
        target_channels=1,
        model_channels=3,
        resolution=256,
        use_augmented=True,
        output_dir="./outputs/cut_rgb2ir",
        train_batch_size=4,
        eval_batch_size=4,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    """SAR-to-IR: 1-band → 1-band (native 1024×1024, default training at 256)."""
    cfg = TaskConfig(
        task_name="sar2ir",
        source_channels=1,
        target_channels=1,
        model_channels=1,
        resolution=256,
        use_augmented=True,
        output_dir="./outputs/cut_sar2ir",
        train_batch_size=4,
        eval_batch_size=4,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    """SAR-to-RGB: 1-band → 3-band (native 1024×1024, default training at 256)."""
    cfg = TaskConfig(
        task_name="sar2rgb",
        source_channels=1,
        target_channels=3,
        model_channels=3,
        resolution=256,
        use_augmented=True,
        output_dir="./outputs/cut_sar2rgb",
        train_batch_size=4,
        eval_batch_size=4,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
