# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Core Pix2Pix-Turbo trainer for MAVIC-T tasks.

Reference: Parmar, Gaurav, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu.
“One-Step Image Translation with Text-to-Image Models.” 2024.
https://doi.org/10.48550/arXiv.2403.12036.

.. note::
   **Lower priority**: the Img2Image-Turbo / Pix2Pix-Turbo method has been
   found less suitable for the MAVIC-T task compared to other baselines
   (CUT, DDBM).  Its code is retained for reference and future
   experimentation, but further implementation effort should focus on the
   other baselines first.

This module adapts the training logic from
``vendor/Img2Image-Turbo/src/train_pix2pix_turbo.py`` into a reusable
:class:`Pix2PixTurboTrainer` class driven by a
:class:`~examples.img2img_turbo.config.TaskConfig`.

The trainer uses the Accelerate framework for distributed / mixed-precision
training and follows the same structure as
:class:`~examples.ddbm.trainer.DDBMTrainer`.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from diffusers.utils import make_image_grid

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from datetime import timedelta

from pathlib import Path

from examples.ddbm.dataset_wrapper import PairedValDataset, resolve_paired_val_manifest
from .config import TaskConfig
from .dataset_wrapper import MavicTTurboDataset
from src.models.pix2pix_turbo import Pix2PixTurbo

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.metrics import MavicCriterion, MetricCalculator  # noqa: E402
from src.utils.training_utils import (  # noqa: E402
    build_accelerate_tracker_config,
    build_accelerate_tracker_init_kwargs,
    checkpoint_dir_sort_key,
    create_optimizer,
    enable_efficient_attention,
    lambda_repa_cosine,
    normalize_accelerate_log_with,
    save_checkpoint_diffusers,
    save_training_config,
    push_checkpoint_to_hub,
)

logger = get_logger(__name__, log_level="INFO")


class Pix2PixTurboTrainer:
    """End-to-end Pix2Pix-Turbo trainer driven by a :class:`TaskConfig`.

    Typical usage inside a per-task script::

        from .config import sar2eo_config
        from .trainer import Pix2PixTurboTrainer

        cfg = sar2eo_config()
        trainer = Pix2PixTurboTrainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg

    # ----- dataset -----------------------------------------------------------

    def build_datasets(self):
        """Return ``(train_dataset, val_dataset)``.
        
        Validation uses the paired val set when the manifest exists (resolved from
        cwd or project root), otherwise the test split."""
        resolved_paired = resolve_paired_val_manifest(
            getattr(self.cfg, "paired_val_manifest", None)
        )
        self._resolved_paired_val_manifest = resolved_paired
        paired_val_manifest_str = str(resolved_paired) if resolved_paired else getattr(
            self.cfg, "paired_val_manifest", None
        )

        train_ds = MavicTTurboDataset(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            model_channels=self.cfg.model_channels,
            use_augmented=self.cfg.use_augmented,
            use_random_crop=getattr(self.cfg, "use_random_crop", False),
            use_horizontal_flip=self.cfg.use_horizontal_flip,
            use_vertical_flip=self.cfg.use_vertical_flip,
            exclude_file=self.cfg.exclude_file,
            paired_val_manifest=paired_val_manifest_str,
            sar2rgb_sup_manifest=self.cfg.sar2rgb_sup_manifest if getattr(self.cfg, "use_sar2rgb_sup", False) else None,
            use_sar_despeckle=getattr(self.cfg, "use_sar_despeckle", False),
            sar_despeckle_kernel_size=getattr(self.cfg, "sar_despeckle_kernel_size", 5),
            sar_despeckle_strength=getattr(self.cfg, "sar_despeckle_strength", 0.6),
        )
        val_ds = None
        if self.cfg.validation_epochs is not None or self.cfg.validation_steps is not None:
            if resolved_paired is not None:
                val_ds = PairedValDataset(
                    manifest_path=resolved_paired,
                    resolution=self.cfg.resolution,
                    source_channels=self.cfg.source_channels,
                    target_channels=self.cfg.target_channels,
                )
                logger.info(
                    "Using paired val set for validation: %s (%d pairs)",
                    resolved_paired,
                    len(val_ds),
                )
            elif getattr(self.cfg, "paired_val_manifest", None):
                logger.warning(
                    "Paired val manifest not found at %s (tried cwd and project root) – falling back to test split",
                    self.cfg.paired_val_manifest,
                )
            if val_ds is None:
                try:
                    val_ds = MavicTTurboDataset(
                        task=self.cfg.task_name,
                        split="test",
                        resolution=self.cfg.resolution,
                        model_channels=self.cfg.model_channels,
                        with_target=False,
                        use_sar_despeckle=getattr(self.cfg, "use_sar_despeckle", False),
                        sar_despeckle_kernel_size=getattr(self.cfg, "sar_despeckle_kernel_size", 5),
                        sar_despeckle_strength=getattr(self.cfg, "sar_despeckle_strength", 0.6),
                    )
                    logger.info("Validation using test split.")
                except (ValueError, FileNotFoundError, RuntimeError):
                    logger.warning("Test split unavailable for %s – skipping validation", self.cfg.task_name)
        return train_ds, val_ds

    # ----- model -------------------------------------------------------------

    def build_model(self) -> Pix2PixTurbo:
        """Create the Pix2Pix-Turbo model."""
        return Pix2PixTurbo(
            pretrained_path=None,
            pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
            lora_rank_unet=self.cfg.lora_rank_unet,
            lora_rank_vae=self.cfg.lora_rank_vae,
        )

    # ----- loss --------------------------------------------------------------

    @staticmethod
    def compute_training_loss(model, batch, prompt_embeds, lambda_l2=1.0,
                              lambda_lpips=5.0, net_lpips=None,
                              mavic_criterion=None, mavic_loss_weight=0.1,
                              latent_target_encoder=None, lambda_latent=1.0,
                              rep_alignment_module=None, lambda_rep_alignment=0.1):
        """Compute the Pix2Pix-Turbo training loss for one batch.

        The loss combines:
        * L2 reconstruction loss (pixel-level)
        * LPIPS perceptual loss (when *net_lpips* is provided)
        * Optional MAVIC metric loss (LPIPS + L1 toward evaluation metric)
        * Optional latent-space L2 loss against a pre-trained VAE encoder
        * Optional representation alignment loss (REPA)
        """
        x_src = batch["conditioning_pixel_values"]
        x_tgt = batch["output_pixel_values"]

        # Source images are in [0, 1]; scale to [-1, 1] for the model
        x_src_norm = x_src * 2 - 1
        x_tgt_pred = model(x_src_norm, prompt_embeds)

        # L2 loss
        loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * lambda_l2
        loss = loss_l2
        extras = {
            "loss_mavic": None,
            "loss_latent": None,
            "loss_rep_alignment": None,
        }

        # LPIPS loss
        if net_lpips is not None:
            loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * lambda_lpips
            loss = loss + loss_lpips

        # Optional MAVIC metric loss
        if mavic_criterion is not None:
            pred_01 = (x_tgt_pred.float() + 1) * 0.5
            target_01 = (x_tgt.float() + 1) * 0.5
            pred_01 = pred_01.clamp(0, 1)
            target_01 = target_01.clamp(0, 1)
            mavic_loss = mavic_criterion(pred_01, target_01)
            loss = loss + mavic_loss_weight * mavic_loss
            extras["loss_mavic"] = mavic_loss.detach()

        # Optional latent-space L2 loss
        if latent_target_encoder is not None:
            latent_pred = latent_target_encoder.encode_with_grad(x_tgt_pred)
            with torch.no_grad():
                latent_tgt = latent_target_encoder.encode(x_tgt).detach()
            loss_latent = F.mse_loss(latent_pred.float(), latent_tgt.float())
            loss = loss + lambda_latent * loss_latent
            extras["loss_latent"] = loss_latent.detach()

        # Optional representation alignment loss (REPA)
        # REPA teacher encodes the *target* (ground-truth) image, not source.
        if rep_alignment_module is not None:
            with torch.no_grad():
                enc_feats = rep_alignment_module.extract_features(x_tgt)
            rep_loss = rep_alignment_module.compute_alignment_loss(x_tgt_pred, enc_feats)
            # Add lambda_rep_alignment * (rep_loss + 1): offset keeps total loss positive for
            # visualization (rep_loss is negative cosine similarity in [-1, 1]); gradient unchanged.
            loss = loss + lambda_rep_alignment * (rep_loss + 1.0)
            extras["loss_rep_alignment"] = rep_loss.detach()

        return loss, extras

    # ----- validation --------------------------------------------------------

    @torch.no_grad()
    def log_validation(self, model, prompt_embeds, val_dataloader, accelerator, global_step):
        """Generate and save test samples. Optionally evaluate on paired val set with LPIPS/L1/FID."""

        logger.info("Running validation at step %d …", global_step)
        cfg = self.cfg
        was_training = model.training
        model.eval()

        sample_dir = Path(self.cfg.output_dir) / "test_results" / f"step-{global_step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        first_grid = None

        has_paired_target = isinstance(val_dataloader.dataset, PairedValDataset)
        cols = 3 if has_paired_target else 2

        for batch_idx, batch in enumerate(val_dataloader):
            if isinstance(batch, dict):
                source = batch["conditioning_pixel_values"]
                target = None
            else:
                target, source = batch  # PairedValDataset: (target, source)
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1

            bsz = source_inp.shape[0]
            batch_embeds = prompt_embeds.expand(bsz, -1, -1)
            with accelerator.autocast():
                output = model(source_inp, batch_embeds)
            generated = (output + 1) * 0.5
            src_vis = source_01
            gen_vis = generated
            tgt_vis = target.to(accelerator.device) if has_paired_target and target is not None else None
            if gen_vis.shape[1] != src_vis.shape[1]:
                if gen_vis.shape[1] == 3 and src_vis.shape[1] == 1:
                    src_vis = src_vis.repeat(1, 3, 1, 1)
                elif gen_vis.shape[1] == 1 and src_vis.shape[1] == 3:
                    gen_vis = gen_vis.repeat(1, 3, 1, 1)
            if tgt_vis is not None and tgt_vis.shape[1] != gen_vis.shape[1]:
                if gen_vis.shape[1] == 3 and tgt_vis.shape[1] == 1:
                    tgt_vis = tgt_vis.repeat(1, 3, 1, 1)
                elif gen_vis.shape[1] == 1 and tgt_vis.shape[1] == 3:
                    gen_vis = gen_vis.repeat(1, 3, 1, 1)

            src_uint8 = (src_vis.clamp(0, 1) * 255).round().to(torch.uint8)
            gen_uint8 = (gen_vis.clamp(0, 1) * 255).round().to(torch.uint8)
            tgt_uint8 = (tgt_vis.clamp(0, 1) * 255).round().to(torch.uint8) if tgt_vis is not None else None

            src_uint8 = src_uint8.permute(0, 2, 3, 1).cpu().numpy()
            gen_uint8 = gen_uint8.permute(0, 2, 3, 1).cpu().numpy()
            tgt_uint8 = tgt_uint8.permute(0, 2, 3, 1).cpu().numpy() if tgt_uint8 is not None else None

            batch_images = []
            batch_size = len(src_uint8)
            for i in range(batch_size):
                src_arr = src_uint8[i]
                gen_arr = gen_uint8[i]
                if src_arr.shape[2] == 1:
                    src_arr = src_arr.squeeze(2)
                if gen_arr.shape[2] == 1:
                    gen_arr = gen_arr.squeeze(2)
                batch_images.append(Image.fromarray(src_arr).convert("RGB"))
                batch_images.append(Image.fromarray(gen_arr).convert("RGB"))
                if tgt_uint8 is not None:
                    tgt_arr = tgt_uint8[i]
                    if tgt_arr.shape[2] == 1:
                        tgt_arr = tgt_arr.squeeze(2)
                    batch_images.append(Image.fromarray(tgt_arr).convert("RGB"))

            grid = make_image_grid(batch_images, rows=batch_size, cols=cols)
            grid.save(sample_dir / f"batch_{batch_idx:03d}.png")
            if first_grid is None:
                first_grid = grid.copy()
            saved += batch_size

        logger.info("Saved %d test sample pairs to %s", saved, sample_dir)

        if first_grid is not None:
            from src.utils.training_utils import log_validation_images_to_trackers
            log_validation_images_to_trackers(accelerator, first_grid, global_step)

        # Evaluate on paired validation set using MAVIC-T metrics (LPIPS, L1, FID)
        metrics_result = {}
        manifest_path = getattr(self, "_resolved_paired_val_manifest", None) or (
            Path(cfg.paired_val_manifest) if getattr(cfg, "paired_val_manifest", None) else None
        )
        if manifest_path is not None and Path(manifest_path).is_file() and accelerator.is_main_process:
            metrics_result = self._evaluate_paired_val_metrics(
                model, prompt_embeds, accelerator, manifest_path=manifest_path
            )
            if metrics_result:
                logger.info(
                    "Paired val metrics: LPIPS=%.4f L1=%.4f score=%.4f (FID=%s)",
                    metrics_result.get("val_lpips", 0),
                    metrics_result.get("val_l1", 0),
                    metrics_result.get("val_score", 0),
                    metrics_result.get("val_fid", "N/A"),
                )

        if was_training:
            model.train()
        out = {"saved_samples": saved, "sample_dir": str(sample_dir)}
        out.update(metrics_result)
        return out

    def _evaluate_paired_val_metrics(self, model, prompt_embeds, accelerator, manifest_path=None):
        """Run inference on paired val set and compute MAVIC-T metrics (LPIPS, L1, FID)."""
        cfg = self.cfg
        manifest_path = Path(manifest_path) if manifest_path is not None else Path(cfg.paired_val_manifest)
        if not manifest_path.is_file():
            logger.warning("Paired val manifest not found: %s - skipping metric evaluation", manifest_path)
            return {}

        res = getattr(cfg, "output_resolution", None) or cfg.resolution
        paired_ds = PairedValDataset(
            manifest_path=manifest_path,
            resolution=res,
            source_channels=cfg.source_channels,
            target_channels=cfg.target_channels,
        )
        paired_loader = DataLoader(
            paired_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        metric_calc = MetricCalculator(device=str(accelerator.device), compute_fid=False)

        unwrapped = accelerator.unwrap_model(model)
        for _target, source in paired_loader:
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1

            bsz = source_inp.shape[0]
            batch_embeds = prompt_embeds.expand(bsz, -1, -1)
            with accelerator.autocast():
                output = unwrapped(source_inp, batch_embeds)
            generated = (output + 1) * 0.5

            target = _target.to(accelerator.device)
            pred_01 = generated.clamp(0, 1)
            tgt_01 = target.clamp(0, 1)
            if pred_01.shape[1] != tgt_01.shape[1]:
                if pred_01.shape[1] == 3 and tgt_01.shape[1] == 1:
                    tgt_01 = tgt_01.repeat(1, 3, 1, 1)
                elif pred_01.shape[1] == 1 and tgt_01.shape[1] == 3:
                    pred_01 = pred_01.repeat(1, 3, 1, 1)
            metric_calc.update(pred_01, tgt_01)

        m = metric_calc.compute()
        return {
            "val_lpips": m.lpips,
            "val_l1": m.l1,
            "val_score": m.score if m.score is not None else m.lpips + m.l1,
        }

    # ----- main training loop ------------------------------------------------

    def train(self):
        """Run the full training loop."""
        cfg = self.cfg
        checkpointing_steps = cfg.checkpointing_steps
        save_model_epochs = cfg.save_model_epochs
        if checkpointing_steps is not None and save_model_epochs is not None:
            logger.warning(
                "checkpointing_steps is set while save_model_epochs is enabled; "
                "epoch checkpoints take priority and step checkpoints will be skipped. "
                "Set save_model_epochs=None to enable step-based checkpointing."
            )
            checkpointing_steps = None

        # Accelerator setup
        logging_dir = os.path.join(cfg.output_dir, "logs")
        # log_with: "tensorboard" | "swanlab" | "wandb" | "all" | "tensorboard,swanlab" etc.
        log_with = normalize_accelerate_log_with(cfg.log_with)
        project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
        # TODO: Multi-GPU validation deadlock – accelerator.wait_for_everyone() / barrier hangs on some
        # setups (e.g. RTX 4090) with "No device id is provided via init_process_group or barrier".
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=7200), backend="nccl")]
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with=log_with,
            project_config=project_config,
            kwargs_handlers=kwargs_handlers,
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        if accelerator.is_main_process:
            os.makedirs(cfg.output_dir, exist_ok=True)
            os.makedirs(os.path.join(cfg.output_dir, "checkpoints"), exist_ok=True)

        # Build components
        logger.info(f"[{cfg.task_name}] Creating Pix2Pix-Turbo model "
                     f"(lora_unet={cfg.lora_rank_unet}, lora_vae={cfg.lora_rank_vae})")
        model = self.build_model()
        model.set_train()

        # Efficient attention (xformers / Flash Attention 2)
        enable_efficient_attention(
            model,
            enable_xformers=cfg.enable_xformers,
            enable_flash_attention_2=cfg.enable_flash_attention_2,
            _logger=logger,
        )

        if cfg.gradient_checkpointing:
            model.unet.enable_gradient_checkpointing()

        # Prompt embeddings (fixed for the entire training run)
        prompt_embeds = model.encode_prompt(cfg.prompt, accelerator.device)

        # LPIPS loss network
        net_lpips = None
        if cfg.lambda_lpips > 0:
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
            net_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
            net_lpips.requires_grad_(False)

        mavic_criterion = None
        if cfg.use_mavic_loss:
            mavic_criterion = MavicCriterion(
                lpips_weight=cfg.mavic_lpips_weight,
                l1_weight=cfg.mavic_l1_weight,
            )
            logger.info(f"[{cfg.task_name}] Using MAVIC metric loss "
                        f"(lpips_w={cfg.mavic_lpips_weight}, l1_w={cfg.mavic_l1_weight}, "
                        f"loss_w={cfg.mavic_loss_weight})")

        # Latent target encoder (RGB2IR ablation)
        latent_target_encoder = None
        if cfg.use_latent_target and cfg.latent_vae_path:
            from src.utils.latent_target import LatentTargetEncoder
            latent_target_encoder = LatentTargetEncoder(cfg.latent_vae_path)
            logger.info(f"[{cfg.task_name}] Using latent target encoder "
                        f"from {cfg.latent_vae_path} (lambda={cfg.lambda_latent})")

        # Representation alignment (REPA)
        # NOTE: REPA encodes the *target* (ground-truth) image, not the source.
        # Only SAR2RGB is currently supported (MaRS-Base-RGB encodes the RGB target).
        rep_alignment_module = None
        if cfg.use_rep_alignment and cfg.rep_alignment_model_path:
            from src.utils.rep_alignment import MaRSRGBAlignment
            if cfg.task_name == "sar2rgb":
                rep_alignment_module = MaRSRGBAlignment(cfg.rep_alignment_model_path)
            else:
                logger.warning(
                    "REPA is not applicable for task '%s' – no pre-trained "
                    "encoder for the target domain; skipping.", cfg.task_name,
                )
            if rep_alignment_module is not None:
                # Build projector with target_channels (features aligned to target encoder)
                rep_alignment_module.build_projector(cfg.target_channels)
                logger.info(
                    f"[{cfg.task_name}] Representation alignment enabled "
                    f"(model={cfg.rep_alignment_model_path}, "
                    f"lambda={cfg.lambda_rep_alignment}"
                    + (f"→{cfg.lambda_rep_alignment_end} cos decay over {cfg.lambda_rep_alignment_decay_steps} steps" if cfg.lambda_rep_alignment_decay_steps > 0 else "")
                    + ")"
                )

        # Optimizer (only trainable parameters)
        trainable_params = list(model.get_trainable_params())
        if rep_alignment_module is not None and rep_alignment_module.projector is not None:
            trainable_params += list(rep_alignment_module.projector.parameters())
        optimizer = create_optimizer(
            trainable_params,
            optimizer_type=cfg.optimizer_type,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            prodigy_d0=getattr(cfg, "prodigy_d0", 1e-6),
        )

        logger.info(f"[{cfg.task_name}] Loading dataset …")
        train_dataset, val_dataset = self.build_datasets()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.dataloader_num_workers,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.dataloader_num_workers,
        ) if val_dataset is not None else None
        # Disable log_validation on multi-GPU to avoid NCCL barrier deadlock
        if val_dataloader is not None and accelerator.num_processes > 1:
            logger.warning(
                "Disabling log_validation on multi-GPU to avoid NCCL barrier deadlock. "
                "Run validation separately on a single GPU."
            )
            val_dataloader = None

        from diffusers.optimization import get_scheduler as get_lr_scheduler
        total_steps = cfg.max_train_steps if cfg.max_train_steps else len(train_dataloader) * cfg.num_epochs
        lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
            num_training_steps=total_steps * cfg.gradient_accumulation_steps,
        )

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler,
        )
        if net_lpips is not None:
            net_lpips = net_lpips.to(accelerator.device)
        if mavic_criterion is not None:
            mavic_criterion = mavic_criterion.to(accelerator.device)
        if latent_target_encoder is not None:
            latent_target_encoder = latent_target_encoder.to(accelerator.device)
        if rep_alignment_module is not None:
            rep_alignment_module = rep_alignment_module.to(accelerator.device)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        if cfg.max_train_steps is None:
            cfg.max_train_steps = cfg.num_epochs * num_update_steps_per_epoch
        cfg.num_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

        if accelerator.is_main_process:
            project_name = f"turbo-{cfg.task_name}"
            tracker_config = build_accelerate_tracker_config(cfg)
            tracker_init_kwargs = build_accelerate_tracker_init_kwargs(cfg, project_name)
            accelerator.init_trackers(
                project_name,
                config=tracker_config,
                init_kwargs=tracker_init_kwargs or {},
            )

        def save_checkpoint_config_for(path: str) -> None:
            base_path = os.path.splitext(path)[0]
            save_training_config(cfg, base_path)

        global_step = 0
        first_epoch = 0

        # Resume
        if cfg.resume_from_checkpoint:
            path = cfg.resume_from_checkpoint
            if path == "latest":
                all_ckpt_dirs = [
                    d for d in os.listdir(cfg.output_dir)
                    if d.startswith("checkpoint") and checkpoint_dir_sort_key(d)[0] == 0
                ]
                dirs = sorted(all_ckpt_dirs, key=lambda x: checkpoint_dir_sort_key(x)[1])
                path = dirs[-1] if dirs else None
            if path is not None:
                if os.path.isabs(path) or os.path.sep in path:
                    load_path = os.path.abspath(path)
                else:
                    load_path = os.path.join(cfg.output_dir, path)
                accelerator.load_state(load_path)
                global_step = int(Path(path).name.split("-")[1])
                first_epoch = global_step // num_update_steps_per_epoch
                logger.info(f"Resumed from {path}")

        num_epochs_this_run = cfg.num_epochs - first_epoch
        logger.info("***** Running training *****")
        logger.info(f"  Task             = {cfg.task_name}")
        logger.info(f"  Num examples     = {len(train_dataset)}")
        logger.info(f"  Num epochs       = {num_epochs_this_run}")
        logger.info(f"  Batch size/dev   = {cfg.train_batch_size}")
        logger.info(f"  Total opt steps  = {cfg.max_train_steps}")

        progress_bar = tqdm(
            range(global_step, cfg.max_train_steps),
            disable=not accelerator.is_local_main_process,
            desc=f"Training {cfg.task_name}",
        )

        for epoch in range(first_epoch, cfg.num_epochs):
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    lambda_repa = (
                        lambda_repa_cosine(
                            global_step,
                            cfg.lambda_rep_alignment,
                            cfg.lambda_rep_alignment_end,
                            cfg.lambda_rep_alignment_decay_steps,
                        )
                        if rep_alignment_module is not None and cfg.lambda_rep_alignment_decay_steps > 0
                        else cfg.lambda_rep_alignment
                    )
                    loss, loss_extras = self.compute_training_loss(
                        accelerator.unwrap_model(model),
                        batch,
                        prompt_embeds,
                        lambda_l2=cfg.lambda_l2,
                        lambda_lpips=cfg.lambda_lpips,
                        net_lpips=net_lpips,
                        mavic_criterion=mavic_criterion,
                        mavic_loss_weight=cfg.mavic_loss_weight,
                        latent_target_encoder=latent_target_encoder,
                        lambda_latent=cfg.lambda_latent,
                        rep_alignment_module=rep_alignment_module,
                        lambda_rep_alignment=lambda_repa,
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }
                    if loss_extras.get("loss_rep_alignment") is not None:
                        logs["loss/repa"] = loss_extras["loss_rep_alignment"].item()
                        if cfg.lambda_rep_alignment_decay_steps > 0:
                            logs["lambda/repa"] = lambda_repa
                    if loss_extras.get("loss_mavic") is not None:
                        logs["loss/mavic"] = loss_extras["loss_mavic"].item()
                    if loss_extras.get("loss_latent") is not None:
                        logs["loss/latent"] = loss_extras["loss_latent"].item()
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    # Step-based validation (TODO: multi-GPU sync deadlock – see InitProcessGroupKwargs)
                    if (
                        val_dataloader is not None
                        and cfg.validation_steps is not None
                        and global_step % cfg.validation_steps == 0
                    ):
                        if accelerator.is_main_process:
                            val_result = self.log_validation(model, prompt_embeds, val_dataloader, accelerator, global_step)
                            if val_result:
                                accelerator.log(val_result, step=global_step)
                        accelerator.wait_for_everyone()

                    if (
                        checkpointing_steps is not None
                        and global_step % checkpointing_steps == 0
                        and accelerator.is_main_process
                    ):
                        checkpoints_dir = os.path.join(cfg.output_dir, "checkpoints")
                        outf = os.path.join(checkpoints_dir, f"model_{global_step}.pkl")
                        unwrapped_for_ckpt = accelerator.unwrap_model(model)
                        unwrapped_for_ckpt.save_model(outf)
                        save_checkpoint_config_for(outf)
                        # Also save diffusers-style structure for pipeline.from_pretrained()
                        diffusers_dir = os.path.join(checkpoints_dir, f"diffusers-{global_step}")
                        save_checkpoint_diffusers(
                            diffusers_dir,
                            unwrapped_for_ckpt.unet,
                            scheduler=unwrapped_for_ckpt.sched,
                            model_name="unet",
                            extra_state_dicts={
                                "vae": {k: v for k, v in unwrapped_for_ckpt.vae.state_dict().items()
                                        if "lora" in k or "skip" in k},
                            },
                        )
                        save_training_config(cfg, diffusers_dir)
                        logger.info(f"Saved checkpoint to {outf}")
                        if cfg.push_to_hub and cfg.hub_model_id:
                            push_checkpoint_to_hub(
                                outf,
                                hub_model_id=cfg.hub_model_id,
                                commit_message=f"img2img_turbo {cfg.task_name} step {global_step}",
                                path_in_repo=f"img2img_turbo/{cfg.task_name}/checkpoints/model_{global_step}.pkl",
                                request_timeout=30,
                            )

                        if cfg.checkpoints_total_limit is not None:
                            ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
                            ckpts = sorted(
                                [f for f in os.listdir(ckpt_dir)
                                 if f.endswith(".pkl") and f != "model_final.pkl"],
                                key=lambda x: int(x.split("_")[1].split(".")[0]),
                            )
                            for old in ckpts[:-cfg.checkpoints_total_limit]:
                                os.remove(os.path.join(ckpt_dir, old))

                if global_step >= cfg.max_train_steps:
                    break

            # Epoch-based validation (TODO: multi-GPU sync deadlock – see InitProcessGroupKwargs)
            if (
                val_dataloader is not None
                and cfg.validation_epochs is not None
                and (epoch + 1) % cfg.validation_epochs == 0
            ):
                if accelerator.is_main_process:
                    val_result = self.log_validation(model, prompt_embeds, val_dataloader, accelerator, global_step)
                    if val_result:
                        accelerator.log(val_result, step=global_step)
                accelerator.wait_for_everyone()

            # Save diffusers-style checkpoint at epoch boundary
            if (
                accelerator.is_main_process
                and save_model_epochs is not None
                and (epoch + 1) % save_model_epochs == 0
            ):
                unwrapped = accelerator.unwrap_model(model)
                epoch_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch + 1}")
                save_checkpoint_diffusers(
                    epoch_dir,
                    unwrapped.unet,
                    scheduler=unwrapped.sched,
                    model_name="unet",
                    extra_state_dicts={
                        "vae": {k: v for k, v in unwrapped.vae.state_dict().items()
                                if "lora" in k or "skip" in k},
                    },
                )
                save_training_config(cfg, epoch_dir)
                logger.info(f"Saved diffusers-style checkpoint at epoch {epoch + 1}")

                if cfg.push_to_hub and cfg.hub_model_id:
                    push_checkpoint_to_hub(
                        epoch_dir,
                        hub_model_id=cfg.hub_model_id,
                        commit_message=f"img2img_turbo {cfg.task_name} epoch {epoch + 1}",
                        path_in_repo=f"img2img_turbo/{cfg.task_name}/checkpoint-epoch-{epoch + 1}",
                        request_timeout=30,
                    )

                if cfg.checkpoints_total_limit is not None:
                    ckpts = sorted(
                        [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")],
                        key=checkpoint_dir_sort_key,
                    )
                    for old in ckpts[: -cfg.checkpoints_total_limit]:
                        shutil.rmtree(os.path.join(cfg.output_dir, old))

        # Save final model
        if accelerator.is_main_process:
            checkpoints_dir = os.path.join(cfg.output_dir, "checkpoints")
            outf = os.path.join(checkpoints_dir, "model_final.pkl")
            accelerator.unwrap_model(model).save_model(outf)
            save_checkpoint_config_for(outf)
            logger.info(f"Saved final model to {outf}")
            if cfg.push_to_hub and cfg.hub_model_id:
                push_checkpoint_to_hub(
                    outf,
                    hub_model_id=cfg.hub_model_id,
                    commit_message=f"img2img_turbo {cfg.task_name} final model",
                    path_in_repo=f"img2img_turbo/{cfg.task_name}/checkpoints/model_final.pkl",
                    request_timeout=30,
                )

        accelerator.end_training()
        logger.info(f"[{cfg.task_name}] Training complete!")
