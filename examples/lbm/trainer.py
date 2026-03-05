# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Core LBM trainer for MAVIC-T tasks.

Reference: Chadebec, Clément, Onur Tasar, Sanjeev Sreetharan, and Benjamin Aubin.
"LBM: Latent Bridge Matching for Fast Image-to-Image Translation."
ICCV 2025 (Highlight). https://arxiv.org/abs/2503.07535

This module implements the training logic for Latent Bridge Matching (LBM)
into a reusable :class:`LBMTrainer` class.  Per-task scripts instantiate
the trainer with their own :class:`~examples.lbm.config.TaskConfig` and
can monkey-patch / sub-class any method for task-specific modifications.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from diffusers.utils import make_image_grid

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from datetime import timedelta

from src.schedulers import LBMScheduler
from .config import TaskConfig
from examples.ddbm.dataset_wrapper import PairedValDataset, resolve_paired_val_manifest
from examples.i2sb.dataset_wrapper import MavicTI2SBDataset
from src.models.unet.unet_2d import create_model

from src.utils.metrics import MavicCriterion, MetricCalculator  # noqa: E402
from src.utils.training_utils import (  # noqa: E402
    build_accelerate_tracker_config,
    build_accelerate_tracker_init_kwargs,
    checkpoint_dir_sort_key,
    checkpoint_has_accelerator_state,
    create_optimizer,
    enable_efficient_attention,
    lambda_repa_cosine,
    normalize_accelerate_log_with,
    save_checkpoint_diffusers,
    save_training_config,
    push_checkpoint_to_hub,
)

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LBMTrainer:
    """End-to-end LBM trainer driven by a :class:`TaskConfig`.

    Typical usage inside a per-task script::

        from .config import sar2eo_config
        from .trainer import LBMTrainer

        cfg = sar2eo_config()
        trainer = LBMTrainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg

    def get_inference_kwargs(self, source_inp: torch.Tensor) -> dict:
        cfg = self.cfg
        kwargs = {
            "source_image": source_inp,
            "num_inference_steps": cfg.num_inference_steps,
            "output_type": "pt",
        }
        if not cfg.use_latent_target:
            kwargs["cfg_scale"] = getattr(cfg, "cfg_scale", 1.0)
        return kwargs

    # ----- dataset -----------------------------------------------------------

    def build_datasets(self):
        """Return ``(train_dataset, val_dataset)``."""
        if self.cfg.use_latent_target:
            src_ch = self.cfg.source_channels
            tgt_ch = self.cfg.target_channels
        else:
            src_ch = self.cfg.model_channels
            tgt_ch = self.cfg.model_channels

        resolved_paired = resolve_paired_val_manifest(
            getattr(self.cfg, "paired_val_manifest", None)
        )
        self._resolved_paired_val_manifest = resolved_paired
        paired_val_manifest_str = str(resolved_paired) if resolved_paired else getattr(
            self.cfg, "paired_val_manifest", None
        )

        train_ds = MavicTI2SBDataset(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            source_channels=src_ch,
            target_channels=tgt_ch,
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
            val_resolution = getattr(self.cfg, "output_resolution", None) or self.cfg.resolution
            if resolved_paired is not None:
                val_ds = PairedValDataset(
                    manifest_path=resolved_paired,
                    resolution=val_resolution,
                    source_channels=src_ch,
                    target_channels=tgt_ch,
                )
                logger.info(
                    "Using paired val set for validation: %s (%d pairs)",
                    resolved_paired,
                    len(val_ds),
                )
            if val_ds is None:
                try:
                    val_ds = MavicTI2SBDataset(
                        task=self.cfg.task_name,
                        split="test",
                        resolution=val_resolution,
                        source_channels=src_ch,
                        target_channels=tgt_ch,
                        with_target=False,
                        use_sar_despeckle=getattr(self.cfg, "use_sar_despeckle", False),
                        sar_despeckle_kernel_size=getattr(self.cfg, "sar_despeckle_kernel_size", 5),
                        sar_despeckle_strength=getattr(self.cfg, "sar_despeckle_strength", 0.6),
                    )
                    logger.info("Validation using test split.")
                except (ValueError, FileNotFoundError, RuntimeError):
                    logger.warning("Test split unavailable for %s – skipping validation", self.cfg.task_name)
        return train_ds, val_ds

    # ----- model / scheduler -------------------------------------------------

    def build_model(self, image_size: int | None = None):
        """Create the LBM UNet model."""
        in_ch = self.cfg.latent_channels if self.cfg.use_latent_target else self.cfg.model_channels
        if image_size is None:
            image_size = self.cfg.resolution
        return create_model(
            image_size=image_size,
            in_channels=in_ch,
            num_channels=self.cfg.num_channels,
            num_res_blocks=self.cfg.num_res_blocks,
            attention_resolutions=self.cfg.attention_resolutions,
            dropout=self.cfg.dropout,
            condition_mode=self.cfg.condition_mode,
            channel_mult=self.cfg.channel_mult,
            backbone_type=self.cfg.backbone_type,
        )

    def build_scheduler(self):
        """Create the LBM scheduler."""
        return LBMScheduler(
            num_train_timesteps=self.cfg.num_train_timesteps,
            bridge_noise_sigma=self.cfg.bridge_noise_sigma,
            timestep_sampling=self.cfg.timestep_sampling,
            logit_mean=self.cfg.logit_mean,
            logit_std=self.cfg.logit_std,
            selected_timesteps=self.cfg.selected_timesteps,
            prob=self.cfg.prob,
        )

    # ----- loss --------------------------------------------------------------

    @staticmethod
    def preprocess_batch(batch, device):
        """Scale a ``(target, source)`` batch from [0,1] to [-1,1]."""
        x0 = batch[0].to(device) * 2 - 1
        x_T = batch[1].to(device) * 2 - 1
        return x0, x_T

    @staticmethod
    def apply_conditioning_dropout(
        condition: torch.Tensor,
        dropout_prob: float,
    ) -> tuple[torch.Tensor, float]:
        """Randomly replace conditioning with zeros for CFG training."""
        if dropout_prob <= 0.0:
            return condition, 0.0

        batch_size = condition.shape[0]
        drop_mask = torch.rand(batch_size, device=condition.device) < dropout_prob
        if not torch.any(drop_mask):
            return condition, 0.0

        dropped = condition.clone()
        dropped[drop_mask] = 0.0
        drop_ratio = drop_mask.float().mean().item()
        return dropped, drop_ratio

    @staticmethod
    def compute_training_loss(model, scheduler, x0, x_T, condition_mode="concat",
                              mavic_criterion=None, mavic_loss_weight=0.1,
                              latent_target_encoder=None, lambda_latent=1.0,
                              rep_alignment_module=None, lambda_rep_alignment=0.1,
                              pixel_target=None, pixel_source=None,
                              latent_decode_fn=None, in_latent_space: bool = False):
        """Compute the LBM bridge flow-matching loss for one batch.

        The loss targets the flow direction ``x_source - x_target``.  The model
        is trained to predict this direction from the interpolant:

            ``x_t = sigma * x_source + (1 - sigma) * x_target + bridge_noise``

        When *mavic_criterion* is provided the loss is augmented with a
        differentiable LPIPS + L1 term computed on the denoised prediction.

        When *latent_target_encoder* is provided an additional latent-space
        L2 loss is computed between the denoised prediction and the target.

        When *rep_alignment_module* is provided an additional representation
        alignment loss (REPA) is computed.
        """
        bsz = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        # Sample random timesteps
        timesteps = scheduler.sample_timesteps(bsz, device=device)

        # Get sigmas for the sampled timesteps
        sigmas = scheduler.get_sigmas(timesteps, n_dim=x0.ndim, device=device, dtype=dtype)

        # Create interpolant: noisy_sample between source (x_T) and target (x0)
        noisy_sample = scheduler.add_noise(x0, x_T, timesteps)

        # Compute training target: x_source - x_target
        target = scheduler.compute_target(x_T, x0)

        # Conditionally pass source as condition
        cond = x_T if condition_mode == "concat" else None

        # Timestep embedding: use timestep indices directly
        t_emb = timesteps.float()
        pred = model(noisy_sample, t_emb, cond=cond)

        # Denoising objective: MSE between predicted flow and target flow
        loss = F.mse_loss(pred, target)
        extras = {
            "loss_mavic": None,
            "loss_latent": None,
            "loss_rep_alignment": None,
        }

        # Compute denoised prediction for optional losses
        denoised = scheduler.compute_pred_x0(noisy_sample, pred, sigmas)

        decoded = None
        if latent_decode_fn is not None and (mavic_criterion is not None or rep_alignment_module is not None):
            decoded = latent_decode_fn(denoised)
            if pixel_target is not None and decoded.shape[1] != pixel_target.shape[1]:
                if decoded.shape[1] == 3 and pixel_target.shape[1] == 1:
                    decoded = decoded.mean(dim=1, keepdim=True)
                elif decoded.shape[1] == 1 and pixel_target.shape[1] == 3:
                    decoded = decoded.repeat(1, 3, 1, 1)

        # Optional metric-based loss
        if mavic_criterion is not None:
            pred_for_metric = decoded if decoded is not None else denoised
            target_for_metric = pixel_target if pixel_target is not None else x0
            pred_01 = (pred_for_metric + 1) * 0.5
            target_01 = (target_for_metric + 1) * 0.5
            pred_01 = pred_01.clamp(0, 1)
            target_01 = target_01.clamp(0, 1)
            mavic_loss = mavic_criterion(pred_01, target_01)
            loss = loss + mavic_loss_weight * mavic_loss
            extras["loss_mavic"] = mavic_loss.detach()

        # Optional latent-space L2 loss
        if latent_target_encoder is not None and not in_latent_space:
            latent_pred = latent_target_encoder.encode_with_grad(denoised)
            with torch.no_grad():
                latent_tgt = latent_target_encoder.encode(x0).detach()
            loss_latent = F.mse_loss(latent_pred.float(), latent_tgt.float())
            loss = loss + lambda_latent * loss_latent
            extras["loss_latent"] = loss_latent.detach()

        # Optional representation alignment loss (REPA)
        if rep_alignment_module is not None:
            target_for_enc = pixel_target if pixel_target is not None else x0
            with torch.no_grad():
                enc_feats = rep_alignment_module.extract_features(target_for_enc)
            rep_features = decoded if decoded is not None else denoised
            if rep_features.shape[1] != target_for_enc.shape[1]:
                if rep_features.shape[1] == 3 and target_for_enc.shape[1] == 1:
                    rep_features = rep_features.mean(dim=1, keepdim=True)
                elif rep_features.shape[1] == 1 and target_for_enc.shape[1] == 3:
                    rep_features = rep_features.repeat(1, 3, 1, 1)
            rep_loss = rep_alignment_module.compute_alignment_loss(rep_features, enc_feats)
            loss = loss + lambda_rep_alignment * (rep_loss + 1.0)
            extras["loss_rep_alignment"] = rep_loss.detach()

        return loss, extras

    # ----- validation --------------------------------------------------------

    @torch.no_grad()
    def log_validation(self, model, scheduler, val_dataloader, accelerator, global_step, latent_target_encoder=None):
        """Generate and save test samples. Optionally evaluate on paired val set with LPIPS/L1/FID."""
        from src.pipelines.lbm import LBMPipeline, LBMLatentPipeline

        logger.info("Running validation at step %d …", global_step)
        cfg = self.cfg
        was_training = model.training
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()

        if latent_target_encoder is not None:
            pipeline = LBMLatentPipeline(unet=unwrapped, scheduler=scheduler, vae=latent_target_encoder.vae)
        else:
            pipeline = LBMPipeline(unet=unwrapped, scheduler=scheduler)
        pipeline = pipeline.to(accelerator.device)

        sample_dir = Path(cfg.output_dir) / "test_results" / f"step-{global_step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        first_grid = None

        has_paired_target = isinstance(val_dataloader.dataset, PairedValDataset)
        cols = 3 if has_paired_target else 2

        for batch_idx, batch in enumerate(val_dataloader):
            if cfg.max_validation_batches is not None and batch_idx >= cfg.max_validation_batches:
                break
            target, source = batch
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1

            with accelerator.autocast():
                pipeline_kwargs = self.get_inference_kwargs(source_inp)
                if latent_target_encoder is not None:
                    pipeline_kwargs["target_channels"] = cfg.target_channels
                result = pipeline(**pipeline_kwargs)
            generated = (result.images + 1) * 0.5

            src_vis = source_01
            gen_vis = generated
            tgt_vis = target.to(accelerator.device) if has_paired_target else None
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

        if was_training:
            unwrapped.train()
        return {"saved_samples": saved, "sample_dir": str(sample_dir)}
