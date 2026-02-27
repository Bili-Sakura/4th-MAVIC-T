"""Core BiBBDM trainer for MAVIC-T tasks.

Reference: Xue, Kaitao, Bo Li, Ziyi Liu, Zhifen He, Bin Liu, Congxuan Zhang,
and Yu-Kun Lai. “BiBBDM: Bidirectional Image Translation With Brownian Bridge
Diffusion Models.” IEEE Transactions on Pattern Analysis and Machine
Intelligence 47, no. 11 (2025): 10546–59. https://doi.org/10.1109/TPAMI.2025.3597667.

This module adapts the BiBBDM training logic (Brownian Bridge Diffusion)
into a reusable :class:`BiBBDMTrainer` class following the same pattern
as :mod:`examples.ddbm.trainer`.  Per-task scripts instantiate the
trainer with their own :class:`~examples.bibbdm.config.TaskConfig`.
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

from src.schedulers import BiBBDMScheduler
from .config import TaskConfig
from examples.ddbm.dataset_wrapper import PairedValDataset, resolve_paired_val_manifest
from .dataset_wrapper import MavicTBiBBDMDataset
from src.models.unet_bibbdm import create_model

from src.utils.metrics import MavicCriterion, MetricCalculator  # noqa: E402
from src.utils.training_utils import (  # noqa: E402
    build_accelerate_tracker_config,
    build_accelerate_tracker_init_kwargs,
    checkpoint_dir_sort_key,
    checkpoint_has_accelerator_state,
    create_optimizer,
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

class BiBBDMTrainer:
    """End-to-end BiBBDM trainer driven by a :class:`TaskConfig`.

    Typical usage inside a per-task script::

        from .config import sar2eo_config
        from .trainer import BiBBDMTrainer

        cfg = sar2eo_config()
        trainer = BiBBDMTrainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg

    # ----- baseline hooks ----------------------------------------------------

    @property
    def baseline_name(self) -> str:
        return "bibbdm"

    @property
    def pipeline_class_name(self) -> str:
        return "BiBBDMPipeline"

    def get_validation_pipelines(self):
        from src.pipelines.bibbdm import BiBBDMPipeline, BiBBDMLatentPipeline

        return BiBBDMPipeline, BiBBDMLatentPipeline

    def get_inference_kwargs(self, source_inp: torch.Tensor) -> dict:
        cfg = self.cfg
        num_steps = getattr(cfg, "num_inference_steps", cfg.sample_step)
        kwargs = {
            "source_image": source_inp,
            "direction": "b2a",
            "num_inference_steps": num_steps,
            "clip_denoised": cfg.clip_denoised,
            "output_type": "pt",
        }
        if not cfg.use_latent_target:
            kwargs["cfg_scale"] = getattr(cfg, "cfg_scale", 1.0)
        return kwargs

    # ----- dataset -----------------------------------------------------------

    def build_datasets(self):
        """Return ``(train_dataset, val_dataset)``.
        
        The validation loader now uses the *test* split and only serves as
        a source of inputs for sample generation.
        """
        # When training in latent space (Stage 1), we can load source and target
        # with their task-native channel counts.  LatentTargetEncoder will
        # handle the expansion to 3-ch if needed.
        if self.cfg.use_latent_target:
            src_ch = self.cfg.source_channels
            tgt_ch = self.cfg.target_channels
        else:
            # In pixel space, BiBBDM expects symmetric channel counts
            src_ch = self.cfg.model_channels
            tgt_ch = self.cfg.model_channels

        resolved_paired = resolve_paired_val_manifest(
            getattr(self.cfg, "paired_val_manifest", None)
        )
        self._resolved_paired_val_manifest = resolved_paired
        paired_val_manifest_str = str(resolved_paired) if resolved_paired else getattr(
            self.cfg, "paired_val_manifest", None
        )

        train_ds = MavicTBiBBDMDataset(
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
            elif getattr(self.cfg, "paired_val_manifest", None):
                logger.warning(
                    "Paired val manifest not found at %s (tried cwd and project root) – falling back to test split",
                    self.cfg.paired_val_manifest,
                )
            if val_ds is None:
                try:
                    val_ds = MavicTBiBBDMDataset(
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
        """Create the BiBBDM UNet model."""
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
            objective=self.cfg.objective,
            unet_type=self.cfg.unet_type,
        )

    def build_scheduler(self):
        """Create the Brownian Bridge noise scheduler."""
        return BiBBDMScheduler(
            num_timesteps=self.cfg.num_timesteps,
            mt_type=self.cfg.mt_type,
            m0=self.cfg.m0,
            mT=self.cfg.mT,
            eta=self.cfg.eta,
            var_scale=self.cfg.var_scale,
            skip_sample=self.cfg.skip_sample,
            sample_step=self.cfg.sample_step,
            sample_step_type=self.cfg.sample_step_type,
            objective=self.cfg.objective,
        )

    # ----- loss --------------------------------------------------------------

    @staticmethod
    def preprocess_batch(batch, device):
        """Scale a ``(target, source)`` batch from [0,1] to [-1,1]."""
        target = batch[0].to(device) * 2 - 1
        source = batch[1].to(device) * 2 - 1
        return target, source

    @staticmethod
    def apply_conditioning_dropout(
        condition: torch.Tensor,
        dropout_prob: float,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """Randomly replace conditioning with zeros for CFG training."""
        if dropout_prob <= 0.0:
            return condition, None, 0.0

        batch_size = condition.shape[0]
        drop_mask = torch.rand(batch_size, device=condition.device) < dropout_prob
        if not torch.any(drop_mask):
            return condition, None, 0.0

        dropped = condition.clone()
        dropped[drop_mask] = 0.0
        drop_ratio = drop_mask.float().mean().item()
        return dropped, drop_mask, drop_ratio

    @staticmethod
    def compute_training_loss(
        model,
        scheduler,
        target,
        source,
        objective="dlns",
        loss_type="l1",
        weight_obj=1.0,
        weight_a_recon=0.0,
        weight_b_recon=0.0,
        mavic_criterion=None,
        mavic_loss_weight=0.1,
        latent_target_encoder=None,
        lambda_latent=1.0,
        rep_alignment_module=None,
        lambda_rep_alignment=0.1,
        pixel_target=None,
        pixel_source=None,
        latent_decode_fn=None,
        conditioning_drop_mask: Optional[torch.Tensor] = None,
        in_latent_space: bool = False,
    ):
        """Compute the BiBBDM training loss for one batch.

        The loss follows the BiBBDM formulation:
        ``loss = w_obj * L_obj + w_a * L_a_recon + w_b * L_b_recon``
        where L_obj is the objective reconstruction error and L_a_recon, L_b_recon
        are optional endpoint reconstruction losses.
        """
        bsz = target.shape[0]
        device = target.device

        t = torch.randint(0, scheduler.num_timesteps, (bsz,), device=device).long()
        noise = torch.randn_like(target)

        # Forward (q-sample) — Brownian Bridge noising
        x_t = scheduler.add_noise(target, source, t, noise)

        # Training objective
        obj = scheduler.get_objective(target, source, t, noise)

        # Context for conditioning: use source image
        context = source
        if conditioning_drop_mask is not None:
            mask = conditioning_drop_mask.view(-1, 1, 1, 1)
            context = torch.where(mask, torch.zeros_like(context), context)

        # UNet prediction
        obj_recon = model(x_t, t, context=context)

        # Reconstruct endpoints
        target_recon = target if objective in ("gradb", "b") else \
            scheduler.predict_target_from_objective(x_t, source, t, obj_recon)
        source_recon = source if objective in ("grada", "a") else \
            scheduler.predict_source_from_objective(x_t, target, t, obj_recon)

        # Compute losses
        if loss_type == "l1":
            obj_loss = (obj - obj_recon).abs().mean()
            a_rec_loss = (target - target_recon).abs().mean()
            b_rec_loss = (source - source_recon).abs().mean()
        elif loss_type == "l2":
            obj_loss = F.mse_loss(obj, obj_recon)
            a_rec_loss = F.mse_loss(target, target_recon)
            b_rec_loss = F.mse_loss(source, source_recon)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        loss = weight_obj * obj_loss + weight_a_recon * a_rec_loss + weight_b_recon * b_rec_loss
        extras = {
            "loss_mavic": None,
            "loss_latent": None,
            "loss_rep_alignment": None,
        }

        decoded = None
        if latent_decode_fn is not None and (mavic_criterion is not None or rep_alignment_module is not None):
            decoded = latent_decode_fn(target_recon)
            if pixel_target is not None and decoded.shape[1] != pixel_target.shape[1]:
                if decoded.shape[1] == 3 and pixel_target.shape[1] == 1:
                    decoded = decoded.mean(dim=1, keepdim=True)
                elif decoded.shape[1] == 1 and pixel_target.shape[1] == 3:
                    decoded = decoded.repeat(1, 3, 1, 1)

        # Optional metric-based loss (LPIPS + L1)
        if mavic_criterion is not None:
            pred_for_metric = decoded if decoded is not None else target_recon
            target_for_metric = pixel_target if pixel_target is not None else target
            pred_01 = (pred_for_metric + 1) * 0.5
            tgt_01 = (target_for_metric + 1) * 0.5
            pred_01 = pred_01.clamp(0, 1)
            tgt_01 = tgt_01.clamp(0, 1)
            mavic_loss = mavic_criterion(pred_01, tgt_01)
            loss = loss + mavic_loss_weight * mavic_loss
            extras["loss_mavic"] = mavic_loss.detach()

        # Optional latent-space L2 loss
        if latent_target_encoder is not None and not in_latent_space:
            latent_pred = latent_target_encoder.encode_with_grad(target_recon)
            with torch.no_grad():
                latent_tgt = latent_target_encoder.encode(target).detach()
            loss_latent = F.mse_loss(latent_pred.float(), latent_tgt.float())
            loss = loss + lambda_latent * loss_latent
            extras["loss_latent"] = loss_latent.detach()

        # Optional representation alignment loss (REPA)
        # REPA teacher encodes the *target* (ground-truth) image, not source.
        if rep_alignment_module is not None:
            target_for_enc = pixel_target if pixel_target is not None else target
            with torch.no_grad():
                enc_feats = rep_alignment_module.extract_features(target_for_enc)
            rep_features = decoded if decoded is not None else target_recon
            if rep_features.shape[1] != target_for_enc.shape[1]:
                if rep_features.shape[1] == 3 and target_for_enc.shape[1] == 1:
                    rep_features = rep_features.mean(dim=1, keepdim=True)
                elif rep_features.shape[1] == 1 and target_for_enc.shape[1] == 3:
                    rep_features = rep_features.repeat(1, 3, 1, 1)
            rep_loss = rep_alignment_module.compute_alignment_loss(rep_features, enc_feats)
            # Add lambda_rep_alignment * (rep_loss + 1): offset keeps total loss positive for
            # visualization (rep_loss is negative cosine similarity in [-1, 1]); gradient unchanged.
            loss = loss + lambda_rep_alignment * (rep_loss + 1.0)
            extras["loss_rep_alignment"] = rep_loss.detach()

        return loss, extras

    # ----- validation --------------------------------------------------------

    @torch.no_grad()
    def log_validation(self, model, scheduler, val_dataloader, accelerator, global_step, latent_target_encoder=None):
        """Generate and save test samples. Optionally evaluate on paired val set with LPIPS/L1/FID."""
        PixelPipeline, LatentPipeline = self.get_validation_pipelines()

        logger.info("Running validation at step %d …", global_step)
        cfg = self.cfg
        was_training = model.training
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()

        if latent_target_encoder is not None:
            pipeline = LatentPipeline(unet=unwrapped, scheduler=scheduler, vae=latent_target_encoder.vae)
        else:
            pipeline = PixelPipeline(unet=unwrapped, scheduler=scheduler)
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
            target, source = batch  # PairedValDataset: (target, source); test split: target is zeros
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

        # Evaluate on paired validation set using MAVIC-T metrics (LPIPS, L1, FID)
        metrics_result = {}
        manifest_path = getattr(self, "_resolved_paired_val_manifest", None) or (
            Path(cfg.paired_val_manifest) if getattr(cfg, "paired_val_manifest", None) else None
        )
        if manifest_path is not None and Path(manifest_path).is_file() and accelerator.is_main_process:
            metrics_result = self._evaluate_paired_val_metrics(
                model, scheduler, pipeline, accelerator, latent_target_encoder, manifest_path=manifest_path
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
            unwrapped.train()
        out = {"saved_samples": saved, "sample_dir": str(sample_dir)}
        out.update(metrics_result)
        return out

    def _evaluate_paired_val_metrics(
        self, model, scheduler, pipeline, accelerator, latent_target_encoder, manifest_path=None
    ):
        """Run inference on paired val set and compute MAVIC-T metrics (LPIPS, L1, FID)."""
        cfg = self.cfg
        manifest_path = Path(manifest_path) if manifest_path is not None else Path(cfg.paired_val_manifest)
        if not manifest_path.is_file():
            logger.warning("Paired val manifest not found: %s - skipping metric evaluation", manifest_path)
            return {}

        if cfg.use_latent_target:
            src_ch, tgt_ch = cfg.source_channels, cfg.target_channels
        else:
            src_ch = tgt_ch = cfg.model_channels

        res = getattr(cfg, "output_resolution", None) or cfg.resolution
        paired_ds = PairedValDataset(
            manifest_path=manifest_path,
            resolution=res,
            source_channels=src_ch,
            target_channels=tgt_ch,
        )
        paired_loader = DataLoader(
            paired_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        metric_calc = MetricCalculator(device=str(accelerator.device), compute_fid=False)

        for _target, source in paired_loader:
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1

            with accelerator.autocast():
                pipeline_kwargs = self.get_inference_kwargs(source_inp)
                if latent_target_encoder is not None:
                    pipeline_kwargs["target_channels"] = cfg.target_channels
                result = pipeline(**pipeline_kwargs)
            generated = (result.images + 1) * 0.5

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
        cond_dropout_prob = float(getattr(cfg, "conditioning_dropout_prob", 0.0))
        if cond_dropout_prob < 0.0 or cond_dropout_prob > 1.0:
            raise ValueError(
                f"conditioning_dropout_prob must be in [0, 1], got {cond_dropout_prob}"
            )
        use_cond_dropout = cond_dropout_prob > 0.0 and not cfg.use_latent_target

        if cfg.task_name:
            cfg.output_dir = os.path.join(cfg.output_dir, self.baseline_name, cfg.task_name)

        checkpointing_steps = cfg.checkpointing_steps
        save_model_epochs = cfg.save_model_epochs
        if save_model_epochs is not None and save_model_epochs <= 0:
            save_model_epochs = None
        if checkpointing_steps is not None and save_model_epochs is not None:
            logger.warning(
                "checkpointing_steps is set while save_model_epochs is enabled; "
                "epoch checkpoints take priority and step checkpoints will be skipped. "
                "Set save_model_epochs=None to enable step-based checkpointing."
            )
            checkpointing_steps = None

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

        mavic_criterion = None
        if cfg.use_mavic_loss:
            mavic_criterion = MavicCriterion(
                lpips_weight=cfg.mavic_lpips_weight,
                l1_weight=cfg.mavic_l1_weight,
            )
            logger.info(
                f"[{cfg.task_name}] Using MAVIC metric loss "
                f"(lpips_w={cfg.mavic_lpips_weight}, l1_w={cfg.mavic_l1_weight}, "
                f"loss_w={cfg.mavic_loss_weight})"
            )

        latent_target_encoder = None
        latent_image_size = None
        if cfg.use_latent_target:
            if not cfg.latent_vae_path:
                raise ValueError("use_latent_target=True requires latent_vae_path to be set.")
            from src.utils.latent_target import LatentTargetEncoder
            latent_target_encoder = LatentTargetEncoder(cfg.latent_vae_path)
            vae_scale_factor = 2 ** (len(latent_target_encoder.vae.config.block_out_channels) - 1)
            if cfg.resolution % vae_scale_factor != 0:
                raise ValueError(
                    f"Resolution {cfg.resolution} is not divisible by VAE scale factor "
                    f"{vae_scale_factor} for latent training."
                )
            latent_image_size = cfg.resolution // vae_scale_factor
            logger.info(
                f"[{cfg.task_name}] Using latent target encoder from {cfg.latent_vae_path} "
                f"(lambda={cfg.lambda_latent})"
            )
            logger.info(
                f"[{cfg.task_name}] Latent resolution set to {latent_image_size} "
                f"(vae_scale_factor={vae_scale_factor})"
            )

        model_in_ch = cfg.latent_channels if cfg.use_latent_target else cfg.model_channels
        model_image_size = latent_image_size if latent_image_size is not None else cfg.resolution
        logger.info(f"[{cfg.task_name}] Creating model (channels={model_in_ch}, res={model_image_size})")
        model = self.build_model(image_size=model_image_size)
        scheduler = self.build_scheduler()

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

        ema_model = None
        if cfg.use_ema:
            from diffusers.training_utils import EMAModel
            ema_model = EMAModel(model.parameters(), decay=cfg.ema_decay, use_ema_warmup=True, model_cls=type(model))

        train_params = list(model.parameters())
        if rep_alignment_module is not None and rep_alignment_module.projector is not None:
            train_params += list(rep_alignment_module.projector.parameters())
        optimizer = create_optimizer(
            train_params,
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

        from diffusers.optimization import get_scheduler as get_lr_scheduler
        total_steps = cfg.max_train_steps if cfg.max_train_steps else len(train_dataloader) * cfg.num_epochs
        lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
            num_training_steps=total_steps * cfg.gradient_accumulation_steps,
        )

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        if cfg.use_ema and ema_model is not None:
            ema_model.to(accelerator.device)
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
            project_name = f"{self.baseline_name}-{cfg.task_name}"
            tracker_config = build_accelerate_tracker_config(cfg)
            tracker_init_kwargs = build_accelerate_tracker_init_kwargs(cfg, project_name)
            accelerator.init_trackers(
                project_name,
                config=tracker_config,
                init_kwargs=tracker_init_kwargs or {},
            )

        global_step = 0
        first_epoch = 0

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
                if checkpoint_has_accelerator_state(load_path):
                    accelerator.load_state(load_path)
                    global_step = int(Path(path).name.split("-")[1])
                    first_epoch = global_step // num_update_steps_per_epoch
                    logger.info(f"Resumed from {path} (full state)")
                    if rep_alignment_module is not None:
                        optimizer.state.clear()
                        logger.info("Cleared optimizer state (REPA projector shape may have changed)")
                else:
                    logger.warning(
                        "Checkpoint %s lacks accelerator state (optimizer.pt/bin); "
                        "skipping resume. Use a step checkpoint (checkpoint-N) for full resume.",
                        load_path,
                    )

        num_epochs_this_run = cfg.num_epochs - first_epoch
        logger.info("***** Running training *****")
        logger.info(f"  Task             = {cfg.task_name}")
        logger.info(f"  Num examples     = {len(train_dataset)}")
        logger.info(f"  Num epochs       = {num_epochs_this_run}")
        logger.info(f"  Batch size/dev   = {cfg.train_batch_size}")
        logger.info(f"  Total opt steps  = {cfg.max_train_steps}")
        if use_cond_dropout:
            logger.info(
                "  CFG cond dropout = %.3f (null condition = zeros)",
                cond_dropout_prob,
            )
        elif cond_dropout_prob > 0.0 and cfg.use_latent_target:
            logger.info("  CFG cond dropout skipped (enabled only for pixel-space training).")

        progress_bar = tqdm(range(global_step, cfg.max_train_steps), disable=not accelerator.is_local_main_process, desc=f"Training {cfg.task_name}")

        for epoch in range(first_epoch, cfg.num_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    target, source = self.preprocess_batch(batch, accelerator.device)
                    pixel_target = target
                    pixel_source = source
                    if cfg.use_latent_target and latent_target_encoder is not None:
                        with torch.no_grad():
                            target = latent_target_encoder.encode(pixel_target)
                            source = latent_target_encoder.encode(pixel_source)
                    cond_drop_mask = None
                    cond_drop_ratio = None
                    if use_cond_dropout:
                        source, cond_drop_mask, cond_drop_ratio = self.apply_conditioning_dropout(
                            source, cond_dropout_prob
                        )
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
                        model, scheduler, target, source,
                        objective=cfg.objective,
                        loss_type=cfg.loss_type,
                        weight_obj=cfg.weight_obj,
                        weight_a_recon=cfg.weight_a_recon,
                        weight_b_recon=cfg.weight_b_recon,
                        mavic_criterion=mavic_criterion,
                        mavic_loss_weight=cfg.mavic_loss_weight,
                        latent_target_encoder=latent_target_encoder,
                        lambda_latent=cfg.lambda_latent,
                        rep_alignment_module=rep_alignment_module,
                        lambda_rep_alignment=lambda_repa,
                        pixel_target=pixel_target if cfg.use_latent_target else None,
                        pixel_source=pixel_source if cfg.use_latent_target else None,
                        latent_decode_fn=latent_target_encoder.decode if cfg.use_latent_target else None,
                        conditioning_drop_mask=cond_drop_mask,
                        in_latent_space=cfg.use_latent_target,
                    )

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    if cfg.use_ema and ema_model is not None:
                        ema_model.step(model.parameters())
                    progress_bar.update(1)
                    global_step += 1

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch}
                    if loss_extras.get("loss_rep_alignment") is not None:
                        logs["loss/repa"] = loss_extras["loss_rep_alignment"].item()
                        if cfg.lambda_rep_alignment_decay_steps > 0:
                            logs["lambda/repa"] = lambda_repa
                    if loss_extras.get("loss_mavic") is not None:
                        logs["loss/mavic"] = loss_extras["loss_mavic"].item()
                    if loss_extras.get("loss_latent") is not None:
                        logs["loss/latent"] = loss_extras["loss_latent"].item()
                    if cond_drop_ratio is not None:
                        logs["cond/drop_ratio"] = cond_drop_ratio
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    # Step-based validation (TODO: multi-GPU sync deadlock – see InitProcessGroupKwargs)
                    if (
                        val_dataloader is not None
                        and cfg.validation_steps is not None
                        and global_step % cfg.validation_steps == 0
                    ):
                        if accelerator.is_main_process:
                            val_result = self.log_validation(
                                model, scheduler, val_dataloader, accelerator, global_step,
                                latent_target_encoder=latent_target_encoder if cfg.use_latent_target else None,
                            )
                            if val_result:
                                accelerator.log(val_result, step=global_step)
                        accelerator.wait_for_everyone()

                    if (
                        checkpointing_steps is not None
                        and global_step % checkpointing_steps == 0
                        and accelerator.is_main_process
                    ):
                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        # Save diffusers-style structure for pipeline.from_pretrained()
                        unwrapped_for_ckpt = accelerator.unwrap_model(model)
                        extra_sd_ckpt = {}
                        if cfg.use_ema and ema_model is not None:
                            model_param_names = list(unwrapped_for_ckpt.state_dict().keys())
                            shadow_params = ema_model.shadow_params
                            if len(model_param_names) == len(shadow_params):
                                extra_sd_ckpt["ema_unet"] = {
                                    name: param.clone().detach()
                                    for name, param in zip(model_param_names, shadow_params)
                                }
                        save_checkpoint_diffusers(
                            save_path,
                            unwrapped_for_ckpt,
                            scheduler=scheduler,
                            model_name="unet",
                            pipeline_class_name=self.pipeline_class_name,
                            extra_state_dicts=extra_sd_ckpt if extra_sd_ckpt else None,
                        )
                        accelerator.save_state(save_path)
                        save_training_config(cfg, save_path)
                        logger.info(f"Saved state to {save_path}")
                        if cfg.push_to_hub and cfg.hub_model_id:
                            push_checkpoint_to_hub(
                                save_path,
                                hub_model_id=cfg.hub_model_id,
                                commit_message=f"{self.baseline_name} {cfg.task_name} step {global_step}",
                                path_in_repo=f"{self.baseline_name}/{cfg.task_name}/checkpoint-{global_step}",
                                request_timeout=30,
                            )

                        if cfg.checkpoints_total_limit is not None:
                            ckpts = sorted(
                                [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")],
                                key=checkpoint_dir_sort_key,
                            )
                            for old in ckpts[: -cfg.checkpoints_total_limit]:
                                shutil.rmtree(os.path.join(cfg.output_dir, old))

                if global_step >= cfg.max_train_steps:
                    break

            # Epoch-based validation (TODO: multi-GPU sync deadlock – see InitProcessGroupKwargs)
            if (
                val_dataloader is not None
                and cfg.validation_epochs is not None
                and (epoch + 1) % cfg.validation_epochs == 0
            ):
                if accelerator.is_main_process:
                    val_result = self.log_validation(
                        model, scheduler, val_dataloader, accelerator, global_step,
                        latent_target_encoder=latent_target_encoder if cfg.use_latent_target else None,
                    )
                    if val_result:
                        accelerator.log(val_result, step=global_step)
                accelerator.wait_for_everyone()

            # Save at epoch boundary
            if (
                accelerator.is_main_process
                and save_model_epochs is not None
                and (epoch + 1) % save_model_epochs == 0
            ):
                unwrapped = accelerator.unwrap_model(model)
                epoch_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch + 1}")
                extra_sd = {}
                if cfg.use_ema and ema_model is not None:
                    model_param_names = list(unwrapped.state_dict().keys())
                    shadow_params = ema_model.shadow_params
                    if len(model_param_names) != len(shadow_params):
                        raise RuntimeError(
                            f"EMA shadow_params length ({len(shadow_params)}) != "
                            f"model state_dict keys ({len(model_param_names)})"
                        )
                    ema_state_dict = {
                        name: param.clone().detach()
                        for name, param in zip(model_param_names, shadow_params)
                    }
                    extra_sd["ema_unet"] = ema_state_dict
                save_checkpoint_diffusers(
                    epoch_dir,
                    unwrapped,
                    scheduler=scheduler,
                    model_name="unet",
                    pipeline_class_name=self.pipeline_class_name,
                    extra_state_dicts=extra_sd if extra_sd else None,
                )
                save_training_config(cfg, epoch_dir)
                logger.info(f"Saved model at epoch {epoch + 1}")

                if cfg.push_to_hub and cfg.hub_model_id:
                    push_checkpoint_to_hub(
                        epoch_dir,
                        hub_model_id=cfg.hub_model_id,
                        commit_message=f"{self.baseline_name} {cfg.task_name} epoch {epoch + 1}",
                        path_in_repo=f"{self.baseline_name}/{cfg.task_name}/checkpoint-epoch-{epoch + 1}",
                        request_timeout=30,
                    )

                if cfg.checkpoints_total_limit is not None:
                    ckpts = sorted(
                        [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")],
                        key=checkpoint_dir_sort_key,
                    )
                    for old in ckpts[: -cfg.checkpoints_total_limit]:
                        shutil.rmtree(os.path.join(cfg.output_dir, old))

        accelerator.end_training()
        logger.info(f"[{cfg.task_name}] Training complete!")
