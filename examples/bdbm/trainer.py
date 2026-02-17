"""Core BDBM trainer for MAVIC-T tasks."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger

from src.schedulers import BDBMScheduler
from src.models.unet_bdbm import create_model as create_bdbm_model

from examples.bibbdm.trainer import BiBBDMTrainer
from examples.ddbm.dataset_wrapper import resolve_paired_val_manifest
from .dataset_wrapper import MavicTBDBMDataset, PairedValDataset


logger = get_logger(__name__, log_level="INFO")


class BDBMTrainer(BiBBDMTrainer):
    """End-to-end BDBM trainer driven by :class:`examples.bdbm.config.TaskConfig`."""

    @property
    def baseline_name(self) -> str:
        return "bdbm"

    @property
    def pipeline_class_name(self) -> str:
        return "BDBMPipeline"

    def get_validation_pipelines(self):
        from src.pipelines.bdbm import BDBMPipeline, BDBMLatentPipeline

        return BDBMPipeline, BDBMLatentPipeline

    def get_inference_kwargs(self, source_inp: torch.Tensor) -> dict:
        cfg = self.cfg
        num_steps = getattr(cfg, "num_inference_steps", cfg.sample_step)
        return {
            "source_image": source_inp,
            "direction": "b2a",
            "num_inference_steps": num_steps,
            "clip_denoised": cfg.clip_denoised,
            "output_type": "pt",
        }

    # ----- dataset -----------------------------------------------------------

    def build_datasets(self):
        if self.cfg.use_latent_target:
            src_ch = self.cfg.source_channels
            tgt_ch = self.cfg.target_channels
        else:
            src_ch = self.cfg.model_channels
            tgt_ch = self.cfg.model_channels

        resolved_paired = resolve_paired_val_manifest(
            getattr(self.cfg, "paired_val_manifest", None)
        )
        paired_val_manifest_str = str(resolved_paired) if resolved_paired else getattr(
            self.cfg, "paired_val_manifest", None
        )

        train_ds = MavicTBDBMDataset(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            source_channels=src_ch,
            target_channels=tgt_ch,
            use_augmented=self.cfg.use_augmented,
            use_horizontal_flip=self.cfg.use_horizontal_flip,
            use_vertical_flip=self.cfg.use_vertical_flip,
            exclude_file=self.cfg.exclude_file,
            paired_val_manifest=paired_val_manifest_str,
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
                    val_ds = MavicTBDBMDataset(
                        task=self.cfg.task_name,
                        split="test",
                        resolution=val_resolution,
                        source_channels=src_ch,
                        target_channels=tgt_ch,
                        with_target=False,
                    )
                    logger.info("Validation using test split.")
                except (ValueError, FileNotFoundError, RuntimeError):
                    logger.warning("Test split unavailable for %s – skipping validation", self.cfg.task_name)
        return train_ds, val_ds

    # ----- model / scheduler -------------------------------------------------

    def build_model(self, image_size: int | None = None):
        in_ch = self.cfg.latent_channels if self.cfg.use_latent_target else self.cfg.model_channels
        if image_size is None:
            image_size = self.cfg.resolution
        return create_bdbm_model(
            image_size=image_size,
            in_channels=in_ch,
            num_channels=self.cfg.num_channels,
            num_res_blocks=self.cfg.num_res_blocks,
            attention_resolutions=self.cfg.attention_resolutions,
            dropout=self.cfg.dropout,
            condition_mode=self.cfg.condition_mode,
            channel_mult=self.cfg.channel_mult,
            objective=self.cfg.objective,
        )

    def build_scheduler(self):
        return BDBMScheduler(
            num_timesteps=self.cfg.num_timesteps,
            mt_type=self.cfg.mt_type,
            eta=self.cfg.eta,
            max_var=self.cfg.max_var,
            skip_sample=self.cfg.skip_sample,
            sample_step=self.cfg.sample_step,
            sample_step_type=self.cfg.sample_step_type,
            objective=self.cfg.objective,
        )

    # ----- loss --------------------------------------------------------------

    @staticmethod
    def compute_training_loss(
        model,
        scheduler,
        target,
        source,
        objective="noise",
        loss_type="l2",
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
        in_latent_space: bool = False,
    ):
        bsz = target.shape[0]
        device = target.device

        t = torch.randint(0, scheduler.num_timesteps, (bsz,), device=device).long()
        noise = torch.randn_like(target)

        x_t = scheduler.add_noise(target, source, t, noise)
        obj = scheduler.get_objective(target, source, t, noise)

        condition_mode = getattr(model, "condition_mode", "concat")
        if condition_mode == "dual":
            mask = torch.randint(0, 2, (bsz,), device=device).float().view(bsz, 1, 1, 1)
            x_masked = target * mask
            y_masked = source * (1.0 - mask)
            context = torch.cat((x_masked, y_masked), dim=1)
        elif condition_mode == "concat":
            context = source
        else:
            context = None

        obj_recon = model(x_t, t, context=context)

        target_recon = target if objective in ("gradb", "b") else scheduler.predict_target_from_objective(
            x_t, source, t, obj_recon
        )
        source_recon = source if objective in ("grada", "a") else scheduler.predict_source_from_objective(
            x_t, target, t, obj_recon
        )

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

        if mavic_criterion is not None:
            pred_for_metric = decoded if decoded is not None else target_recon
            target_for_metric = pixel_target if pixel_target is not None else target
            pred_01 = ((pred_for_metric + 1) * 0.5).clamp(0, 1)
            tgt_01 = ((target_for_metric + 1) * 0.5).clamp(0, 1)
            mavic_loss = mavic_criterion(pred_01, tgt_01)
            loss = loss + mavic_loss_weight * mavic_loss
            extras["loss_mavic"] = mavic_loss.detach()

        if latent_target_encoder is not None and not in_latent_space:
            latent_pred = latent_target_encoder.encode_with_grad(target_recon)
            with torch.no_grad():
                latent_tgt = latent_target_encoder.encode(target).detach()
            loss_latent = F.mse_loss(latent_pred.float(), latent_tgt.float())
            loss = loss + lambda_latent * loss_latent
            extras["loss_latent"] = loss_latent.detach()

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
            loss = loss + lambda_rep_alignment * (rep_loss + 1.0)
            extras["loss_rep_alignment"] = rep_loss.detach()

        return loss, extras
