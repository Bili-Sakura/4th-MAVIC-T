"""Core CDTSDE trainer for MAVIC-T tasks."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger

from src.models.unet_cdtsde import create_model as create_cdtsde_model
from src.schedulers import CDTSDEScheduler

from examples.ddbm.trainer import DDBMTrainer
from examples.ddbm.dataset_wrapper import resolve_paired_val_manifest
from .dataset_wrapper import MavicTCDTSDEDataset, PairedValDataset


logger = get_logger(__name__, log_level="INFO")


class CDTSDETrainer(DDBMTrainer):
    """End-to-end CDTSDE trainer driven by :class:`examples.cdtsde.config.TaskConfig`."""

    @property
    def baseline_name(self) -> str:
        return "cdtsde"

    @property
    def pipeline_class_name(self) -> str:
        return "CDTSDEPipeline"

    def get_validation_pipelines(self):
        from src.pipelines.cdtsde import CDTSDEPipeline, CDTSDELatentPipeline

        return CDTSDEPipeline, CDTSDELatentPipeline

    def get_inference_kwargs(self, source_inp):
        cfg = self.cfg
        return {
            "source_image": source_inp,
            "num_inference_steps": cfg.num_inference_steps,
            "stochastic": cfg.stochastic,
            "apply_domain_shift": cfg.apply_domain_shift,
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
        self._resolved_paired_val_manifest = resolved_paired
        paired_val_manifest_str = str(resolved_paired) if resolved_paired else getattr(
            self.cfg, "paired_val_manifest", None
        )

        train_ds = MavicTCDTSDEDataset(
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
            sar2rgb_sup_manifest=self.cfg.sar2rgb_sup_manifest if getattr(self.cfg, "use_sar2rgb_sup", False) else None,
        )
        val_ds = None
        if (
            (self.cfg.validation_epochs is not None and self.cfg.validation_epochs > 0)
            or (self.cfg.validation_steps is not None and self.cfg.validation_steps > 0)
        ):
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
                    val_ds = MavicTCDTSDEDataset(
                        task=self.cfg.task_name,
                        split="test",
                        resolution=val_resolution,
                        source_channels=src_ch,
                        target_channels=tgt_ch,
                        with_target=False,
                    )
                    logger.info("Validation using test split.")
                except (ValueError, FileNotFoundError, RuntimeError):
                    logger.warning(
                        "Test split unavailable for %s - skipping validation",
                        self.cfg.task_name,
                    )
        return train_ds, val_ds

    # ----- model / scheduler -------------------------------------------------

    def build_model(self, image_size: int | None = None):
        in_ch = self.cfg.latent_channels if self.cfg.use_latent_target else self.cfg.model_channels
        if image_size is None:
            image_size = self.cfg.resolution
        return create_cdtsde_model(
            image_size=image_size,
            in_channels=in_ch,
            num_channels=self.cfg.num_channels,
            num_res_blocks=self.cfg.num_res_blocks,
            attention_resolutions=self.cfg.attention_resolutions,
            dropout=self.cfg.dropout,
            condition_mode=self.cfg.condition_mode,
            channel_mult=self.cfg.channel_mult,
            lambda_hidden_channels=self.cfg.lambda_hidden_channels,
        )

    def build_scheduler(self):
        return CDTSDEScheduler(
            num_train_timesteps=self.cfg.cdtsde_num_train_timesteps,
            beta_schedule=self.cfg.beta_schedule,
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
            eta_schedule=self.cfg.eta_schedule,
            eta_start=self.cfg.eta_start,
            eta_end=self.cfg.eta_end,
        )

    # ----- loss --------------------------------------------------------------

    @staticmethod
    def compute_training_loss(
        model,
        scheduler,
        x0,
        x_T,
        pred_mode="vp",
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
        bsz = x0.shape[0]
        device = x0.device

        t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noise = torch.randn_like(x0)

        lambda_linear = scheduler.train_etas.to(device=device, dtype=x0.dtype)[t]
        x_mix = model.mix_target_source(x0, x_T, lambda_linear=lambda_linear)
        x_t = scheduler.add_noise(x_mix, noise, t)

        pred_noise = model(x_t, t, xT=x_T)
        loss = F.mse_loss(pred_noise, noise)
        extras = {
            "loss_denoise": loss.detach(),
            "loss_mavic": None,
            "loss_latent": None,
            "loss_rep_alignment": None,
        }

        pred_x0 = scheduler.predict_start_from_noise(
            sample=x_t,
            timesteps=t,
            noise=pred_noise,
            use_inference_schedule=False,
        )
        lambda_hat = model.predict_lambda(lambda_linear, pred_x0.shape)
        pred_x0 = lambda_hat * x_T + (1.0 - lambda_hat) * pred_x0

        decoded = None
        if latent_decode_fn is not None and (mavic_criterion is not None or rep_alignment_module is not None):
            decoded = latent_decode_fn(pred_x0)
            if pixel_target is not None and decoded.shape[1] != pixel_target.shape[1]:
                if decoded.shape[1] == 3 and pixel_target.shape[1] == 1:
                    decoded = decoded.mean(dim=1, keepdim=True)
                elif decoded.shape[1] == 1 and pixel_target.shape[1] == 3:
                    decoded = decoded.repeat(1, 3, 1, 1)

        if mavic_criterion is not None:
            pred_for_metric = decoded if decoded is not None else pred_x0
            target_for_metric = pixel_target if pixel_target is not None else x0
            pred_01 = ((pred_for_metric + 1) * 0.5).clamp(0, 1)
            tgt_01 = ((target_for_metric + 1) * 0.5).clamp(0, 1)
            mavic_loss = mavic_criterion(pred_01, tgt_01)
            loss = loss + mavic_loss_weight * mavic_loss
            extras["loss_mavic"] = mavic_loss.detach()

        if latent_target_encoder is not None and not in_latent_space:
            latent_pred = latent_target_encoder.encode_with_grad(pred_x0)
            with torch.no_grad():
                latent_tgt = latent_target_encoder.encode(x0).detach()
            loss_latent = F.mse_loss(latent_pred.float(), latent_tgt.float())
            loss = loss + lambda_latent * loss_latent
            extras["loss_latent"] = loss_latent.detach()

        if rep_alignment_module is not None:
            target_for_enc = pixel_target if pixel_target is not None else x0
            with torch.no_grad():
                enc_feats = rep_alignment_module.extract_features(target_for_enc)
            rep_features = decoded if decoded is not None else pred_x0
            if rep_features.shape[1] != target_for_enc.shape[1]:
                if rep_features.shape[1] == 3 and target_for_enc.shape[1] == 1:
                    rep_features = rep_features.mean(dim=1, keepdim=True)
                elif rep_features.shape[1] == 1 and target_for_enc.shape[1] == 3:
                    rep_features = rep_features.repeat(1, 3, 1, 1)
            rep_loss = rep_alignment_module.compute_alignment_loss(rep_features, enc_feats)
            loss = loss + lambda_rep_alignment * (rep_loss + 1.0)
            extras["loss_rep_alignment"] = rep_loss.detach()

        return loss, extras

