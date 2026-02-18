"""Standalone SiD trainer built on shared DDBM training infrastructure."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.schedulers import SiDScheduler
from src.utils.training_utils import multiscale_weighted_mse
from src.models.unet_sid import create_sid_model

from examples.ddbm.trainer import DDBMTrainer


def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


class SIDTrainer(DDBMTrainer):
    """Standalone SiD baseline trainer for conditional image translation."""

    @property
    def baseline_name(self) -> str:
        return "sid"

    @property
    def pipeline_class_name(self) -> str:
        return "SIDPipeline"

    def get_validation_pipelines(self):
        from src.pipelines.sid import SIDPipeline

        return SIDPipeline, SIDPipeline

    def get_inference_kwargs(self, source_inp: torch.Tensor) -> dict:
        cfg = self.cfg
        return {
            "source_image": source_inp,
            "num_inference_steps": cfg.num_inference_steps,
            "output_type": "pt",
        }

    def build_model(self, image_size: int | None = None):
        cfg = self.cfg
        in_ch = cfg.latent_channels if cfg.use_latent_target else cfg.model_channels
        if image_size is None:
            image_size = cfg.resolution
        return create_sid_model(
            image_size=image_size,
            in_channels=in_ch,
            num_channels=cfg.num_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=cfg.attention_resolutions,
            dropout=cfg.dropout,
            condition_mode=cfg.condition_mode,
            channel_mult=cfg.channel_mult,
            attention_head_dim=getattr(cfg, "attention_head_dim", 64),
        )

    def build_scheduler(self):
        cfg = self.cfg
        return SiDScheduler(
            logsnr_min=getattr(cfg, "logsnr_min", -15.0),
            logsnr_max=getattr(cfg, "logsnr_max", 15.0),
            noise_d=float(getattr(cfg, "noise_d", 64.0)),
            image_d=float(getattr(cfg, "resolution", 256)),
            num_train_timesteps=int(getattr(cfg, "num_train_timesteps", 1000)),
            prediction_type=getattr(cfg, "prediction_type", "v"),
            clip_sample=bool(getattr(cfg, "clip_sample", True)),
        )

    def compute_training_loss(
        self,
        model,
        scheduler,
        x0,
        x_T,
        pred_mode="vp",  # kept for API compatibility with parent trainer
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
        del pred_mode  # SiD does not use DDBM pred_mode.

        bsz = x0.shape[0]
        device = x0.device
        dtype = x0.dtype

        # Sample continuous diffusion time t ~ U(0,1).
        t = torch.rand(bsz, device=device, dtype=dtype)
        logsnr_t = scheduler._logsnr_shifted_cosine(t)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))

        noise = torch.randn_like(x0)
        noisy_samples = scheduler.add_noise(x0, noise, logsnr_t)

        # Keep timestep embedding scale aligned with diffusers UNet convention.
        max_train_t = float(scheduler.config.num_train_timesteps - 1)
        t_embed = t * max_train_t
        pred = model(noisy_samples, t_embed, xT=x_T)

        alpha_b = _append_dims(alpha_t, x0.ndim)
        sigma_b = _append_dims(sigma_t, x0.ndim)
        prediction_type = scheduler.config.prediction_type

        if prediction_type == "eps":
            target = noise
            denoised = (noisy_samples - sigma_b * pred) / alpha_b
        elif prediction_type == "v":
            target = alpha_b * noise - sigma_b * x0
            denoised = alpha_b * noisy_samples - sigma_b * pred
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")

        use_multiscale_loss = bool(getattr(self.cfg, "use_multiscale_loss", True))
        multiscale_base_resolution = int(getattr(self.cfg, "multiscale_base_resolution", 32))

        if use_multiscale_loss:
            loss = multiscale_weighted_mse(
                pred,
                target,
                sample_weights=None,
                base_resolution=multiscale_base_resolution,
            )
        else:
            loss = F.mse_loss(pred, target)

        extras = {
            "loss_mavic": None,
            "loss_latent": None,
            "loss_rep_alignment": None,
        }

        decoded = None
        if latent_decode_fn is not None and (
            mavic_criterion is not None or rep_alignment_module is not None
        ):
            decoded = latent_decode_fn(denoised)
            if pixel_target is not None and decoded.shape[1] != pixel_target.shape[1]:
                if decoded.shape[1] == 3 and pixel_target.shape[1] == 1:
                    decoded = decoded.mean(dim=1, keepdim=True)
                elif decoded.shape[1] == 1 and pixel_target.shape[1] == 3:
                    decoded = decoded.repeat(1, 3, 1, 1)

        if mavic_criterion is not None:
            pred_for_metric = decoded if decoded is not None else denoised
            target_for_metric = pixel_target if pixel_target is not None else x0
            pred_01 = ((pred_for_metric + 1) * 0.5).clamp(0, 1)
            target_01 = ((target_for_metric + 1) * 0.5).clamp(0, 1)
            mavic_loss = mavic_criterion(pred_01, target_01)
            loss = loss + mavic_loss_weight * mavic_loss
            extras["loss_mavic"] = mavic_loss.detach()

        if latent_target_encoder is not None and not in_latent_space:
            latent_pred = latent_target_encoder.encode_with_grad(denoised)
            with torch.no_grad():
                latent_tgt = latent_target_encoder.encode(x0).detach()
            loss_latent = F.mse_loss(latent_pred.float(), latent_tgt.float())
            loss = loss + lambda_latent * loss_latent
            extras["loss_latent"] = loss_latent.detach()

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
