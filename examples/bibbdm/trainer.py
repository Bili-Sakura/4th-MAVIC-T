"""Core BiBBDM trainer for MAVIC-T tasks.

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

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from datetime import timedelta

from src.schedulers import BiBBDMScheduler
from .config import TaskConfig
from .dataset_wrapper import MavicTBiBBDMDataset
from src.models.unet_bibbdm import create_model

from src.utils.metrics import MavicCriterion  # noqa: E402
from src.utils.training_utils import (  # noqa: E402
    create_optimizer,
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

        train_ds = MavicTBiBBDMDataset(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            source_channels=src_ch,
            target_channels=tgt_ch,
            use_augmented=self.cfg.use_augmented,
            use_horizontal_flip=self.cfg.use_horizontal_flip,
            use_vertical_flip=self.cfg.use_vertical_flip,
            exclude_file=self.cfg.exclude_file,
        )
        val_ds = None
        if self.cfg.validation_epochs is not None or self.cfg.validation_steps is not None:
            try:
                val_ds = MavicTBiBBDMDataset(
                    task=self.cfg.task_name,
                    split="test",
                    resolution=self.cfg.resolution,
                    source_channels=src_ch,
                    target_channels=tgt_ch,
                    with_target=False,
                )
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
            rep_loss = rep_alignment_module.compute_alignment_loss(rep_features, enc_feats)
            loss = loss + lambda_rep_alignment * rep_loss
            extras["loss_rep_alignment"] = rep_loss.detach()

        return loss, extras

    # ----- validation --------------------------------------------------------

    @torch.no_grad()
    def log_validation(self, model, scheduler, val_dataloader, accelerator, global_step, latent_target_encoder=None):
        """Generate and save test samples (first four inputs)."""
        from src.pipelines.bibbdm import BiBBDMPipeline, BiBBDMLatentPipeline

        logger.info("Running validation at step %d …", global_step)
        cfg = self.cfg
        was_training = model.training
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()

        if latent_target_encoder is not None:
            pipeline = BiBBDMLatentPipeline(unet=unwrapped, scheduler=scheduler, vae=latent_target_encoder.vae)
        else:
            pipeline = BiBBDMPipeline(unet=unwrapped, scheduler=scheduler)
        pipeline = pipeline.to(accelerator.device)

        sample_dir = Path(cfg.output_dir) / "test_results" / f"step-{global_step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        saved = 0

        for batch in val_dataloader:
            _zeros, source = batch
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1

            with accelerator.autocast():
                result = pipeline(
                    source_image=source_inp,
                    direction="b2a",
                    num_inference_steps=cfg.num_inference_steps,
                    clip_denoised=cfg.clip_denoised,
                    output_type="pt",
                    target_channels=cfg.target_channels,
                )
            generated = (result.images + 1) * 0.5

            src_vis = source_01
            gen_vis = generated
            if gen_vis.shape[1] != src_vis.shape[1]:
                if gen_vis.shape[1] == 3 and src_vis.shape[1] == 1:
                    src_vis = src_vis.repeat(1, 3, 1, 1)
                elif gen_vis.shape[1] == 1 and src_vis.shape[1] == 3:
                    gen_vis = gen_vis.repeat(1, 3, 1, 1)

            src_uint8 = (src_vis.clamp(0, 1) * 255).round().to(torch.uint8)
            gen_uint8 = (gen_vis.clamp(0, 1) * 255).round().to(torch.uint8)

            src_uint8 = src_uint8.permute(0, 2, 3, 1).cpu().numpy()
            gen_uint8 = gen_uint8.permute(0, 2, 3, 1).cpu().numpy()

            for src_arr, gen_arr in zip(src_uint8, gen_uint8):
                if saved >= 4:
                    break
                if src_arr.shape[2] == 1:
                    src_arr = src_arr.squeeze(2)
                if gen_arr.shape[2] == 1:
                    gen_arr = gen_arr.squeeze(2)
                concat = np.concatenate([src_arr, gen_arr], axis=1)
                Image.fromarray(concat).save(sample_dir / f"sample_{saved:02d}.png")
                saved += 1

            if saved >= 4:
                break

        logger.info("Saved %d test samples to %s", saved, sample_dir)

        if was_training:
            unwrapped.train()
        return {"saved_samples": saved, "sample_dir": str(sample_dir)}

    # ----- main training loop ------------------------------------------------

    def train(self):
        """Run the full training loop."""
        cfg = self.cfg

        if cfg.task_name:
            cfg.output_dir = os.path.join(cfg.output_dir, "bibbdm", cfg.task_name)

        checkpointing_steps = cfg.checkpointing_steps
        save_model_epochs = cfg.save_model_epochs
        if checkpointing_steps is not None and save_model_epochs is not None:
            logger.warning(
                "checkpointing_steps is set while save_model_epochs is enabled; "
                "epoch checkpoints take priority and step checkpoints will be skipped. "
                "Set save_model_epochs=None to enable step-based checkpointing."
            )
            checkpointing_steps = None

        logging_dir = os.path.join(cfg.output_dir, "logs")
        project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=7200))]
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with=cfg.log_with,
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
            logger.info(f"[{cfg.task_name}] Using MAVIC metric loss")

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
                logger.info(f"[{cfg.task_name}] Representation alignment enabled")

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
            tracker_config = {k: str(v) for k, v in vars(cfg).items()}
            accelerator.init_trackers(f"bibbdm-{cfg.task_name}", config=tracker_config)

        logger.info("***** Running training *****")
        logger.info(f"  Task             = {cfg.task_name}")
        logger.info(f"  Num examples     = {len(train_dataset)}")
        logger.info(f"  Num epochs       = {cfg.num_epochs}")
        logger.info(f"  Batch size/dev   = {cfg.train_batch_size}")
        logger.info(f"  Total opt steps  = {cfg.max_train_steps}")

        global_step = 0
        first_epoch = 0

        if cfg.resume_from_checkpoint:
            path = cfg.resume_from_checkpoint
            if path == "latest":
                dirs = sorted(
                    [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")],
                    key=lambda x: int(x.split("-")[1]),
                )
                path = dirs[-1] if dirs else None
            if path is not None:
                accelerator.load_state(os.path.join(cfg.output_dir, path))
                global_step = int(Path(path).name.split("-")[1])
                first_epoch = global_step // num_update_steps_per_epoch
                logger.info(f"Resumed from {path}")

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
                        lambda_rep_alignment=cfg.lambda_rep_alignment,
                        pixel_target=pixel_target if cfg.use_latent_target else None,
                        pixel_source=pixel_source if cfg.use_latent_target else None,
                        latent_decode_fn=latent_target_encoder.decode if cfg.use_latent_target else None,
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
                    if loss_extras.get("loss_mavic") is not None:
                        logs["loss/mavic"] = loss_extras["loss_mavic"].item()
                    if loss_extras.get("loss_latent") is not None:
                        logs["loss/latent"] = loss_extras["loss_latent"].item()
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    # Step-based validation
                    if (
                        val_dataloader is not None
                        and cfg.validation_steps is not None
                        and global_step % cfg.validation_steps == 0
                        and accelerator.is_main_process
                    ):
                        self.log_validation(
                            model, scheduler, val_dataloader, accelerator, global_step,
                            latent_target_encoder=latent_target_encoder if cfg.use_latent_target else None,
                        )

                    if (
                        checkpointing_steps is not None
                        and global_step % checkpointing_steps == 0
                        and accelerator.is_main_process
                    ):
                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        save_training_config(cfg, save_path)
                        logger.info(f"Saved state to {save_path}")

                        if cfg.checkpoints_total_limit is not None:
                            ckpts = sorted(
                                [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
                            )
                            for old in ckpts[: -cfg.checkpoints_total_limit]:
                                shutil.rmtree(os.path.join(cfg.output_dir, old))

                if global_step >= cfg.max_train_steps:
                    break

            # Epoch-based validation
            if (
                val_dataloader is not None
                and cfg.validation_epochs is not None
                and (epoch + 1) % cfg.validation_epochs == 0
                and accelerator.is_main_process
            ):
                self.log_validation(
                    model, scheduler, val_dataloader, accelerator, global_step,
                    latent_target_encoder=latent_target_encoder if cfg.use_latent_target else None,
                )

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
                    pipeline_class_name="BiBBDMPipeline",
                    extra_state_dicts=extra_sd if extra_sd else None,
                )
                save_training_config(cfg, epoch_dir)
                logger.info(f"Saved model at epoch {epoch + 1}")

                if cfg.push_to_hub and cfg.hub_model_id:
                    push_checkpoint_to_hub(
                        epoch_dir,
                        hub_model_id=cfg.hub_model_id,
                        commit_message=f"bibbdm {cfg.task_name} epoch {epoch + 1}",
                        path_in_repo=f"bibbdm/{cfg.task_name}/checkpoint-epoch-{epoch + 1}",
                    )

        accelerator.end_training()
        logger.info(f"[{cfg.task_name}] Training complete!")
