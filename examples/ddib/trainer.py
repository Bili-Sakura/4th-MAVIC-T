"""Core DDIB trainer for MAVIC-T tasks.

DDIB (Dual Diffusion Implicit Bridges, ICLR 2023) translates images between
two domains by training *independent unconditional* diffusion models on each
domain and connecting them through a shared DDIM latent space.

Reference: Su, Xuan, Jiaming Song, Chenlin Meng, and Stefano Ermon. “Dual
Diffusion Implicit Bridges for Image-to-Image Translation.” ICLR 2023.
https://openreview.net/forum?id=5HLoTvVGDe.

This module provides a :class:`DDIBTrainer` that trains **both** the source-
and target-domain diffusion models for a given task.  Each model is a standard
Gaussian diffusion model with DDIM-compatible noise prediction.

Per-task scripts instantiate the trainer with their own
:class:`~examples.ddib.config.TaskConfig` and can monkey-patch / sub-class
any method for task-specific modifications.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
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

from src.schedulers import DDIBScheduler
from examples.ddbm.dataset_wrapper import PairedValDataset, resolve_paired_val_manifest
from .config import TaskConfig
from .dataset_wrapper import MavicTDDIBDataset
from src.models.unet_ddib import create_model

from src.utils.metrics import MetricCalculator  # noqa: E402
from src.utils.training_utils import (  # noqa: E402
    build_accelerate_tracker_config,
    build_accelerate_tracker_init_kwargs,
    checkpoint_dir_sort_key,
    create_optimizer,
    lambda_repa_cosine,
    normalize_accelerate_log_with,
    save_checkpoint_diffusers,
    save_training_config,
    push_checkpoint_to_hub,
)

logger = get_logger(__name__, log_level="INFO")


class DDIBTrainer:
    """End-to-end DDIB trainer driven by a :class:`TaskConfig`.

    Trains two unconditional diffusion models (source domain + target domain)
    for image-to-image translation via DDIM encode→decode.

    Typical usage inside a per-task script::

        from .config import sar2eo_config
        from .trainer import DDIBTrainer

        cfg = sar2eo_config()
        trainer = DDIBTrainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg

    # ----- datasets ----------------------------------------------------------

    def build_datasets(self):
        """Return ``(source_dataset, target_dataset)`` for single-domain training."""
        resolved_paired = resolve_paired_val_manifest(
            getattr(self.cfg, "paired_val_manifest", None)
        )
        self._resolved_paired_val_manifest = resolved_paired
        paired_val_manifest_str = str(resolved_paired) if resolved_paired else getattr(
            self.cfg, "paired_val_manifest", None
        )
        common = dict(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            use_augmented=self.cfg.use_augmented,
            use_horizontal_flip=self.cfg.use_horizontal_flip,
            use_vertical_flip=self.cfg.use_vertical_flip,
            exclude_file=self.cfg.exclude_file,
            paired_val_manifest=paired_val_manifest_str,
        )
        source_ds = MavicTDDIBDataset(
            domain="source",
            model_channels=self.cfg.source_channels,
            **common,
        )
        target_ds = MavicTDDIBDataset(
            domain="target",
            model_channels=self.cfg.target_channels,
            **common,
        )
        return source_ds, target_ds

    # ----- model / scheduler -------------------------------------------------

    def build_model(self, in_channels: int, image_size: int | None = None):
        """Create an unconditional DDIB UNet for a single domain."""
        in_ch = self.cfg.latent_channels if self.cfg.use_latent_target else in_channels
        if image_size is None:
            image_size = self.cfg.resolution
        return create_model(
            image_size=image_size,
            in_channels=in_ch,
            num_channels=self.cfg.num_channels,
            num_res_blocks=self.cfg.num_res_blocks,
            attention_resolutions=self.cfg.attention_resolutions,
            dropout=self.cfg.dropout,
            learn_sigma=self.cfg.learn_sigma,
            channel_mult=self.cfg.channel_mult,
        )

    def build_scheduler(self):
        """Create the DDIB Gaussian diffusion scheduler."""
        return DDIBScheduler(
            num_train_timesteps=self.cfg.diffusion_steps,
            noise_schedule=self.cfg.noise_schedule,
            learn_sigma=self.cfg.learn_sigma,
            predict_xstart=self.cfg.predict_xstart,
            rescale_timesteps=self.cfg.rescale_timesteps,
        )

    # ----- validation --------------------------------------------------------

    @torch.no_grad()
    def log_validation(self, source_model, target_model, scheduler, val_dataloader, accelerator, global_step, latent_target_encoder=None):
        """Generate and save test samples. Optionally evaluate on paired val set with LPIPS/L1/FID."""
        from src.pipelines.ddib import DDIBPipeline, DDIBLatentPipeline

        logger.info("Running validation at step %d …", global_step)
        cfg = self.cfg
        src_unwrapped = accelerator.unwrap_model(source_model)
        tgt_unwrapped = accelerator.unwrap_model(target_model)
        src_training = src_unwrapped.training
        tgt_training = tgt_unwrapped.training
        src_unwrapped.eval()
        tgt_unwrapped.eval()

        if latent_target_encoder is not None:
            pipeline = DDIBLatentPipeline(
                source_unet=src_unwrapped,
                target_unet=tgt_unwrapped,
                scheduler=scheduler,
                vae=latent_target_encoder.vae,
            )
        else:
            pipeline = DDIBPipeline(
                source_unet=src_unwrapped,
                target_unet=tgt_unwrapped,
                scheduler=scheduler,
            )
        pipeline = pipeline.to(accelerator.device)

        sample_dir = Path(cfg.output_dir) / "test_results" / f"step-{global_step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        first_grid = None

        has_paired_target = isinstance(val_dataloader.dataset, PairedValDataset)
        cols = 3 if has_paired_target else 2

        for batch_idx, batch in enumerate(val_dataloader):
            target, source = batch  # PairedValDataset: (target, source); test split: target is zeros
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1

            with accelerator.autocast():
                result = pipeline(
                    source_image=source_inp,
                    num_inference_steps=cfg.num_inference_steps,
                    clip_denoised=True,
                    output_type="pt",
                )
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
                source_model, target_model, scheduler, pipeline, accelerator, latent_target_encoder, manifest_path=manifest_path
            )
            if metrics_result:
                logger.info(
                    "Paired val metrics: LPIPS=%.4f L1=%.4f score=%.4f (FID=%s)",
                    metrics_result.get("val_lpips", 0),
                    metrics_result.get("val_l1", 0),
                    metrics_result.get("val_score", 0),
                    metrics_result.get("val_fid", "N/A"),
                )

        if src_training:
            src_unwrapped.train()
        if tgt_training:
            tgt_unwrapped.train()
        out = {"saved_samples": saved, "sample_dir": str(sample_dir)}
        out.update(metrics_result)
        return out

    def _evaluate_paired_val_metrics(
        self, source_model, target_model, scheduler, pipeline, accelerator, latent_target_encoder, manifest_path=None
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
                result = pipeline(
                    source_image=source_inp,
                    num_inference_steps=cfg.num_inference_steps,
                    clip_denoised=True,
                    output_type="pt",
                )
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

    # ----- single-domain training loop ---------------------------------------

    def _train_single_domain(
        self,
        domain_label: str,
        model: torch.nn.Module,
        scheduler: DDIBScheduler,
        dataset: MavicTDDIBDataset,
        accelerator: Accelerator,
        rep_alignment_module=None,
        lambda_rep_alignment: float = 0.1,
        latent_target_encoder=None,
    ):
        """Train one unconditional diffusion model on a single domain.

        Parameters
        ----------
        domain_label : str
            ``"source"`` or ``"target"``, used for logging and save paths.
        model : DDIBUNet
            The model to train.
        scheduler : DDIBScheduler
            Gaussian diffusion scheduler.
        dataset : MavicTDDIBDataset
            Single-domain dataset.
        accelerator : Accelerator
            Shared Accelerator instance.
        rep_alignment_module : optional
            Frozen encoder + trainable projector for REPA loss.
        lambda_rep_alignment : float
            Weight for the representation alignment loss.
        """
        cfg = self.cfg
        domain_output_dir = os.path.join(cfg.output_dir, f"ddib_{domain_label}", cfg.task_name)
        os.makedirs(domain_output_dir, exist_ok=True)

        checkpointing_steps = cfg.checkpointing_steps
        save_model_epochs = cfg.save_model_epochs
        if checkpointing_steps is not None and save_model_epochs is not None:
            checkpointing_steps = None

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

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.dataloader_num_workers,
            drop_last=True,
        )

        from diffusers.optimization import get_scheduler as get_lr_scheduler
        total_steps = cfg.max_train_steps if cfg.max_train_steps else len(dataloader) * cfg.num_epochs
        lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
            num_training_steps=total_steps * cfg.gradient_accumulation_steps,
        )

        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, dataloader, lr_scheduler
        )
        if cfg.use_ema and ema_model is not None:
            ema_model.to(accelerator.device)
        if rep_alignment_module is not None:
            rep_alignment_module = rep_alignment_module.to(accelerator.device)

        num_update_steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
        local_max_train_steps = cfg.max_train_steps if cfg.max_train_steps else cfg.num_epochs * num_update_steps_per_epoch
        num_epochs = math.ceil(local_max_train_steps / num_update_steps_per_epoch)

        if accelerator.is_main_process:
            project_name = f"ddib-{domain_label}-{cfg.task_name}"
            tracker_config = build_accelerate_tracker_config(cfg)
            tracker_init_kwargs = build_accelerate_tracker_init_kwargs(cfg, project_name)
            accelerator.init_trackers(
                project_name,
                config=tracker_config,
                init_kwargs=tracker_init_kwargs or {},
            )

        global_step = 0
        first_epoch = 0

        # Resume
        if cfg.resume_from_checkpoint:
            path = cfg.resume_from_checkpoint
            if path == "latest":
                all_ckpt_dirs = [
                    d for d in os.listdir(domain_output_dir)
                    if d.startswith("checkpoint") and checkpoint_dir_sort_key(d)[0] == 0
                ]
                dirs = sorted(all_ckpt_dirs, key=lambda x: checkpoint_dir_sort_key(x)[1])
                path = dirs[-1] if dirs else None
            if path is not None:
                if os.path.isabs(path) or os.path.sep in path:
                    full_path = os.path.abspath(path)
                else:
                    full_path = os.path.join(domain_output_dir, path)
                if os.path.isdir(full_path):
                    accelerator.load_state(full_path)
                    global_step = int(Path(path).name.split("-")[1])
                    first_epoch = global_step // num_update_steps_per_epoch
                    logger.info(f"Resumed {domain_label} from {path}")

        num_epochs_this_run = num_epochs - first_epoch
        logger.info(f"***** Training DDIB {domain_label} model *****")
        logger.info(f"  Task             = {cfg.task_name}")
        logger.info(f"  Domain           = {domain_label}")
        logger.info(f"  Num examples     = {len(dataset)}")
        logger.info(f"  Num epochs       = {num_epochs_this_run}")
        logger.info(f"  Batch size/dev   = {cfg.train_batch_size}")
        logger.info(f"  Total opt steps  = {local_max_train_steps}")

        progress_bar = tqdm(
            range(global_step, local_max_train_steps),
            disable=not accelerator.is_local_main_process,
            desc=f"Training DDIB {domain_label} ({cfg.task_name})",
        )

        for epoch in range(first_epoch, num_epochs):
            model.train()
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(model):
                    # batch is a single tensor (B, C, H, W) in [0, 1]
                    x_0 = batch.to(accelerator.device) * 2 - 1  # scale to [-1, 1]
                    pixel_x0 = x_0
                    if cfg.use_latent_target and latent_target_encoder is not None:
                        with torch.no_grad():
                            x_0 = latent_target_encoder.encode(pixel_x0)

                    rep_loss = None
                    if rep_alignment_module is not None:
                        loss, pred_xstart = scheduler.compute_training_loss(
                            model, x_0, return_pred_xstart=True,
                        )
                        # REPA teacher encodes the clean target image (pixel_x0).
                        # DDIB trains per-domain; this is only meaningful for
                        # the target-domain model (e.g. RGB in SAR→RGB).
                        with torch.no_grad():
                            enc_feats = rep_alignment_module.extract_features(pixel_x0)
                        if cfg.use_latent_target and latent_target_encoder is not None:
                            pred_for_align = latent_target_encoder.decode(
                                pred_xstart,
                                target_channels=pixel_x0.shape[1],
                            )
                        else:
                            pred_for_align = pred_xstart
                        rep_loss = rep_alignment_module.compute_alignment_loss(
                            pred_for_align, enc_feats,
                        )
                        lambda_repa = (
                            lambda_repa_cosine(
                                global_step,
                                cfg.lambda_rep_alignment,
                                cfg.lambda_rep_alignment_end,
                                cfg.lambda_rep_alignment_decay_steps,
                            )
                            if cfg.lambda_rep_alignment_decay_steps > 0
                            else cfg.lambda_rep_alignment
                        )
                        # Add (rep_loss + 1): offset keeps total loss positive for visualization.
                        loss = loss + lambda_repa * (rep_loss + 1.0)
                    else:
                        loss = scheduler.compute_training_loss(model, x_0)

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
                    if rep_loss is not None:
                        logs["loss/repa"] = rep_loss.detach().item()
                        if cfg.lambda_rep_alignment_decay_steps > 0:
                            logs["lambda/repa"] = lambda_repa
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if (
                        checkpointing_steps is not None
                        and global_step % checkpointing_steps == 0
                        and accelerator.is_main_process
                    ):
                        save_path = os.path.join(domain_output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # Also save diffusers-style structure for pipeline.from_pretrained()
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
                            extra_state_dicts=extra_sd_ckpt if extra_sd_ckpt else None,
                        )
                        save_training_config(cfg, save_path)
                        logger.info(f"Saved {domain_label} state to {save_path}")
                        if cfg.push_to_hub and cfg.hub_model_id:
                            push_checkpoint_to_hub(
                                save_path,
                                hub_model_id=cfg.hub_model_id,
                                commit_message=f"ddib {domain_label} {cfg.task_name} step {global_step}",
                                path_in_repo=f"ddib/{domain_label}/{cfg.task_name}/checkpoint-{global_step}",
                                request_timeout=30,
                            )

                        if cfg.checkpoints_total_limit is not None:
                            ckpts = sorted(
                                [d for d in os.listdir(domain_output_dir) if d.startswith("checkpoint")],
                                key=checkpoint_dir_sort_key,
                            )
                            for old in ckpts[: -cfg.checkpoints_total_limit]:
                                shutil.rmtree(os.path.join(domain_output_dir, old))

                if global_step >= local_max_train_steps:
                    break

            # Save at epoch boundary
            if (
                accelerator.is_main_process
                and save_model_epochs is not None
                and (epoch + 1) % save_model_epochs == 0
            ):
                unwrapped = accelerator.unwrap_model(model)
                epoch_dir = os.path.join(domain_output_dir, f"checkpoint-epoch-{epoch + 1}")
                extra_sd = {}
                if cfg.use_ema and ema_model is not None:
                    model_param_names = list(unwrapped.state_dict().keys())
                    shadow_params = ema_model.shadow_params
                    assert len(model_param_names) == len(shadow_params), (
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
                    extra_state_dicts=extra_sd if extra_sd else None,
                )
                save_training_config(cfg, epoch_dir)
                logger.info(f"Saved {domain_label} model at epoch {epoch + 1}")

                if cfg.push_to_hub and cfg.hub_model_id:
                    push_checkpoint_to_hub(
                        epoch_dir,
                        hub_model_id=cfg.hub_model_id,
                        commit_message=f"ddib {domain_label} {cfg.task_name} epoch {epoch + 1}",
                        path_in_repo=f"ddib/{domain_label}/{cfg.task_name}/checkpoint-epoch-{epoch + 1}",
                        request_timeout=30,
                    )

                if cfg.checkpoints_total_limit is not None:
                    ckpts = sorted(
                        [d for d in os.listdir(domain_output_dir) if d.startswith("checkpoint")],
                        key=checkpoint_dir_sort_key,
                    )
                    for old in ckpts[: -cfg.checkpoints_total_limit]:
                        shutil.rmtree(os.path.join(domain_output_dir, old))

        accelerator.end_training()
        logger.info(f"[{cfg.task_name}] DDIB {domain_label} training complete!")

    # ----- main training entry point -----------------------------------------

    def train(self):
        """Run the full DDIB training loop (both source and target domain models)."""
        cfg = self.cfg

        # Accelerator setup (shared)
        logging_dir = os.path.join(cfg.output_dir, "logs")
        # log_with: "tensorboard" | "swanlab" | "wandb" | "all" | "tensorboard,swanlab" etc.
        log_with = normalize_accelerate_log_with(cfg.log_with)
        project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=7200))]
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

        # Build datasets
        logger.info(f"[{cfg.task_name}] Loading datasets …")
        source_dataset, target_dataset = self.build_datasets()

        # Build scheduler (shared between both models)
        scheduler = self.build_scheduler()

        # Latent target encoder (ablation / latent-space training)
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
        if latent_target_encoder is not None:
            latent_target_encoder = latent_target_encoder.to(accelerator.device)

        # --- Phase 1: Train source-domain model ---
        logger.info(f"[{cfg.task_name}] === Phase 1: Training source-domain model ===")
        source_model = self.build_model(
            in_channels=cfg.source_channels,
            image_size=latent_image_size if latent_image_size is not None else cfg.resolution,
        )
        self._train_single_domain(
            "source", source_model, scheduler, source_dataset, accelerator,
            rep_alignment_module=rep_alignment_module,
            lambda_rep_alignment=cfg.lambda_rep_alignment,
            latent_target_encoder=latent_target_encoder,
        )


        # --- Phase 2: Train target-domain model ---
        logger.info(f"[{cfg.task_name}] === Phase 2: Training target-domain model ===")
        target_model = self.build_model(
            in_channels=cfg.target_channels,
            image_size=latent_image_size if latent_image_size is not None else cfg.resolution,
        )
        self._train_single_domain(
            "target", target_model, scheduler, target_dataset, accelerator,
            rep_alignment_module=rep_alignment_module,
            lambda_rep_alignment=cfg.lambda_rep_alignment,
            latent_target_encoder=latent_target_encoder,
        )

        # Post-training validation (requires both models)
        if (
            accelerator.is_main_process
            and (cfg.validation_epochs is not None or cfg.validation_steps is not None)
        ):
            from examples.ddbm.dataset_wrapper import MavicTDDBMDataset, PairedValDataset
            resolved_paired = resolve_paired_val_manifest(getattr(cfg, "paired_val_manifest", None))
            val_ds = None
            if resolved_paired is not None:
                val_ds = PairedValDataset(
                    manifest_path=resolved_paired,
                    resolution=cfg.resolution,
                    source_channels=cfg.source_channels,
                    target_channels=cfg.target_channels,
                )
                logger.info(
                    "Using paired val set for validation: %s (%d pairs)",
                    resolved_paired,
                    len(val_ds),
                )
            elif getattr(cfg, "paired_val_manifest", None):
                logger.warning(
                    "Paired val manifest not found at %s (tried cwd and project root) – falling back to test split",
                    cfg.paired_val_manifest,
                )
            if val_ds is None:
                try:
                    val_ds = MavicTDDBMDataset(
                        task=cfg.task_name,
                        split="test",
                        resolution=cfg.resolution,
                        model_channels=cfg.source_channels,
                        with_target=False,
                    )
                    logger.info("Validation using test split.")
                except (ValueError, FileNotFoundError, RuntimeError):
                    pass
            if val_ds is not None:
                val_dataloader = DataLoader(
                    val_ds,
                    batch_size=cfg.eval_batch_size,
                    shuffle=False,
                    num_workers=cfg.dataloader_num_workers,
                )
                global_step = cfg.max_train_steps or (cfg.num_epochs * len(source_dataset))
                self.log_validation(
                    source_model, target_model, scheduler,
                    val_dataloader, accelerator, global_step,
                    latent_target_encoder=latent_target_encoder if cfg.use_latent_target else None,
                )
            else:
                logger.warning("Val split unavailable for %s – skipping validation", cfg.task_name)

        # --- Save combined DDIBPipeline checkpoint ---
        if accelerator.is_main_process:
            from .pipelines import DDIBPipeline, DDIBLatentPipeline
            combined_dir = os.path.join(cfg.output_dir, "pipeline")
            if cfg.use_latent_target and latent_target_encoder is not None:
                pipeline = DDIBLatentPipeline(
                    source_unet=source_model, target_unet=target_model, scheduler=scheduler,
                    vae=latent_target_encoder.vae,
                )
            else:
                pipeline = DDIBPipeline(
                    source_unet=source_model, target_unet=target_model, scheduler=scheduler,
                )
            pipeline.save_pretrained(combined_dir)
            logger.info("Saved combined DDIBPipeline to %s", combined_dir)

        logger.info(f"[{cfg.task_name}] DDIB training complete (both domains)!")
