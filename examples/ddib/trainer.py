"""Core DDIB trainer for MAVIC-T tasks.

DDIB (Dual Diffusion Implicit Bridges, ICLR 2023) translates images between
two domains by training *independent unconditional* diffusion models on each
domain and connecting them through a shared DDIM latent space.

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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from datetime import timedelta

from src.schedulers import DDIBScheduler
from .config import TaskConfig
from .dataset_wrapper import MavicTDDIBDataset
from src.models.unet_ddib import create_model

from src.utils.training_utils import (  # noqa: E402
    create_optimizer,
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
        common = dict(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            use_augmented=self.cfg.use_augmented,
            use_horizontal_flip=self.cfg.use_horizontal_flip,
            use_vertical_flip=self.cfg.use_vertical_flip,
            exclude_file=self.cfg.exclude_file,
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

    def build_model(self, in_channels: int):
        """Create an unconditional DDIB UNet for a single domain."""
        in_ch = self.cfg.latent_channels if self.cfg.use_latent_target else in_channels
        return create_model(
            image_size=self.cfg.resolution,
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

    # ----- single-domain training loop ---------------------------------------

    def _train_single_domain(
        self,
        domain_label: str,
        model: torch.nn.Module,
        scheduler: DDIBScheduler,
        dataset: MavicTDDIBDataset,
        accelerator: Accelerator,
        rep_alignment_module=None,
        lambda_rep_alignment: float = 1.0,
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
            tracker_config = {k: str(v) for k, v in vars(cfg).items()}
            accelerator.init_trackers(f"ddib-{domain_label}-{cfg.task_name}", config=tracker_config)

        logger.info(f"***** Training DDIB {domain_label} model *****")
        logger.info(f"  Task             = {cfg.task_name}")
        logger.info(f"  Domain           = {domain_label}")
        logger.info(f"  Num examples     = {len(dataset)}")
        logger.info(f"  Num epochs       = {num_epochs}")
        logger.info(f"  Batch size/dev   = {cfg.train_batch_size}")
        logger.info(f"  Total opt steps  = {local_max_train_steps}")

        global_step = 0
        first_epoch = 0

        # Resume
        if cfg.resume_from_checkpoint:
            path = cfg.resume_from_checkpoint
            if path == "latest":
                dirs = sorted(
                    [d for d in os.listdir(domain_output_dir) if d.startswith("checkpoint")],
                    key=lambda x: int(x.split("-")[1]),
                )
                path = dirs[-1] if dirs else None
            if path is not None:
                full_path = os.path.join(domain_output_dir, path)
                if os.path.isdir(full_path):
                    accelerator.load_state(full_path)
                    global_step = int(Path(path).name.split("-")[1])
                    first_epoch = global_step // num_update_steps_per_epoch
                    logger.info(f"Resumed {domain_label} from {path}")

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

                    if rep_alignment_module is not None:
                        loss, pred_xstart = scheduler.compute_training_loss(
                            model, x_0, return_pred_xstart=True,
                        )
                        with torch.no_grad():
                            enc_feats = rep_alignment_module.extract_features(x_0)
                        rep_loss = rep_alignment_module.compute_alignment_loss(
                            pred_xstart, enc_feats,
                        )
                        loss = loss + lambda_rep_alignment * rep_loss
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
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if (
                        checkpointing_steps is not None
                        and global_step % checkpointing_steps == 0
                        and accelerator.is_main_process
                    ):
                        save_path = os.path.join(domain_output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        save_training_config(cfg, save_path)
                        logger.info(f"Saved {domain_label} state to {save_path}")

                        if cfg.checkpoints_total_limit is not None:
                            ckpts = sorted(
                                [d for d in os.listdir(domain_output_dir) if d.startswith("checkpoint")],
                                key=lambda x: int(x.split("-")[1]),
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
                    )

        accelerator.end_training()
        logger.info(f"[{cfg.task_name}] DDIB {domain_label} training complete!")

    # ----- main training entry point -----------------------------------------

    def train(self):
        """Run the full DDIB training loop (both source and target domain models)."""
        cfg = self.cfg

        # Accelerator setup (shared)
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

        # Build datasets
        logger.info(f"[{cfg.task_name}] Loading datasets …")
        source_dataset, target_dataset = self.build_datasets()

        # Build scheduler (shared between both models)
        scheduler = self.build_scheduler()

        # Representation alignment (REPA)
        rep_alignment_module = None
        if cfg.use_rep_alignment and cfg.rep_alignment_model_path:
            from src.utils.rep_alignment import MaRSRGBAlignment, MaRSSARAlignment
            if cfg.task_name == "rgb2ir":
                # Default for RGB2IR: MaRS-RGB alignment
                rep_alignment_module = MaRSRGBAlignment(cfg.rep_alignment_model_path)
            else:
                # Default for SAR2EO, SAR2IR, SAR2RGB: MaRS-SAR alignment
                rep_alignment_module = MaRSSARAlignment(cfg.rep_alignment_model_path)
            rep_alignment_module.build_projector(cfg.model_channels)
            logger.info(
                f"[{cfg.task_name}] Representation alignment enabled "
                f"(model={cfg.rep_alignment_model_path}, "
                f"lambda={cfg.lambda_rep_alignment})"
            )

        # --- Phase 1: Train source-domain model ---
        logger.info(f"[{cfg.task_name}] === Phase 1: Training source-domain model ===")
        source_model = self.build_model(in_channels=cfg.source_channels)
        self._train_single_domain(
            "source", source_model, scheduler, source_dataset, accelerator,
            rep_alignment_module=rep_alignment_module,
            lambda_rep_alignment=cfg.lambda_rep_alignment,
        )


        # --- Phase 2: Train target-domain model ---
        logger.info(f"[{cfg.task_name}] === Phase 2: Training target-domain model ===")
        target_model = self.build_model(in_channels=cfg.target_channels)
        self._train_single_domain(
            "target", target_model, scheduler, target_dataset, accelerator,
            rep_alignment_module=rep_alignment_module,
            lambda_rep_alignment=cfg.lambda_rep_alignment,
        )

        # --- Save combined DDIBPipeline checkpoint ---
        if accelerator.is_main_process:
            from .pipelines import DDIBPipeline
            combined_dir = os.path.join(cfg.output_dir, "pipeline")
            pipeline = DDIBPipeline(
                source_unet=source_model, target_unet=target_model, scheduler=scheduler,
            )
            pipeline.save_pretrained(combined_dir)
            logger.info("Saved combined DDIBPipeline to %s", combined_dir)

        logger.info(f"[{cfg.task_name}] DDIB training complete (both domains)!")
