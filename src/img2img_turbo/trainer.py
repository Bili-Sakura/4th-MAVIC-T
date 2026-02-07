"""Core Pix2Pix-Turbo trainer for MAVIC-T tasks.

.. note::
   **Lower priority**: the Img2Image-Turbo / Pix2Pix-Turbo method has been
   found less suitable for the MAVIC-T task compared to other baselines
   (CUT, DDBM).  Its code is retained for reference and future
   experimentation, but further implementation effort should focus on the
   other baselines first.

This module adapts the training logic from
``vendor/Img2Image-Turbo/src/train_pix2pix_turbo.py`` into a reusable
:class:`Pix2PixTurboTrainer` class driven by a
:class:`~src.img2img_turbo.config.TaskConfig`.

The trainer uses the Accelerate framework for distributed / mixed-precision
training and follows the same structure as
:class:`~src.ddbm_baseline.trainer.DDBMTrainer`.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from datetime import timedelta

from .config import TaskConfig
from .dataset_wrapper import MavicTTurboDataset
from .models import Pix2PixTurbo

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.metrics import MavicCriterion  # noqa: E402
from src.training_utils import (  # noqa: E402
    create_optimizer,
    save_checkpoint_diffusers,
    push_checkpoint_to_hub,
)

logger = get_logger(__name__, log_level="INFO")


class Pix2PixTurboTrainer:
    """End-to-end Pix2Pix-Turbo trainer driven by a :class:`TaskConfig`.

    Typical usage inside a per-task script::

        from src.img2img_turbo.config import sar2eo_config
        from src.img2img_turbo.trainer import Pix2PixTurboTrainer

        cfg = sar2eo_config()
        trainer = Pix2PixTurboTrainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg

    # ----- dataset -----------------------------------------------------------

    def build_datasets(self):
        """Return ``(train_dataset, val_dataset)``.
        
        Currently val_dataset is set to None as we only use the train set.
        Can be enabled later by splitting a validation set from the training data.
        """
        train_ds = MavicTTurboDataset(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            model_channels=self.cfg.model_channels,
            use_augmented=self.cfg.use_augmented,
            use_horizontal_flip=self.cfg.use_horizontal_flip,
            use_vertical_flip=self.cfg.use_vertical_flip,
            exclude_file=self.cfg.exclude_file,
        )
        # val_ds = MavicTTurboDataset(
        #     task=self.cfg.task_name,
        #     split="val",
        #     resolution=self.cfg.resolution,
        #     model_channels=self.cfg.model_channels,
        #     with_target=False,
        # )
        val_ds = None  # Disabled: we only work with train set for now
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
                              rep_alignment_module=None, lambda_rep_alignment=1.0):
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

        # Optional latent-space L2 loss
        if latent_target_encoder is not None:
            latent_pred = latent_target_encoder.encode_with_grad(x_tgt_pred)
            with torch.no_grad():
                latent_tgt = latent_target_encoder.encode(x_tgt).detach()
            loss_latent = F.mse_loss(latent_pred.float(), latent_tgt.float())
            loss = loss + lambda_latent * loss_latent

        # Optional representation alignment loss (REPA)
        if rep_alignment_module is not None:
            with torch.no_grad():
                enc_feats = rep_alignment_module.extract_features(x_src_norm)
            rep_loss = rep_alignment_module.compute_alignment_loss(x_tgt_pred, enc_feats)
            loss = loss + lambda_rep_alignment * rep_loss

        return loss

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
            os.makedirs(os.path.join(cfg.output_dir, "checkpoints"), exist_ok=True)

        # Build components
        logger.info(f"[{cfg.task_name}] Creating Pix2Pix-Turbo model "
                     f"(lora_unet={cfg.lora_rank_unet}, lora_vae={cfg.lora_rank_vae})")
        model = self.build_model()
        model.set_train()

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
            from src.latent_target import LatentTargetEncoder
            latent_target_encoder = LatentTargetEncoder(cfg.latent_vae_path)
            logger.info(f"[{cfg.task_name}] Using latent target encoder "
                        f"from {cfg.latent_vae_path} (lambda={cfg.lambda_latent})")

        # Representation alignment (REPA)
        rep_alignment_module = None
        if cfg.use_rep_alignment and cfg.rep_alignment_model_path:
            from src.rep_alignment import SARCLIPAlignment, DINOv3SatAlignment
            if cfg.task_name == "rgb2ir":
                rep_alignment_module = DINOv3SatAlignment(cfg.rep_alignment_model_path)
            else:
                rep_alignment_module = SARCLIPAlignment(cfg.rep_alignment_model_path)
            rep_alignment_module.build_projector(cfg.model_channels)
            logger.info(
                f"[{cfg.task_name}] Representation alignment enabled "
                f"(model={cfg.rep_alignment_model_path}, "
                f"lambda={cfg.lambda_rep_alignment})"
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
        # val_dataloader is disabled - we only use train set for now
        # val_dataloader = DataLoader(
        #     val_dataset,
        #     batch_size=cfg.eval_batch_size,
        #     shuffle=False,
        #     num_workers=cfg.dataloader_num_workers,
        # ) if val_dataset is not None else None

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
            tracker_config = {k: str(v) for k, v in vars(cfg).items()}
            accelerator.init_trackers(f"turbo-{cfg.task_name}", config=tracker_config)

        logger.info("***** Running training *****")
        logger.info(f"  Task             = {cfg.task_name}")
        logger.info(f"  Num examples     = {len(train_dataset)}")
        logger.info(f"  Num epochs       = {cfg.num_epochs}")
        logger.info(f"  Batch size/dev   = {cfg.train_batch_size}")
        logger.info(f"  Total opt steps  = {cfg.max_train_steps}")

        global_step = 0
        first_epoch = 0

        # Resume
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

        progress_bar = tqdm(
            range(global_step, cfg.max_train_steps),
            disable=not accelerator.is_local_main_process,
            desc=f"Training {cfg.task_name}",
        )

        for epoch in range(first_epoch, cfg.num_epochs):
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    loss = self.compute_training_loss(
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
                        lambda_rep_alignment=cfg.lambda_rep_alignment,
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
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if (
                        checkpointing_steps is not None
                        and global_step % checkpointing_steps == 0
                        and accelerator.is_main_process
                    ):
                        outf = os.path.join(cfg.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(model).save_model(outf)
                        logger.info(f"Saved checkpoint to {outf}")

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
                logger.info(f"Saved diffusers-style checkpoint at epoch {epoch + 1}")

                if cfg.push_to_hub and cfg.hub_model_id:
                    push_checkpoint_to_hub(
                        epoch_dir,
                        hub_model_id=cfg.hub_model_id,
                        commit_message=f"img2img_turbo {cfg.task_name} epoch {epoch + 1}",
                        path_in_repo=f"img2img_turbo/{cfg.task_name}/checkpoint-epoch-{epoch + 1}",
                    )

        # Save final model
        if accelerator.is_main_process:
            outf = os.path.join(cfg.output_dir, "checkpoints", "model_final.pkl")
            accelerator.unwrap_model(model).save_model(outf)
            logger.info(f"Saved final model to {outf}")

        accelerator.end_training()
        logger.info(f"[{cfg.task_name}] Training complete!")
