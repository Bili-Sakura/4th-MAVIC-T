"""Core CUT trainer for MAVIC-T tasks.

This module implements the CUT (Contrastive Unpaired Translation) training
loop as a reusable :class:`CUTTrainer` class, following the same structure
as :class:`src.ddbm_baseline.trainer.DDBMTrainer`.  Per-task scripts
instantiate the trainer with their own
:class:`~src.cut_baseline.config.TaskConfig` and can sub-class any method for
task-specific modifications.

The training loop uses *Accelerate* for mixed-precision, multi-GPU, and
gradient-accumulation support, matching the diffusers-style conventions
established in the DDBM baseline.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from datetime import timedelta

from .config import TaskConfig
from .dataset_wrapper import MavicTCUTDataset
from .model import (
    create_generator,
    create_discriminator,
    create_patch_sample_mlp,
    GANLoss,
    PatchNCELoss,
)

from src.metrics import MavicCriterion  # noqa: E402
from src.training_utils import (  # noqa: E402
    create_optimizer,
    save_checkpoint_diffusers,
    push_checkpoint_to_hub,
)

logger = get_logger(__name__, log_level="INFO")


class CUTTrainer:
    """End-to-end CUT trainer driven by a :class:`TaskConfig`.

    Typical usage inside a per-task script::

        from src.cut_baseline.config import sar2eo_config
        from src.cut_baseline.trainer import CUTTrainer

        cfg = sar2eo_config()
        trainer = CUTTrainer(cfg)
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
        train_ds = MavicTCUTDataset(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            model_channels=self.cfg.model_channels,
            use_augmented=self.cfg.use_augmented,
            use_horizontal_flip=self.cfg.use_horizontal_flip,
            use_vertical_flip=self.cfg.use_vertical_flip,
            exclude_file=self.cfg.exclude_file,
        )
        # val_ds = MavicTCUTDataset(
        #     task=self.cfg.task_name,
        #     split="val",
        #     resolution=self.cfg.resolution,
        #     model_channels=self.cfg.model_channels,
        #     with_target=False,
        # )
        val_ds = None  # Disabled: we only work with train set for now
        return train_ds, val_ds

    # ----- model / losses ----------------------------------------------------

    def build_generator(self):
        """Create the CUT generator."""
        return create_generator(
            input_nc=self.cfg.model_channels,
            output_nc=self.cfg.model_channels,
            ngf=self.cfg.ngf,
            netG=self.cfg.netG,
            norm_type=self.cfg.normG,
            use_dropout=not self.cfg.no_dropout,
            no_antialias=self.cfg.no_antialias,
            no_antialias_up=self.cfg.no_antialias_up,
            init_type=self.cfg.init_type,
            init_gain=self.cfg.init_gain,
        )

    def build_discriminator(self):
        """Create the CUT PatchGAN discriminator."""
        return create_discriminator(
            input_nc=self.cfg.model_channels,
            ndf=self.cfg.ndf,
            netD=self.cfg.netD,
            n_layers_D=self.cfg.n_layers_D,
            norm_type=self.cfg.normD,
            no_antialias=self.cfg.no_antialias,
            init_type=self.cfg.init_type,
            init_gain=self.cfg.init_gain,
        )

    def build_patch_sample_mlp(self):
        """Create the PatchSampleMLP feature network."""
        return create_patch_sample_mlp(
            use_mlp=(self.cfg.netF == "mlp_sample"),
            nc=self.cfg.netF_nc,
            init_type=self.cfg.init_type,
            init_gain=self.cfg.init_gain,
        )

    # ----- loss helpers ------------------------------------------------------

    @staticmethod
    def preprocess_batch(batch, device):
        """Scale a ``(source, target)`` batch from [0,1] to [-1,1]."""
        source = batch[0].to(device) * 2 - 1
        target = batch[1].to(device) * 2 - 1
        return source, target

    @staticmethod
    def compute_D_loss(netD, criterion_GAN, real_B, fake_B):
        """Compute discriminator loss."""
        pred_fake = netD(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, False).mean()
        pred_real = netD(real_B)
        loss_D_real = criterion_GAN(pred_real, True).mean()
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D

    @staticmethod
    def compute_G_loss(
        netG, netD, netF, criterion_GAN, nce_criteria,
        real_A, fake_B, real_B,
        nce_layers, lambda_GAN, lambda_NCE,
        nce_idt, num_patches,
        mavic_criterion=None, mavic_loss_weight=0.1,
        latent_target_encoder=None, lambda_latent=1.0,
    ):
        """Compute generator loss (GAN + NCE + optional identity NCE + optional MAVIC + optional latent).

        Returns
        -------
        loss_G : Tensor
            Total generator loss.
        loss_G_GAN : Tensor or float
            GAN component.
        loss_NCE : Tensor or float
            NCE component.
        loss_NCE_Y : Tensor or float
            Identity NCE component (0.0 if disabled).
        """
        # GAN loss
        if lambda_GAN > 0.0:
            pred_fake = netD(fake_B)
            loss_G_GAN = criterion_GAN(pred_fake, True).mean() * lambda_GAN
        else:
            loss_G_GAN = torch.tensor(0.0, device=real_A.device)

        # NCE loss
        if lambda_NCE > 0.0:
            loss_NCE = CUTTrainer._calculate_NCE_loss(
                netG, netF, nce_criteria, real_A, fake_B, nce_layers, lambda_NCE, num_patches,
            )
        else:
            loss_NCE = torch.tensor(0.0, device=real_A.device)

        # Identity NCE loss
        loss_NCE_Y = torch.tensor(0.0, device=real_A.device)
        if nce_idt and lambda_NCE > 0.0:
            # Pass real_B through generator to get identity output
            idt_B = netG(real_B)
            loss_NCE_Y = CUTTrainer._calculate_NCE_loss(
                netG, netF, nce_criteria, real_B, idt_B, nce_layers, lambda_NCE, num_patches,
            )
            loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = loss_NCE

        loss_G = loss_G_GAN + loss_NCE_both

        # Optional MAVIC metric-based loss
        if mavic_criterion is not None:
            pred_01 = (fake_B + 1) * 0.5
            target_01 = (real_B + 1) * 0.5
            pred_01 = pred_01.clamp(0, 1)
            target_01 = target_01.clamp(0, 1)
            mavic_loss = mavic_criterion(pred_01, target_01)
            loss_G = loss_G + mavic_loss_weight * mavic_loss

        # Optional latent-space L2 loss
        if latent_target_encoder is not None:
            latent_pred = latent_target_encoder.vae.encode(fake_B).latent_dist.mean
            latent_pred = latent_pred * latent_target_encoder.scaling_factor
            with torch.no_grad():
                latent_tgt = latent_target_encoder.encode(real_B).detach()
            loss_latent = F.mse_loss(latent_pred.float(), latent_tgt.float())
            loss_G = loss_G + lambda_latent * loss_latent

        return loss_G, loss_G_GAN, loss_NCE, loss_NCE_Y

    @staticmethod
    def _calculate_NCE_loss(netG, netF, nce_criteria, src, tgt, nce_layers, lambda_NCE, num_patches):
        """Compute contrastive loss across multiple encoder layers."""
        n_layers = len(nce_layers)
        feat_q = netG(tgt, nce_layers, encode_only=True)
        feat_k = netG(src, nce_layers, encode_only=True)
        feat_k_pool, sample_ids = netF(feat_k, num_patches, None)
        feat_q_pool, _ = netF(feat_q, num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, nce_criteria):
            loss = crit(f_q, f_k) * lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    # ----- linear lr decay scheduler ----------------------------------------

    @staticmethod
    def _get_scheduler(optimizer, n_epochs, n_epochs_decay, last_epoch=-1):
        """Linear LR decay scheduler: constant for n_epochs, then linear to 0."""

        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)

    # ----- main training loop ------------------------------------------------

    def train(self):
        """Run the full CUT training loop."""
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
            np.random.seed(cfg.seed)

        if accelerator.is_main_process:
            os.makedirs(cfg.output_dir, exist_ok=True)

        # Build components
        logger.info(f"[{cfg.task_name}] Creating CUT models (channels={cfg.model_channels}, res={cfg.resolution})")
        netG = self.build_generator()
        netD = self.build_discriminator()
        netF = self.build_patch_sample_mlp()

        nce_layers = [int(i) for i in cfg.nce_layers.split(",")]

        criterion_GAN = GANLoss(cfg.gan_mode)
        nce_criteria = [
            PatchNCELoss(nce_T=cfg.nce_T, batch_size=cfg.train_batch_size,
                         nce_includes_all_negatives_from_minibatch=cfg.nce_includes_all_negatives_from_minibatch)
            for _ in nce_layers
        ]

        mavic_criterion = None
        if cfg.use_mavic_loss:
            mavic_criterion = MavicCriterion(
                lpips_weight=cfg.mavic_lpips_weight,
                l1_weight=cfg.mavic_l1_weight,
            )
            logger.info(f"[{cfg.task_name}] Using MAVIC metric loss "
                        f"(lpips_w={cfg.mavic_lpips_weight}, l1_w={cfg.mavic_l1_weight}, "
                        f"loss_w={cfg.mavic_loss_weight})")

        # Latent target encoder (ablation)
        latent_target_encoder = None
        if cfg.use_latent_target and cfg.latent_vae_path:
            from src.latent_target import LatentTargetEncoder
            latent_target_encoder = LatentTargetEncoder(cfg.latent_vae_path)
            logger.info(f"[{cfg.task_name}] Using latent target encoder "
                        f"from {cfg.latent_vae_path} (lambda={cfg.lambda_latent})")

        # Representation alignment (placeholder – will log but not activate
        # until concrete implementations are provided)
        if cfg.use_rep_alignment and cfg.rep_alignment_model_path:
            logger.info(
                f"[{cfg.task_name}] Representation alignment configured "
                f"(model={cfg.rep_alignment_model_path}, "
                f"lambda={cfg.lambda_rep_alignment}) – placeholder, not yet active"
            )

        # Optimisers (G and D share the same lr/beta but are separate)
        optimizer_G = create_optimizer(
            netG.parameters(),
            optimizer_type=cfg.optimizer_type,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
        )
        optimizer_D = create_optimizer(
            netD.parameters(),
            optimizer_type=cfg.optimizer_type,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
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

        # Total epochs
        total_epochs = cfg.n_epochs + cfg.n_epochs_decay
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        if cfg.max_train_steps is None:
            cfg.max_train_steps = total_epochs * num_update_steps_per_epoch

        # LR schedulers
        if cfg.lr_policy == "linear":
            scheduler_G = self._get_scheduler(optimizer_G, cfg.n_epochs, cfg.n_epochs_decay)
            scheduler_D = self._get_scheduler(optimizer_D, cfg.n_epochs, cfg.n_epochs_decay)
        else:
            from diffusers.optimization import get_scheduler as get_lr_scheduler
            scheduler_G = get_lr_scheduler(
                "constant", optimizer=optimizer_G,
                num_warmup_steps=0,
                num_training_steps=cfg.max_train_steps,
            )
            scheduler_D = get_lr_scheduler(
                "constant", optimizer=optimizer_D,
                num_warmup_steps=0,
                num_training_steps=cfg.max_train_steps,
            )

        # Prepare with accelerator
        netG, netD, optimizer_G, optimizer_D, train_dataloader = accelerator.prepare(
            netG, netD, optimizer_G, optimizer_D, train_dataloader,
        )
        netF = netF.to(accelerator.device)
        criterion_GAN = criterion_GAN.to(accelerator.device)
        for crit in nce_criteria:
            crit.to(accelerator.device)
        if mavic_criterion is not None:
            mavic_criterion = mavic_criterion.to(accelerator.device)
        if latent_target_encoder is not None:
            latent_target_encoder = latent_target_encoder.to(accelerator.device)

        if accelerator.is_main_process:
            tracker_config = {k: str(v) for k, v in vars(cfg).items()}
            accelerator.init_trackers(f"cut-{cfg.task_name}", config=tracker_config)

        logger.info("***** Running CUT training *****")
        logger.info(f"  Task             = {cfg.task_name}")
        logger.info(f"  Num examples     = {len(train_dataset)}")
        logger.info(f"  Num epochs       = {total_epochs}")
        logger.info(f"  Batch size/dev   = {cfg.train_batch_size}")
        logger.info(f"  Total opt steps  = {cfg.max_train_steps}")

        global_step = 0
        first_epoch = 0
        # optimizer_F is created after data-dependent initialisation of netF
        # (see PatchSampleMLP.create_mlp which is called on first forward pass)
        optimizer_F = None

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
            desc=f"Training CUT {cfg.task_name}",
        )

        for epoch in range(first_epoch, total_epochs):
            netG.train()
            netD.train()

            for step, batch in enumerate(train_dataloader):
                real_A, real_B = self.preprocess_batch(batch, accelerator.device)

                # Data-dependent initialisation of netF (first step only).
                # PatchSampleMLP lazily creates its MLP layers on the first
                # forward pass based on the encoder feature dimensions;
                # optimizer_F is created once those parameters exist.
                if not netF.mlp_init:
                    with torch.no_grad():
                        fake_B_init = netG(real_A)
                        feat_init = netG(fake_B_init, nce_layers, encode_only=True)
                        netF(feat_init, cfg.num_patches, None)
                    optimizer_F = create_optimizer(
                        netF.parameters(),
                        optimizer_type=cfg.optimizer_type,
                        lr=cfg.learning_rate,
                        betas=(cfg.beta1, cfg.beta2),
                    )

                with accelerator.accumulate(netG, netD):
                    # Forward G
                    fake_B = netG(real_A)

                    # ---- Update D ----
                    loss_D = self.compute_D_loss(netD, criterion_GAN, real_B, fake_B)
                    accelerator.backward(loss_D)
                    optimizer_D.step()
                    optimizer_D.zero_grad()

                    # ---- Update G + F ----
                    loss_G, loss_G_GAN, loss_NCE, loss_NCE_Y = self.compute_G_loss(
                        netG, netD, netF, criterion_GAN, nce_criteria,
                        real_A, fake_B, real_B,
                        nce_layers=nce_layers,
                        lambda_GAN=cfg.lambda_GAN,
                        lambda_NCE=cfg.lambda_NCE,
                        nce_idt=cfg.nce_idt,
                        num_patches=cfg.num_patches,
                        mavic_criterion=mavic_criterion,
                        mavic_loss_weight=cfg.mavic_loss_weight,
                        latent_target_encoder=latent_target_encoder,
                        lambda_latent=cfg.lambda_latent,
                    )
                    accelerator.backward(loss_G)
                    optimizer_G.step()
                    optimizer_G.zero_grad()
                    if optimizer_F is not None:
                        optimizer_F.step()
                        optimizer_F.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    logs = {
                        "loss_D": loss_D.detach().item(),
                        "loss_G": loss_G.detach().item(),
                        "loss_G_GAN": loss_G_GAN.detach().item() if torch.is_tensor(loss_G_GAN) else loss_G_GAN,
                        "loss_NCE": loss_NCE.detach().item() if torch.is_tensor(loss_NCE) else loss_NCE,
                        "lr": optimizer_G.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if (
                        checkpointing_steps is not None
                        and global_step % checkpointing_steps == 0
                        and accelerator.is_main_process
                    ):
                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
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

            # Step LR schedulers at epoch boundary
            if cfg.lr_policy == "linear":
                scheduler_G.step()
                scheduler_D.step()

            # Save at epoch boundary
            if (
                accelerator.is_main_process
                and save_model_epochs is not None
                and (epoch + 1) % save_model_epochs == 0
            ):
                unwrapped_G = accelerator.unwrap_model(netG)
                unwrapped_D = accelerator.unwrap_model(netD)
                epoch_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch + 1}")
                save_checkpoint_diffusers(
                    epoch_dir,
                    unwrapped_G,
                    scheduler=None,
                    model_name="unet",
                    extra_state_dicts={
                        "discriminator": unwrapped_D.state_dict(),
                        "feature_network": netF.state_dict(),
                    },
                )
                logger.info(f"Saved models at epoch {epoch + 1}")

                if cfg.push_to_hub and cfg.hub_model_id:
                    push_checkpoint_to_hub(
                        epoch_dir,
                        hub_model_id=cfg.hub_model_id,
                        commit_message=f"epoch {epoch + 1}",
                    )

        accelerator.end_training()
        logger.info(f"[{cfg.task_name}] CUT training complete!")
