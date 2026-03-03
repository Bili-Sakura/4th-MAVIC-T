# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Core StegoGAN trainer for MAVIC-T tasks.

Reference: Wu, Sidi, et al. "StegoGAN: Leveraging Steganography for
Non-Bijective Image-to-Image Translation." CVPR 2024.

This module implements the StegoGAN training loop as a reusable
:class:`StegoGANTrainer` class, following the same structure as
:class:`examples.cut.trainer.CUTTrainer`.

The training loop uses *Accelerate* for mixed-precision, multi-GPU, and
gradient-accumulation support.
"""

from __future__ import annotations

import itertools
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
from PIL import Image
from diffusers.utils import make_image_grid

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from datetime import timedelta

from examples.ddbm.dataset_wrapper import PairedValDataset, resolve_paired_val_manifest
from .config import TaskConfig
from .dataset_wrapper import MavicTStegoGANDataset

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.models.stegogan_model import (
    create_generator_a,
    create_generator_b,
    create_discriminator,
    StegoGANLoss,
)

from src.utils.metrics import MavicCriterion, MetricCalculator
from src.utils.training_utils import (
    build_accelerate_tracker_config,
    build_accelerate_tracker_init_kwargs,
    checkpoint_dir_sort_key,
    checkpoint_has_accelerator_state,
    create_optimizer,
    normalize_accelerate_log_with,
    save_checkpoint_diffusers,
    save_training_config,
    push_checkpoint_to_hub,
)

logger = get_logger(__name__, log_level="INFO")


class StegoGANTrainer:
    """End-to-end StegoGAN trainer driven by a :class:`TaskConfig`.

    Typical usage inside a per-task script::

        from .config import sar2eo_config
        from .trainer import StegoGANTrainer

        cfg = sar2eo_config()
        trainer = StegoGANTrainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg

    # ----- dataset -----------------------------------------------------------

    def build_datasets(self):
        """Return ``(train_dataset, val_dataset)``."""
        resolved_paired = resolve_paired_val_manifest(
            getattr(self.cfg, "paired_val_manifest", None)
        )
        self._resolved_paired_val_manifest = resolved_paired
        paired_val_manifest_str = str(resolved_paired) if resolved_paired else getattr(
            self.cfg, "paired_val_manifest", None
        )
        train_ds = MavicTStegoGANDataset(
            task=self.cfg.task_name,
            split="train",
            resolution=self.cfg.resolution,
            load_size=self.cfg.load_size,
            source_channels=self.cfg.source_channels,
            target_channels=self.cfg.target_channels,
            model_channels=self.cfg.model_channels,
            use_augmented=self.cfg.use_augmented,
            use_random_crop=getattr(self.cfg, "use_random_crop", True),
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
            val_res = self.cfg.validation_resolution if self.cfg.validation_resolution is not None else self.cfg.resolution
            if resolved_paired is not None:
                val_ds = PairedValDataset(
                    manifest_path=resolved_paired,
                    resolution=val_res,
                    source_channels=self.cfg.source_channels,
                    target_channels=self.cfg.target_channels,
                    return_order="source_target",
                )
                logger.info(
                    "Using paired val set for validation: %s (%d pairs)",
                    resolved_paired, len(val_ds),
                )
            if val_ds is None:
                try:
                    val_ds = MavicTStegoGANDataset(
                        task=self.cfg.task_name,
                        split="test",
                        resolution=val_res,
                        load_size=self.cfg.load_size,
                        source_channels=self.cfg.source_channels,
                        target_channels=self.cfg.target_channels,
                        model_channels=self.cfg.model_channels,
                        with_target=False,
                        use_sar_despeckle=getattr(self.cfg, "use_sar_despeckle", False),
                        sar_despeckle_kernel_size=getattr(self.cfg, "sar_despeckle_kernel_size", 5),
                        sar_despeckle_strength=getattr(self.cfg, "sar_despeckle_strength", 0.6),
                    )
                    logger.info("Validation using test split.")
                except (ValueError, FileNotFoundError, RuntimeError):
                    logger.warning("Test split unavailable for %s – skipping validation", self.cfg.task_name)
        return train_ds, val_ds

    # ----- model / losses ----------------------------------------------------

    def build_generator_a(self):
        """Create Generator A (source→target)."""
        return create_generator_a(
            input_nc=self.cfg.source_channels,
            output_nc=self.cfg.target_channels,
            ngf=self.cfg.ngf,
            n_blocks=self.cfg.n_blocks,
            norm_type=self.cfg.normG,
            use_dropout=not self.cfg.no_dropout,
            resnet_layer=self.cfg.resnet_layer,
            use_fusion_block=self.cfg.use_fusion_block,
            init_type=self.cfg.init_type,
            init_gain=self.cfg.init_gain,
        )

    def build_generator_b(self):
        """Create Generator B (target→source, with mask)."""
        return create_generator_b(
            input_nc=self.cfg.target_channels,
            output_nc=self.cfg.source_channels,
            ngf=self.cfg.ngf,
            n_blocks=self.cfg.n_blocks,
            norm_type=self.cfg.normG,
            use_dropout=not self.cfg.no_dropout,
            mask_group=self.cfg.mask_group,
            resnet_layer=self.cfg.resnet_layer,
            init_type=self.cfg.init_type,
            init_gain=self.cfg.init_gain,
        )

    def build_discriminator_a(self):
        """Create discriminator D_A for the target domain."""
        return create_discriminator(
            input_nc=self.cfg.target_channels,
            ndf=self.cfg.ndf,
            n_layers=self.cfg.n_layers_D,
            norm_type=self.cfg.normD,
            init_type=self.cfg.init_type,
            init_gain=self.cfg.init_gain,
        )

    def build_discriminator_b(self):
        """Create discriminator D_B for the source domain."""
        return create_discriminator(
            input_nc=self.cfg.source_channels,
            ndf=self.cfg.ndf,
            n_layers=self.cfg.n_layers_D,
            norm_type=self.cfg.normD,
            init_type=self.cfg.init_type,
            init_gain=self.cfg.init_gain,
        )

    # ----- validation --------------------------------------------------------

    @torch.no_grad()
    def log_validation(self, netG_A, val_dataloader, accelerator, global_step):
        """Generate and save test samples."""
        from src.pipelines.stegogan import StegoGANPipeline

        logger.info("Running validation at step %d …", global_step)
        cfg = self.cfg
        was_training = netG_A.training
        unwrapped = accelerator.unwrap_model(netG_A)
        unwrapped.eval()

        pipeline = StegoGANPipeline(generator=unwrapped)
        pipeline = pipeline.to(accelerator.device)

        sample_dir = Path(cfg.output_dir) / "test_results" / f"step-{global_step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        first_grid = None
        has_paired_target = isinstance(val_dataloader.dataset, PairedValDataset)
        cols = 3 if has_paired_target else 2

        for batch_idx, batch in enumerate(val_dataloader):
            source, target = batch
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1

            with accelerator.autocast():
                result = pipeline(source_image=source_inp, output_type="pt")
            generated = (result.images + 1) * 0.5

            src_vis = source_01
            gen_vis = generated
            tgt_vis = target.to(accelerator.device) if has_paired_target else None
            if cfg.target_channels == 1 and gen_vis.shape[1] == 3:
                gen_vis = gen_vis.mean(dim=1, keepdim=True)
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

    # ----- main training loop ------------------------------------------------

    def train(self):
        """Run the full StegoGAN training loop."""
        cfg = self.cfg

        if cfg.task_name:
            cfg.output_dir = os.path.join(cfg.output_dir, "stegogan", cfg.task_name)

        checkpointing_steps = cfg.checkpointing_steps
        save_model_epochs = cfg.save_model_epochs
        if save_model_epochs is not None and save_model_epochs <= 0:
            save_model_epochs = None
        if checkpointing_steps is not None and save_model_epochs is not None:
            logging.warning(
                "checkpointing_steps is set while save_model_epochs is enabled; "
                "epoch checkpoints take priority."
            )
            checkpointing_steps = None

        # Accelerator setup
        logging_dir = os.path.join(cfg.output_dir, "logs")
        log_with = normalize_accelerate_log_with(cfg.log_with)
        project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
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
            np.random.seed(cfg.seed)

        if accelerator.is_main_process:
            os.makedirs(cfg.output_dir, exist_ok=True)

        # Build models
        logger.info(f"[{cfg.task_name}] Creating StegoGAN models "
                     f"(in={cfg.source_channels}, out={cfg.target_channels}, res={cfg.resolution})")
        netG_A = self.build_generator_a()
        netG_B = self.build_generator_b()
        netD_A = self.build_discriminator_a()
        netD_B = self.build_discriminator_b()

        criterion_GAN = StegoGANLoss(cfg.gan_mode)
        criterion_cycle = torch.nn.L1Loss()
        criterion_idt = torch.nn.L1Loss()

        # Optimisers
        optimizer_G = torch.optim.Adam(
            itertools.chain(netG_A.parameters(), netG_B.parameters()),
            lr=cfg.learning_rate_G,
            betas=(cfg.beta1, cfg.beta2),
        )
        optimizer_D = torch.optim.Adam(
            itertools.chain(netD_A.parameters(), netD_B.parameters()),
            lr=cfg.learning_rate_D,
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
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.dataloader_num_workers,
        ) if val_dataset is not None else None
        if val_dataloader is not None and accelerator.num_processes > 1:
            logger.warning("Disabling log_validation on multi-GPU to avoid deadlock.")
            val_dataloader = None

        total_epochs = cfg.n_epochs + cfg.n_epochs_decay
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        if cfg.max_train_steps is None:
            cfg.max_train_steps = total_epochs * num_update_steps_per_epoch

        # LR schedulers (linear decay)
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - cfg.n_epochs) / float(cfg.n_epochs_decay + 1)

        scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
        scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

        # Prepare with accelerator
        netG_A, netG_B, netD_A, netD_B, optimizer_G, optimizer_D, train_dataloader = accelerator.prepare(
            netG_A, netG_B, netD_A, netD_B, optimizer_G, optimizer_D, train_dataloader,
        )
        criterion_GAN = criterion_GAN.to(accelerator.device)

        # Tracker init
        tracker_config = build_accelerate_tracker_config(
            cfg, log_with, method_name="stegogan", tracker_name=f"stegogan_{cfg.task_name}",
        )
        init_kwargs = build_accelerate_tracker_init_kwargs(cfg, log_with)
        accelerator.init_trackers(f"stegogan_{cfg.task_name}", config=tracker_config, init_kwargs=init_kwargs)

        # Save config
        if accelerator.is_main_process:
            save_training_config(cfg, cfg.output_dir)

        # Training loop
        global_step = 0
        first_epoch = 0

        progress_bar = tqdm(
            range(cfg.max_train_steps),
            desc="Steps",
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(first_epoch, total_epochs):
            netG_A.train()
            netG_B.train()
            netD_A.train()
            netD_B.train()

            for step, batch in enumerate(train_dataloader):
                source = batch[0].to(accelerator.device) * 2 - 1  # [0,1] → [-1,1]
                target = batch[1].to(accelerator.device) * 2 - 1

                with accelerator.accumulate(netG_A, netG_B, netD_A, netD_B):
                    # ---- Forward pass ----
                    # B → A (with mask)
                    fake_A, latent_real_B, latent_real_B_mask = netG_B(target)
                    # A → B (clean, without latent)
                    fake_B_clean = netG_A(source)
                    # A → B (with latent from real B)
                    fake_B = netG_A(source, latent_real_B.detach())
                    # Cycle: B → A → B (with noise injection for robustness)
                    noise = 0.01 * torch.randn_like(fake_A)
                    rec_B = netG_A(fake_A + noise, latent_real_B)
                    rec_B_clean = netG_A(fake_A)
                    # Cycle: A → B → A
                    rec_A_clean, _, _ = netG_B(fake_B_clean)
                    rec_A, latent_fake_B, latent_fake_B_mask = netG_B(fake_B)

                    # Upsample masks for consistency loss
                    mask_real_B_up = F.interpolate(latent_real_B_mask, scale_factor=4, mode="bilinear", align_corners=True)
                    mask_fake_B_up = F.interpolate(latent_fake_B_mask, scale_factor=4, mode="bilinear", align_corners=True)

                    # ---- Generator losses ----
                    # GAN losses
                    loss_G_A = criterion_GAN(netD_A(fake_B), True)
                    loss_G_B = criterion_GAN(netD_B(fake_A), True)

                    # Cycle consistency losses
                    loss_cycle_A = criterion_cycle(rec_A, source) * cfg.lambda_A
                    loss_cycle_B = criterion_cycle(rec_B, target) * cfg.lambda_B

                    # Identity losses
                    loss_idt_A = torch.tensor(0.0, device=accelerator.device)
                    loss_idt_B = torch.tensor(0.0, device=accelerator.device)
                    if cfg.lambda_identity > 0 and cfg.source_channels == cfg.target_channels:
                        idt_A = netG_A(target)
                        loss_idt_A = criterion_idt(idt_A, target) * cfg.lambda_B * cfg.lambda_identity
                        idt_B, _, _ = netG_B(source)
                        loss_idt_B = criterion_idt(idt_B, source) * cfg.lambda_A * cfg.lambda_identity

                    # Mask sparsity regularisation
                    loss_reg = cfg.lambda_reg * (
                        torch.mean((latent_real_B_mask + 1e-10) ** 0.5)
                        + torch.mean((latent_fake_B_mask + 1e-10) ** 0.5)
                    )

                    # Steganographic consistency loss
                    loss_consistency_B = cfg.lambda_consistency * (
                        criterion_cycle(
                            rec_B_clean * (1 - mask_real_B_up),
                            target * (1 - mask_real_B_up),
                        )
                        + criterion_cycle(
                            fake_B_clean * (1 - mask_fake_B_up),
                            fake_B * (1 - mask_fake_B_up),
                        )
                    )

                    # Feature consistency loss
                    loss_consistency_feature = cfg.lambda_consistency * criterion_cycle(
                        latent_fake_B, latent_real_B.detach()
                    )

                    loss_G = (
                        loss_G_A + loss_G_B
                        + loss_cycle_A + loss_cycle_B
                        + loss_idt_A + loss_idt_B
                        + loss_reg + loss_consistency_B
                    )

                    accelerator.backward(loss_G)
                    if cfg.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(
                            itertools.chain(netG_A.parameters(), netG_B.parameters()),
                            cfg.max_grad_norm,
                        )
                    optimizer_G.step()
                    optimizer_G.zero_grad()

                    # ---- Discriminator losses ----
                    # D_A
                    pred_real_A = netD_A(target)
                    loss_D_A_real = criterion_GAN(pred_real_A, True)
                    pred_fake_A = netD_A(fake_B.detach())
                    loss_D_A_fake = criterion_GAN(pred_fake_A, False)
                    loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

                    # D_B
                    pred_real_B = netD_B(source)
                    loss_D_B_real = criterion_GAN(pred_real_B, True)
                    pred_fake_B = netD_B(fake_A.detach())
                    loss_D_B_fake = criterion_GAN(pred_fake_B, False)
                    loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

                    loss_D = loss_D_A + loss_D_B
                    accelerator.backward(loss_D)
                    if cfg.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(
                            itertools.chain(netD_A.parameters(), netD_B.parameters()),
                            cfg.max_grad_norm,
                        )
                    optimizer_D.step()
                    optimizer_D.zero_grad()

                global_step += 1
                progress_bar.update(1)
                logs = {
                    "loss_G": loss_G.detach().item(),
                    "loss_D": loss_D.detach().item(),
                    "loss_cycle_A": loss_cycle_A.detach().item(),
                    "loss_cycle_B": loss_cycle_B.detach().item(),
                    "loss_consistency_B": loss_consistency_B.detach().item(),
                    "loss_reg": loss_reg.detach().item(),
                    "lr_G": optimizer_G.param_groups[0]["lr"],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # Step-based validation
                if (
                    val_dataloader is not None
                    and cfg.validation_steps is not None
                    and global_step % cfg.validation_steps == 0
                    and accelerator.is_main_process
                ):
                    self.log_validation(netG_A, val_dataloader, accelerator, global_step)

                if global_step >= cfg.max_train_steps:
                    break

            # End-of-epoch LR update
            scheduler_G.step()
            scheduler_D.step()

            # Epoch-based validation
            if (
                val_dataloader is not None
                and cfg.validation_epochs is not None
                and (epoch + 1) % cfg.validation_epochs == 0
                and accelerator.is_main_process
            ):
                self.log_validation(netG_A, val_dataloader, accelerator, global_step)

            # Epoch-based checkpointing
            if save_model_epochs is not None and (epoch + 1) % save_model_epochs == 0:
                if accelerator.is_main_process:
                    ckpt_name = f"checkpoint-epoch-{epoch + 1}"
                    save_path = os.path.join(cfg.output_dir, ckpt_name)
                    save_checkpoint_diffusers(
                        save_path,
                        accelerator,
                        {"generator": netG_A, "generator_b": netG_B,
                         "discriminator_a": netD_A, "discriminator_b": netD_B},
                    )
                    logger.info("Saved checkpoint to %s", save_path)

                    # Clean up old checkpoints
                    if cfg.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in Path(cfg.output_dir).iterdir()
                             if d.is_dir() and d.name.startswith("checkpoint-")],
                            key=checkpoint_dir_sort_key,
                        )
                        while len(ckpts) > cfg.checkpoints_total_limit:
                            old = ckpts.pop(0)
                            logger.info("Removing old checkpoint: %s", old)
                            shutil.rmtree(old)

                    if cfg.push_to_hub:
                        push_checkpoint_to_hub(
                            save_path,
                            hub_model_id=cfg.hub_model_id,
                            hub_path_tier=cfg.hub_path_tier,
                        )

        accelerator.end_training()
        logger.info("Training complete.")
