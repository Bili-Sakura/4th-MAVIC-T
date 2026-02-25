"""Core UniDB trainer for MAVIC-T tasks.

UniDB uses a noise-predicting model with x0=GT, mu=LQ (condition).
Training loss: L1(reverse_sde_step_mean(xt, score, t), reverse_optimum_step(xt, x0, t))
Based on https://github.com/2769433owo/UniDB-plusplus
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
from PIL import Image
from diffusers.utils import make_image_grid

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from datetime import timedelta

from src.schedulers import UniDBScheduler
from src.pipelines.unidb import UniDBPipeline
from .config import TaskConfig
from examples.ddbm.dataset_wrapper import resolve_paired_val_manifest
from .dataset_wrapper import MavicTUniDBDataset, PairedValDataset
from .model import create_model

from src.utils.metrics import MetricCalculator
from src.utils.training_utils import (
    build_accelerate_tracker_config,
    build_accelerate_tracker_init_kwargs,
    checkpoint_dir_sort_key,
    create_optimizer,
    normalize_accelerate_log_with,
    save_checkpoint_diffusers,
    save_training_config,
)

logger = get_logger(__name__, log_level="INFO")


class UniDBTrainer:
    """End-to-end UniDB trainer driven by TaskConfig."""

    def __init__(self, cfg: TaskConfig) -> None:
        self.cfg = cfg

    @property
    def baseline_name(self) -> str:
        return "unidb"

    @property
    def pipeline_class_name(self) -> str:
        return "UniDBPipeline"

    def get_validation_pipelines(self):
        return UniDBPipeline, None  # No latent pipeline for UniDB

    def get_inference_kwargs(self, source_inp: torch.Tensor) -> dict:
        cfg = self.cfg
        return {
            "image": source_inp,
            "num_inference_steps": cfg.num_inference_steps,
            "cfg_scale": getattr(cfg, "cfg_scale", 1.0),
            "method": cfg.method,
            "solver_type": cfg.solver_type,
            "solver_step": cfg.solver_step,
            "output_type": "pt",
        }

    def build_datasets(self):
        cfg = self.cfg
        src_ch = cfg.source_channels
        tgt_ch = cfg.target_channels

        resolved_paired = resolve_paired_val_manifest(getattr(cfg, "paired_val_manifest", None))
        self._resolved_paired_val_manifest = resolved_paired
        paired_val_manifest_str = str(resolved_paired) if resolved_paired else getattr(
            cfg, "paired_val_manifest", None
        )

        train_ds = MavicTUniDBDataset(
            task=cfg.task_name,
            split="train",
            resolution=cfg.resolution,
            source_channels=src_ch,
            target_channels=tgt_ch,
            use_augmented=cfg.use_augmented,
            use_random_crop=getattr(cfg, "use_random_crop", False),
            use_horizontal_flip=cfg.use_horizontal_flip,
            use_vertical_flip=cfg.use_vertical_flip,
            exclude_file=cfg.exclude_file,
            paired_val_manifest=paired_val_manifest_str,
            sar2rgb_sup_manifest=cfg.sar2rgb_sup_manifest if getattr(cfg, "use_sar2rgb_sup", False) else None,
            use_sar_despeckle=getattr(cfg, "use_sar_despeckle", False),
            sar_despeckle_kernel_size=getattr(cfg, "sar_despeckle_kernel_size", 5),
            sar_despeckle_strength=getattr(cfg, "sar_despeckle_strength", 0.6),
        )
        val_ds = None
        if (
            (cfg.validation_epochs and cfg.validation_epochs > 0)
            or (cfg.validation_steps and cfg.validation_steps > 0)
        ):
            val_res = getattr(cfg, "output_resolution", None) or cfg.resolution
            if resolved_paired is not None:
                val_ds = PairedValDataset(
                    manifest_path=resolved_paired,
                    resolution=val_res,
                    source_channels=src_ch,
                    target_channels=tgt_ch,
                    use_sar_despeckle=getattr(cfg, "use_sar_despeckle", False),
                    sar_despeckle_kernel_size=getattr(cfg, "sar_despeckle_kernel_size", 5),
                    sar_despeckle_strength=getattr(cfg, "sar_despeckle_strength", 0.6),
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
                    val_ds = MavicTUniDBDataset(
                        task=cfg.task_name,
                        split="test",
                        resolution=val_res,
                        source_channels=src_ch,
                        target_channels=tgt_ch,
                        with_target=False,
                        use_sar_despeckle=getattr(cfg, "use_sar_despeckle", False),
                        sar_despeckle_kernel_size=getattr(cfg, "sar_despeckle_kernel_size", 5),
                        sar_despeckle_strength=getattr(cfg, "sar_despeckle_strength", 0.6),
                    )
                    logger.info("Validation using test split.")
                except (ValueError, FileNotFoundError, RuntimeError):
                    logger.warning("Test split unavailable for %s", cfg.task_name)
        return train_ds, val_ds

    def build_model(self, image_size: int | None = None):
        cfg = self.cfg
        ch = cfg.model_channels
        return create_model(
            in_channels=ch,
            out_channels=ch,
            nf=cfg.nf,
            depth=cfg.depth,
        )

    def build_scheduler(self):
        cfg = self.cfg
        return UniDBScheduler(
            lambda_square=cfg.lambda_square,
            gamma=cfg.gamma,
            num_train_timesteps=cfg.num_train_timesteps,
            schedule=cfg.schedule,
            eps=cfg.eps,
            method=cfg.method,
            solver_type=cfg.solver_type,
            solver_step=cfg.solver_step,
        )

    @staticmethod
    def preprocess_batch(batch, device):
        """Scale (target, source) from [0,1] to [-1,1]. Expand channels to match for model."""
        x0 = batch[0].to(device)
        x_T = batch[1].to(device)
        if x0.shape[1] != x_T.shape[1]:
            if x0.shape[1] == 3 and x_T.shape[1] == 1:
                x_T = x_T.repeat(1, 3, 1, 1)
            elif x0.shape[1] == 1 and x_T.shape[1] == 3:
                x0 = x0.repeat(1, 3, 1, 1)
        return x0 * 2 - 1, x_T * 2 - 1

    @staticmethod
    def apply_conditioning_dropout(
        condition: torch.Tensor,
        dropout_prob: float,
    ) -> tuple[torch.Tensor, float]:
        """Randomly replace conditioning with zeros for CFG training."""
        if dropout_prob <= 0.0:
            return condition, 0.0

        batch_size = condition.shape[0]
        drop_mask = torch.rand(batch_size, device=condition.device) < dropout_prob
        if not torch.any(drop_mask):
            return condition, 0.0

        dropped = condition.clone()
        dropped[drop_mask] = 0.0
        drop_ratio = drop_mask.float().mean().item()
        return dropped, drop_ratio

    @staticmethod
    def compute_training_loss(model, scheduler, x0, x_T):
        """UniDB training loss: L1(xt_1_expection, xt_1_optimum)."""
        scheduler._initialize(device=x0.device)
        scheduler._mu = x_T

        timesteps, noisy_states = scheduler.generate_random_states(x0, x_T)

        t_flat = timesteps.squeeze()
        noise = model(noisy_states, x_T, t_flat)
        score = scheduler.get_score_from_noise(noise, timesteps)

        xt_1_expection = scheduler.reverse_sde_step_mean(noisy_states, score, timesteps)
        xt_1_optimum = scheduler.reverse_optimum_step(noisy_states, x0, timesteps)

        loss = F.l1_loss(xt_1_expection, xt_1_optimum)
        return loss, {"loss": loss.detach()}

    @torch.no_grad()
    def log_validation(self, model, scheduler, val_dataloader, accelerator, global_step):
        """Generate and save validation samples."""
        cfg = self.cfg
        was_training = model.training
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()

        pipeline = UniDBPipeline(unet=unwrapped, scheduler=scheduler)
        pipeline = pipeline.to(accelerator.device)

        sample_dir = Path(cfg.output_dir) / "test_results" / f"step-{global_step:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        first_grid = None
        has_paired = isinstance(val_dataloader.dataset, PairedValDataset)
        cols = 3 if has_paired else 2

        for sample_idx, batch in enumerate(val_dataloader):
            target, source = batch
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1

            with accelerator.autocast():
                result = pipeline(**self.get_inference_kwargs(source_inp))
            generated = (result.images + 1) * 0.5

            src_vis = source_01
            gen_vis = generated
            tgt_vis = target.to(accelerator.device) if has_paired else None
            if gen_vis.shape[1] != src_vis.shape[1]:
                if gen_vis.shape[1] == 3 and src_vis.shape[1] == 1:
                    src_vis = src_vis.repeat(1, 3, 1, 1)
                elif gen_vis.shape[1] == 1 and src_vis.shape[1] == 3:
                    gen_vis = gen_vis.repeat(1, 3, 1, 1)
            if tgt_vis is not None and tgt_vis.shape[1] != gen_vis.shape[1]:
                if gen_vis.shape[1] == 3 and tgt_vis.shape[1] == 1:
                    tgt_vis = tgt_vis.repeat(1, 3, 1, 1)
                elif gen_vis.shape[1] == 1 and tgt_vis.shape[1] == 3:
                    tgt_vis = tgt_vis.repeat(1, 3, 1, 1)

            src_uint8 = (src_vis.clamp(0, 1) * 255).round().to(torch.uint8)
            gen_uint8 = (gen_vis.clamp(0, 1) * 255).round().to(torch.uint8)
            tgt_uint8 = (tgt_vis.clamp(0, 1) * 255).round().to(torch.uint8) if tgt_vis is not None else None

            src_uint8 = src_uint8.permute(0, 2, 3, 1).cpu().numpy()
            gen_uint8 = gen_uint8.permute(0, 2, 3, 1).cpu().numpy()
            tgt_uint8 = tgt_uint8.permute(0, 2, 3, 1).cpu().numpy() if tgt_uint8 is not None else None

            batch_images = []
            for i in range(len(src_uint8)):
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

            grid = make_image_grid(batch_images, rows=1, cols=cols)
            grid.save(sample_dir / f"sample_{sample_idx:03d}.png")
            if first_grid is None:
                first_grid = grid.copy()
            saved += 1

        logger.info("Saved %d test samples to %s", saved, sample_dir)
        if first_grid is not None:
            from src.utils.training_utils import log_validation_images_to_trackers
            log_validation_images_to_trackers(accelerator, first_grid, global_step)

        metrics_result = {}
        manifest_path = getattr(self, "_resolved_paired_val_manifest", None) or (
            Path(cfg.paired_val_manifest) if getattr(cfg, "paired_val_manifest", None) else None
        )
        if manifest_path is not None and Path(manifest_path).is_file() and accelerator.is_main_process:
            metrics_result = self._evaluate_paired_val_metrics(
                model, scheduler, pipeline, accelerator, manifest_path=manifest_path
            )

        if was_training:
            unwrapped.train()
        return {"saved_samples": saved, "sample_dir": str(sample_dir), **metrics_result}

    def _evaluate_paired_val_metrics(self, model, scheduler, pipeline, accelerator, manifest_path=None):
        cfg = self.cfg
        manifest_path = Path(manifest_path) if manifest_path is not None else Path(cfg.paired_val_manifest)
        if not manifest_path.is_file():
            return {}

        src_ch = cfg.source_channels
        tgt_ch = cfg.target_channels
        res = getattr(cfg, "output_resolution", None) or cfg.resolution
        paired_ds = PairedValDataset(
            manifest_path=manifest_path,
            resolution=res,
            source_channels=src_ch,
            target_channels=tgt_ch,
        )
        paired_loader = DataLoader(paired_ds, batch_size=1, shuffle=False, num_workers=0)

        metric_calc = MetricCalculator(device=str(accelerator.device), compute_fid=False)
        for _target, source in paired_loader:
            source_01 = source.to(accelerator.device)
            source_inp = source_01 * 2 - 1
            with accelerator.autocast():
                result = pipeline(**self.get_inference_kwargs(source_inp))
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

    def train(self):
        cfg = self.cfg
        cond_dropout_prob = float(getattr(cfg, "conditioning_dropout_prob", 0.0))
        if cond_dropout_prob < 0.0 or cond_dropout_prob > 1.0:
            raise ValueError(
                f"conditioning_dropout_prob must be in [0, 1], got {cond_dropout_prob}"
            )
        use_cond_dropout = cond_dropout_prob > 0.0
        if cfg.task_name:
            cfg.output_dir = os.path.join(cfg.output_dir, self.baseline_name, cfg.task_name)

        checkpointing_steps = cfg.checkpointing_steps
        save_model_epochs = cfg.save_model_epochs
        if save_model_epochs is not None and save_model_epochs <= 0:
            save_model_epochs = None
        if checkpointing_steps and save_model_epochs:
            checkpointing_steps = None

        log_with = normalize_accelerate_log_with(cfg.log_with)
        logging_dir = os.path.join(cfg.output_dir, "logs")
        project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
        # TODO: Multi-GPU validation deadlock – accelerator.wait_for_everyone() / barrier hangs on some
        # setups (e.g. RTX 4090) with "No device id is provided via init_process_group or barrier".
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
            log_with=log_with,
            project_config=project_config,
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=7200), backend="nccl")],
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

        model = self.build_model()
        scheduler = self.build_scheduler()

        ema_model = None
        if cfg.use_ema:
            from diffusers.training_utils import EMAModel
            ema_model = EMAModel(model.parameters(), decay=cfg.ema_decay, use_ema_warmup=True, model_cls=type(model))

        optimizer = create_optimizer(
            model.parameters(),
            optimizer_type=cfg.optimizer_type,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

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
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataloader_num_workers,
        ) if val_dataset is not None else None

        from diffusers.optimization import get_scheduler as get_lr_scheduler
        total_steps = cfg.max_train_steps or len(train_dataloader) * cfg.num_epochs
        lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
            num_training_steps=total_steps * cfg.gradient_accumulation_steps,
        )

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        if ema_model is not None:
            ema_model.to(accelerator.device)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        if cfg.max_train_steps is None:
            cfg.max_train_steps = cfg.num_epochs * num_update_steps_per_epoch
        cfg.num_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

        if accelerator.is_main_process:
            tracker_config = build_accelerate_tracker_config(cfg)
            tracker_init = build_accelerate_tracker_init_kwargs(cfg, f"{self.baseline_name}-{cfg.task_name}")
            accelerator.init_trackers(
                f"{self.baseline_name}-{cfg.task_name}",
                config=tracker_config,
                init_kwargs=tracker_init or {},
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
            if path:
                if os.path.isabs(path) or os.path.sep in path:
                    load_path = os.path.abspath(path)
                else:
                    load_path = os.path.join(cfg.output_dir, path)
                accelerator.load_state(load_path)
                global_step = int(Path(path).name.split("-")[1])
                first_epoch = global_step // num_update_steps_per_epoch

        progress_bar = tqdm(
            range(global_step, cfg.max_train_steps),
            disable=not accelerator.is_local_main_process,
            desc=f"Training {cfg.task_name}",
        )
        if use_cond_dropout:
            logger.info(
                "CFG conditioning dropout enabled: p=%.3f (null condition = zeros)",
                cond_dropout_prob,
            )

        for epoch in range(first_epoch, cfg.num_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    x0, x_T = self.preprocess_batch(batch, accelerator.device)
                    cond_drop_ratio = None
                    if use_cond_dropout:
                        x_T, cond_drop_ratio = self.apply_conditioning_dropout(
                            x_T, cond_dropout_prob
                        )
                    loss, _ = self.compute_training_loss(model, scheduler, x0, x_T)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    if ema_model is not None:
                        ema_model.step(model.parameters())
                    progress_bar.update(1)
                    global_step += 1

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "epoch": epoch}
                    if cond_drop_ratio is not None:
                        logs["cond/drop_ratio"] = cond_drop_ratio
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    # TODO: multi-GPU sync deadlock – see InitProcessGroupKwargs
                    if (
                        val_dataloader is not None
                        and cfg.validation_steps
                        and global_step % cfg.validation_steps == 0
                    ):
                        if accelerator.is_main_process:
                            val_result = self.log_validation(model, scheduler, val_dataloader, accelerator, global_step)
                            if val_result:
                                accelerator.log(val_result, step=global_step)
                        accelerator.wait_for_everyone()

                    if (
                        checkpointing_steps
                        and global_step % checkpointing_steps == 0
                        and accelerator.is_main_process
                    ):
                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        unwrapped = accelerator.unwrap_model(model)
                        extra_sd = {}
                        if ema_model is not None:
                            extra_sd["ema_unet"] = {
                                n: p.clone().detach()
                                for n, p in zip(unwrapped.state_dict().keys(), ema_model.shadow_params)
                            }
                        save_checkpoint_diffusers(
                            save_path,
                            unwrapped,
                            scheduler=scheduler,
                            model_name="unet",
                            pipeline_class_name=self.pipeline_class_name,
                            extra_state_dicts=extra_sd if extra_sd else None,
                        )
                        save_training_config(cfg, save_path)
                        logger.info("Saved checkpoint to %s", save_path)

                        if cfg.checkpoints_total_limit:
                            ckpts = sorted([d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")], key=checkpoint_dir_sort_key)
                            for old in ckpts[:-cfg.checkpoints_total_limit]:
                                shutil.rmtree(os.path.join(cfg.output_dir, old))

                if global_step >= cfg.max_train_steps:
                    break

            # TODO: multi-GPU sync deadlock – see InitProcessGroupKwargs
            if (
                val_dataloader is not None
                and cfg.validation_epochs
                and (epoch + 1) % cfg.validation_epochs == 0
            ):
                if accelerator.is_main_process:
                    val_result = self.log_validation(model, scheduler, val_dataloader, accelerator, global_step)
                    if val_result:
                        accelerator.log(val_result, step=global_step)
                accelerator.wait_for_everyone()

            if (
                accelerator.is_main_process
                and save_model_epochs
                and (epoch + 1) % save_model_epochs == 0
            ):
                unwrapped = accelerator.unwrap_model(model)
                epoch_dir = os.path.join(cfg.output_dir, f"checkpoint-epoch-{epoch + 1}")
                extra_sd = {}
                if ema_model is not None:
                    extra_sd["ema_unet"] = {
                        n: p.clone().detach()
                        for n, p in zip(unwrapped.state_dict().keys(), ema_model.shadow_params)
                    }
                save_checkpoint_diffusers(
                    epoch_dir,
                    unwrapped,
                    scheduler=scheduler,
                    model_name="unet",
                    pipeline_class_name=self.pipeline_class_name,
                    extra_state_dicts=extra_sd if extra_sd else None,
                )
                save_training_config(cfg, epoch_dir)
                logger.info("Saved model at epoch %d", epoch + 1)

                if cfg.checkpoints_total_limit:
                    ckpts = sorted(
                        [d for d in os.listdir(cfg.output_dir) if d.startswith("checkpoint")],
                        key=checkpoint_dir_sort_key,
                    )
                    for old in ckpts[:-cfg.checkpoints_total_limit]:
                        shutil.rmtree(os.path.join(cfg.output_dir, old))

        accelerator.end_training()
        logger.info("[%s] Training complete!", cfg.task_name)
