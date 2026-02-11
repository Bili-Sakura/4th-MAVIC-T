"""Training loop for target-domain real/fake classifier pre-training."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Optional

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import torch
from torch.utils.data import DataLoader
from transformers import ResNetConfig, ResNetForImageClassification, get_scheduler

# Ensure project root is importable when executed as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import (  # noqa: E402
    DomainClassifierConfig,
    normalization_stats_for_channels,
)
from .dataset_wrapper import (  # noqa: E402
    BinaryDomainImageDataset,
    build_binary_records,
    split_binary_records,
    summarize_binary_records,
)


LOGGER = get_logger(__name__, log_level="INFO")


def _find_input_conv_key(
    pretrained_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> Optional[str]:
    preferred = [
        key
        for key, tensor in target_state_dict.items()
        if key in pretrained_state_dict
        and tensor.ndim == 4
        and "embedder" in key
        and key.endswith("convolution.weight")
    ]
    if preferred:
        return preferred[0]

    fallback = [
        key
        for key, tensor in target_state_dict.items()
        if key in pretrained_state_dict and tensor.ndim == 4
    ]
    return fallback[0] if fallback else None


def _adapt_input_conv_weight(weight: torch.Tensor, target_in_channels: int) -> torch.Tensor:
    """Adapt pretrained input conv weights to a new input-channel count."""
    if weight.shape[1] == target_in_channels:
        return weight

    if target_in_channels == 1:
        return weight.mean(dim=1, keepdim=True)

    current_in = weight.shape[1]
    if target_in_channels < current_in:
        scale = current_in / float(target_in_channels)
        return weight[:, :target_in_channels] * scale

    repeats = math.ceil(target_in_channels / current_in)
    expanded = weight.repeat(1, repeats, 1, 1)[:, :target_in_channels]
    # Keep activation magnitude similar when expanding channels.
    expanded = expanded * (current_in / float(target_in_channels))
    return expanded


def build_resnet18_binary_classifier(
    pretrained_model_name_or_path: str,
    *,
    num_channels: int,
    image_size: int,
) -> tuple[ResNetForImageClassification, dict[str, object]]:
    """Build ResNet-18 classifier and load adapted pretrained weights.

    Initializes from ``Francesco/resnet18-224-1k`` (or a custom HF path), then:
      - rewires ``num_channels`` for 1ch/3ch domains,
      - replaces 1000-way head with 2-way real/fake head,
      - adapts first convolution weights when channel count changes.
    """

    config = ResNetConfig.from_pretrained(pretrained_model_name_or_path)
    config.num_channels = int(num_channels)
    config.num_labels = 2
    config.problem_type = "single_label_classification"
    config.id2label = {0: "fake", 1: "real"}
    config.label2id = {"fake": 0, "real": 1}
    # Keep native spatial size in config for traceability.
    config.image_size = int(image_size)

    model = ResNetForImageClassification(config)
    target_state = model.state_dict()

    pretrained_model = ResNetForImageClassification.from_pretrained(
        pretrained_model_name_or_path
    )
    pretrained_state = pretrained_model.state_dict()

    input_conv_key = _find_input_conv_key(pretrained_state, target_state)
    if input_conv_key is not None:
        pretrained_state[input_conv_key] = _adapt_input_conv_weight(
            pretrained_state[input_conv_key],
            target_in_channels=target_state[input_conv_key].shape[1],
        )

    filtered_state = {}
    dropped_keys = []
    for key, tensor in pretrained_state.items():
        if key not in target_state:
            dropped_keys.append(key)
            continue
        if tensor.shape != target_state[key].shape:
            dropped_keys.append(key)
            continue
        filtered_state[key] = tensor

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    load_report = {
        "input_conv_key": input_conv_key,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "dropped_pretrained_keys": dropped_keys,
    }
    return model, load_report


def _resolve_resume_checkpoint(
    run_dir: Path,
    resume_from_checkpoint: Optional[str],
) -> Optional[Path]:
    if not resume_from_checkpoint:
        return None

    if resume_from_checkpoint == "latest":
        pattern = re.compile(r"^checkpoint-epoch-(\d+)$")
        candidates: list[tuple[int, Path]] = []
        if run_dir.exists():
            for child in run_dir.iterdir():
                match = pattern.match(child.name)
                if child.is_dir() and match:
                    candidates.append((int(match.group(1)), child))
        if not candidates:
            raise FileNotFoundError(
                f"No epoch checkpoint found in {run_dir} for resume_from_checkpoint=latest"
            )
        return sorted(candidates, key=lambda x: x[0])[-1][1]

    checkpoint = Path(resume_from_checkpoint)
    if checkpoint.is_dir():
        return checkpoint

    local = run_dir / resume_from_checkpoint
    if local.is_dir():
        return local

    raise FileNotFoundError(
        f"Could not resolve checkpoint '{resume_from_checkpoint}' as an existing directory."
    )


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _cleanup_old_checkpoints(run_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    pattern = re.compile(r"^checkpoint-epoch-(\d+)$")
    checkpoints: list[tuple[int, Path]] = []
    if not run_dir.exists():
        return
    for child in run_dir.iterdir():
        match = pattern.match(child.name)
        if child.is_dir() and match:
            checkpoints.append((int(match.group(1)), child))
    checkpoints.sort(key=lambda item: item[0])
    for _, old_path in checkpoints[:-keep]:
        shutil.rmtree(old_path, ignore_errors=True)


class DomainClassifierTrainer:
    """End-to-end trainer for target-domain ResNet-18 real/fake classifier."""

    def __init__(self, cfg: DomainClassifierConfig) -> None:
        self.cfg = cfg

    @torch.no_grad()
    def evaluate(
        self,
        model: ResNetForImageClassification,
        dataloader: DataLoader,
        accelerator: Accelerator,
    ) -> dict[str, float]:
        model.eval()
        device = accelerator.device

        loss_sum = torch.zeros(1, device=device)
        correct_sum = torch.zeros(1, device=device)
        count_sum = torch.zeros(1, device=device)
        real_prob_sum = torch.zeros(1, device=device)
        real_prob_pos_sum = torch.zeros(1, device=device)
        real_prob_neg_sum = torch.zeros(1, device=device)
        pos_count_sum = torch.zeros(1, device=device)
        neg_count_sum = torch.zeros(1, device=device)

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            probs_real = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            batch_size = labels.shape[0]
            loss_sum += outputs.loss.detach() * batch_size
            correct_sum += (preds == labels).sum()
            count_sum += batch_size
            real_prob_sum += probs_real.sum()

            pos_mask = labels == 1
            neg_mask = labels == 0
            if pos_mask.any():
                real_prob_pos_sum += probs_real[pos_mask].sum()
                pos_count_sum += pos_mask.sum()
            if neg_mask.any():
                real_prob_neg_sum += probs_real[neg_mask].sum()
                neg_count_sum += neg_mask.sum()

        packed = torch.cat(
            [
                loss_sum,
                correct_sum,
                count_sum,
                real_prob_sum,
                real_prob_pos_sum,
                real_prob_neg_sum,
                pos_count_sum,
                neg_count_sum,
            ]
        ).unsqueeze(0)
        gathered = accelerator.gather_for_metrics(packed)
        if gathered.ndim == 2:
            gathered = gathered.sum(dim=0)
        elif gathered.ndim == 1:
            gathered = gathered.reshape(-1, 8).sum(dim=0)

        total_count = max(gathered[2].item(), 1.0)
        total_pos = max(gathered[6].item(), 1.0)
        total_neg = max(gathered[7].item(), 1.0)

        metrics = {
            "loss": gathered[0].item() / total_count,
            "accuracy": gathered[1].item() / total_count,
            "mean_p_real": gathered[3].item() / total_count,
            "mean_p_real_on_real": gathered[4].item() / total_pos,
            "mean_p_real_on_fake": gathered[5].item() / total_neg,
            "num_samples": gathered[2].item(),
        }
        model.train()
        return metrics

    def train(self) -> dict[str, object]:
        cfg = self.cfg
        target_domain = cfg.resolved_target_domain()
        resolution = cfg.resolved_resolution()
        num_channels = cfg.resolved_num_channels()
        normalize_mean, normalize_std = normalization_stats_for_channels(num_channels)

        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            mixed_precision=cfg.mixed_precision,
        )
        set_seed(cfg.seed)

        run_dir = Path(cfg.output_dir) / cfg.resolved_run_name()
        if accelerator.is_main_process:
            run_dir.mkdir(parents=True, exist_ok=True)

        records = build_binary_records(cfg)
        train_records, val_records = split_binary_records(
            records,
            val_ratio=cfg.val_split_ratio,
            seed=cfg.seed,
        )

        train_summary = summarize_binary_records(train_records)
        val_summary = summarize_binary_records(val_records) if val_records else {}

        if accelerator.is_main_process:
            LOGGER.info("Training records: %s", train_summary)
            if val_records:
                LOGGER.info("Validation records: %s", val_summary)

        train_dataset = BinaryDomainImageDataset(
            train_records,
            resolution=resolution,
            num_channels=num_channels,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            use_horizontal_flip=cfg.use_horizontal_flip,
            use_vertical_flip=cfg.use_vertical_flip,
        )
        val_dataset = (
            BinaryDomainImageDataset(
                val_records,
                resolution=resolution,
                num_channels=num_channels,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
                use_horizontal_flip=False,
                use_vertical_flip=False,
            )
            if val_records
            else None
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.dataloader_num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_dataloader = (
            DataLoader(
                val_dataset,
                batch_size=cfg.eval_batch_size,
                shuffle=False,
                num_workers=cfg.dataloader_num_workers,
                pin_memory=True,
                drop_last=False,
            )
            if val_dataset is not None
            else None
        )

        resume_dir = _resolve_resume_checkpoint(run_dir, cfg.resume_from_checkpoint)
        if resume_dir is None:
            model, load_report = build_resnet18_binary_classifier(
                cfg.pretrained_model_name_or_path,
                num_channels=num_channels,
                image_size=resolution,
            )
            resume_state = None
        else:
            model = ResNetForImageClassification.from_pretrained(str(resume_dir))
            load_report = {"resumed_from_checkpoint": str(resume_dir)}
            state_path = resume_dir / "trainer_state.pt"
            resume_state = (
                torch.load(state_path, map_location="cpu")
                if state_path.is_file()
                else None
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        num_update_steps_per_epoch = max(
            1, math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
        )
        max_train_steps = cfg.max_train_steps or (cfg.num_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=max_train_steps,
        )

        start_epoch = 0
        global_step = 0
        if resume_state is not None:
            optimizer_state = resume_state.get("optimizer")
            scheduler_state = resume_state.get("lr_scheduler")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            if scheduler_state is not None:
                lr_scheduler.load_state_dict(scheduler_state)
            start_epoch = int(resume_state.get("epoch", 0)) + 1
            global_step = int(resume_state.get("global_step", 0))
            LOGGER.info(
                "Resuming from epoch=%s global_step=%s",
                start_epoch,
                global_step,
            )

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        if val_dataloader is not None:
            val_dataloader = accelerator.prepare(val_dataloader)

        if accelerator.is_main_process:
            payload = {
                "config": vars(cfg),
                "resolved": {
                    "target_domain": target_domain,
                    "resolution": resolution,
                    "num_channels": num_channels,
                    "normalize_mean": normalize_mean,
                    "normalize_std": normalize_std,
                    "positive_tasks": cfg.resolved_positive_tasks(),
                    "negative_target_tasks": cfg.resolved_negative_target_tasks(),
                },
                "dataset_summary": {
                    "train": train_summary,
                    "val": val_summary,
                },
                "pretrained_load_report": load_report,
            }
            _save_json(run_dir / "training_setup.json", payload)

        best_metric = float("inf")
        best_checkpoint = None
        completed_epochs = start_epoch

        LOGGER.info("***** Running domain-classifier training *****")
        LOGGER.info("  target_domain      = %s", target_domain)
        LOGGER.info("  num_channels       = %s", num_channels)
        LOGGER.info("  resolution         = %s", resolution)
        LOGGER.info("  train_samples      = %s", len(train_dataset))
        LOGGER.info("  val_samples        = %s", len(val_dataset) if val_dataset else 0)
        LOGGER.info("  batch_size         = %s", cfg.train_batch_size)
        LOGGER.info("  num_epochs         = %s", num_train_epochs)
        LOGGER.info("  max_train_steps    = %s", max_train_steps)

        for epoch in range(start_epoch, num_train_epochs):
            model.train()
            device = accelerator.device

            local_loss_sum = torch.zeros(1, device=device)
            local_correct_sum = torch.zeros(1, device=device)
            local_count_sum = torch.zeros(1, device=device)

            for batch in train_dataloader:
                with accelerator.accumulate(model):
                    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    batch_size = labels.shape[0]
                    local_loss_sum += loss.detach() * batch_size
                    local_correct_sum += (preds == labels).sum()
                    local_count_sum += batch_size

                if accelerator.sync_gradients:
                    global_step += 1
                    if (
                        cfg.log_every_steps > 0
                        and global_step % cfg.log_every_steps == 0
                        and accelerator.is_main_process
                    ):
                        LOGGER.info(
                            "step=%d/%d loss=%.5f lr=%.6g",
                            global_step,
                            max_train_steps,
                            float(loss.detach().item()),
                            float(lr_scheduler.get_last_lr()[0]),
                        )

                if global_step >= max_train_steps:
                    break

            train_packed = torch.cat(
                [local_loss_sum, local_correct_sum, local_count_sum]
            ).unsqueeze(0)
            gathered = accelerator.gather_for_metrics(train_packed)
            if gathered.ndim == 2:
                gathered = gathered.sum(dim=0)
            elif gathered.ndim == 1:
                gathered = gathered.reshape(-1, 3).sum(dim=0)
            train_count = max(gathered[2].item(), 1.0)
            train_metrics = {
                "loss": gathered[0].item() / train_count,
                "accuracy": gathered[1].item() / train_count,
                "num_samples": gathered[2].item(),
            }

            val_metrics = {}
            if val_dataloader is not None:
                val_metrics = self.evaluate(model, val_dataloader, accelerator)

            target_metric = (
                val_metrics.get("loss", train_metrics["loss"])
                if val_metrics
                else train_metrics["loss"]
            )
            if target_metric < best_metric:
                best_metric = target_metric
                best_checkpoint = f"checkpoint-epoch-{epoch + 1:03d}"

            if accelerator.is_main_process:
                LOGGER.info(
                    "epoch=%d train_loss=%.5f train_acc=%.4f%s",
                    epoch + 1,
                    train_metrics["loss"],
                    train_metrics["accuracy"],
                    (
                        f" val_loss={val_metrics['loss']:.5f} "
                        f"val_acc={val_metrics['accuracy']:.4f} "
                        f"val_mean_p_real={val_metrics['mean_p_real']:.4f}"
                        if val_metrics
                        else ""
                    ),
                )

                if cfg.checkpoint_every_epochs > 0 and (epoch + 1) % cfg.checkpoint_every_epochs == 0:
                    ckpt_dir = run_dir / f"checkpoint-epoch-{epoch + 1:03d}"
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(str(ckpt_dir), safe_serialization=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "train_metrics": train_metrics,
                            "val_metrics": val_metrics,
                        },
                        ckpt_dir / "trainer_state.pt",
                    )
                    _save_json(
                        ckpt_dir / "metrics.json",
                        {
                            "train": train_metrics,
                            "val": val_metrics,
                            "global_step": global_step,
                            "epoch": epoch + 1,
                        },
                    )

                    # Keep a copy of best model for easier downstream scoring.
                    if best_checkpoint == ckpt_dir.name:
                        best_dir = run_dir / "best"
                        if best_dir.exists():
                            shutil.rmtree(best_dir)
                        shutil.copytree(ckpt_dir, best_dir)

                    _cleanup_old_checkpoints(run_dir, keep=cfg.checkpoints_total_limit)

            completed_epochs = epoch + 1
            if global_step >= max_train_steps:
                break

        if accelerator.is_main_process:
            final_dir = run_dir / "final"
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(str(final_dir), safe_serialization=True)
            _save_json(
                run_dir / "training_summary.json",
                {
                    "best_metric": best_metric,
                    "best_checkpoint": best_checkpoint,
                    "global_step": global_step,
                    "completed_epochs": completed_epochs,
                    "max_train_steps": max_train_steps,
                },
            )

        accelerator.wait_for_everyone()
        return {
            "run_dir": str(run_dir),
            "best_metric": best_metric,
            "best_checkpoint": best_checkpoint,
            "global_step": global_step,
            "max_train_steps": max_train_steps,
        }

