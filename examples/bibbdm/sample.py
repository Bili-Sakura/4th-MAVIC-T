#!/usr/bin/env python
"""Sample (inference) script for a trained BiBBDM model on any MAVIC-T task.

Usage (pretrained directory — recommended)::

    python -m examples.bibbdm.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./ckpt/bibbdm/sar2ir/checkpoint-epoch-100 \
        --split test \
        --output_dir ./samples/sar2ir \
        --batch_size 32

Use ``--batch_size`` to control inference batch size (default 32). EMA UNet is
loaded by default when ``ema_unet/`` exists under the checkpoint.

Usage (legacy ``.pt`` file)::

    python -m examples.bibbdm.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./outputs/bibbdm_sar2ir/model_epoch_100.pt \
        --split test \
        --output_dir ./samples/sar2ir

When a directory is provided the script loads the UNet and scheduler via
``from_pretrained`` following the HuggingFace *diffusers* convention.
Legacy ``.pt`` / ``.safetensors`` single-file checkpoints are still
supported for backward compatibility.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.schedulers import BiBBDMScheduler  # noqa: E402
from src.pipelines.bibbdm import BiBBDMPipeline  # noqa: E402

from .config import (  # noqa: E402
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from .dataset_wrapper import MavicTBiBBDMDataset  # noqa: E402
from src.models.unet_bibbdm import BiBBDMUNet, create_model  # noqa: E402

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def _get_output_name(dataset_obj, idx: int) -> str:
    """Resolve submission filename for index, preserving Subset mapping."""
    if isinstance(dataset_obj, Subset):
        base_dataset = dataset_obj.dataset
        base_idx = dataset_obj.indices[idx]
    else:
        base_dataset = dataset_obj
        base_idx = idx

    if hasattr(base_dataset, "get_output_name"):
        return base_dataset.get_output_name(base_idx)

    records = getattr(base_dataset, "_records", None)
    if records is not None:
        stem = Path(records[base_idx]["input_path"]).stem
        return f"{stem}.png"
    return f"sample_{base_idx:05d}.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained BiBBDM model.")
    parser.add_argument("--task", type=str, required=True, choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to a diffusers-style checkpoint directory or a legacy .pt/.safetensors file.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--direction", type=str, default="b2a", choices=["b2a", "a2b"])
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic sampling (eta=0, no stochastic noise in steps). "
        "May trade quality for speed/reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="+",
        default=None,
        help="Device(s) for inference. Single: 'cuda:0'. Multi-GPU: 'cuda:0' 'cuda:1'. Uses DataParallel for multi-GPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples whose output file already exists (default: True, for resumability).",
    )
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--save_npz", action="store_true")
    args = parser.parse_args()
    # Normalize device: default single device, or list when multi-GPU
    if args.device is None:
        args.device = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    args.primary_device = args.device[0] if isinstance(args.device, list) else args.device
    args.use_multi_gpu = len(args.device) > 1 and all(d.startswith("cuda") for d in args.device)
    return args


def _load_pipeline(
    pretrained_path: str,
    cfg: TaskConfig,
    primary_device: str,
    num_inference_steps: int,
    device_ids: list[int] | None = None,
) -> BiBBDMPipeline:
    """Load the BiBBDM pipeline from a pretrained directory or legacy file."""
    path = Path(pretrained_path)

    if path.is_dir():
        # ---- diffusers from_pretrained path ----
        logger.info("Loading pipeline from pretrained directory: %s", path)
        pipeline = BiBBDMPipeline.from_pretrained(pretrained_path)
        ema_unet_dir = path / "ema_unet"
        if ema_unet_dir.is_dir():
            logger.info("Loading EMA UNet from %s", ema_unet_dir)
            pipeline.unet = BiBBDMUNet.from_pretrained(pretrained_path, subfolder="ema_unet")
    else:
        # ---- legacy single-file checkpoint ----
        logger.info("Loading model from legacy checkpoint: %s", path)
        model = create_model(
            image_size=cfg.resolution,
            in_channels=cfg.model_channels,
            num_channels=cfg.num_channels,
            num_res_blocks=cfg.num_res_blocks,
            attention_resolutions=cfg.attention_resolutions,
            dropout=0.0,
            condition_mode=cfg.condition_mode,
            channel_mult=cfg.channel_mult,
            objective=cfg.objective,
            unet_type=getattr(cfg, "unet_type", "adm"),
        )
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(str(path))
        else:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)

        scheduler = BiBBDMScheduler(
            num_timesteps=cfg.num_timesteps,
            mt_type=cfg.mt_type,
            m0=cfg.m0,
            mT=cfg.mT,
            eta=cfg.eta,
            var_scale=cfg.var_scale,
            skip_sample=cfg.skip_sample,
            sample_step=num_inference_steps,
            sample_step_type=cfg.sample_step_type,
            objective=cfg.objective,
        )
        pipeline = BiBBDMPipeline(unet=model, scheduler=scheduler)

    pipeline = pipeline.to(primary_device)
    if device_ids is not None and len(device_ids) > 1:
        pipeline.unet = torch.nn.DataParallel(pipeline.unet, device_ids=device_ids)
        logger.info("Using DataParallel on GPUs %s", device_ids)
    return pipeline


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    cfg: TaskConfig = _TASK_CONFIG_MAP[args.task]()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_ids = None
    if args.use_multi_gpu:
        device_ids = [
            int(d.split(":")[1]) if ":" in d else i
            for i, d in enumerate(args.device)
        ]
    pipeline = _load_pipeline(
        args.pretrained_model_name_or_path,
        cfg,
        args.primary_device,
        args.num_inference_steps,
        device_ids=device_ids,
    )

    if args.deterministic:
        # eta=0: no stochastic noise in Brownian Bridge steps
        pipeline.scheduler.eta = 0.0
        pipeline.scheduler.config.eta = 0.0

    # Load evaluation data
    dataset = MavicTBiBBDMDataset(
        task=args.task,
        split=args.split,
        resolution=cfg.resolution,
        model_channels=cfg.model_channels,
        with_target=False,
    )
    if args.skip_existing:
        indices_to_process = [
            i for i in range(len(dataset))
            if not (output_dir / _get_output_name(dataset, i)).exists()
        ]
        n_total = len(dataset)
        if not indices_to_process:
            logger.info("All %d outputs already exist in %s. Nothing to do.", n_total, output_dir)
            return
        if len(indices_to_process) < n_total:
            dataset = Subset(dataset, indices_to_process)
            logger.info("Skipping %d existing, processing %d remaining.", n_total - len(indices_to_process), len(indices_to_process))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    eta_info = " (deterministic)" if args.deterministic else ""
    logger.info(f"Generating samples for {args.task} ({args.split}), {len(dataset)} inputs{eta_info} …")
    all_samples = []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Sampling"):
            source = batch[1].to(args.primary_device) * 2 - 1  # [0,1] → [-1,1]

            result = pipeline(
                source_image=source,
                direction=args.direction,
                num_inference_steps=args.num_inference_steps,
                clip_denoised=args.clip_denoised,
                output_type="pt",
            )
            images = result.images
            images_uint8 = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            images_uint8 = images_uint8.permute(0, 2, 3, 1).cpu().numpy()

            for i, img_arr in enumerate(images_uint8):
                out_name = _get_output_name(dataset, sample_idx + i)
                if img_arr.shape[2] == 1:
                    img_arr = img_arr.squeeze(2)
                img = Image.fromarray(img_arr)
                img.save(output_dir / out_name)
            sample_idx += len(images_uint8)
            all_samples.append(images_uint8)

    all_samples = np.concatenate(all_samples, axis=0)

    if args.save_npz:
        np.savez(output_dir / f"samples_{len(all_samples)}.npz", arr_0=all_samples)
        logger.info(f"Saved NPZ with {len(all_samples)} samples.")

    logger.info(f"Sampling complete – {sample_idx} images saved to {output_dir}")


if __name__ == "__main__":
    main()
