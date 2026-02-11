#!/usr/bin/env python
"""Sample (inference) script for trained DDIB models on any MAVIC-T task.

DDIB translates images by:
  1. Encoding source images to a shared latent via DDIM reverse sampling
     with the source-domain diffusion model.
  2. Decoding the latent to the target domain via DDIM forward sampling
     with the target-domain diffusion model.

Usage (combined pipeline checkpoint — recommended)::

    python -m examples.ddib.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./ckpt/ddib/sar2ir/pipeline \
        --split test \
        --output_dir ./samples/ddib_sar2ir \
        --batch_size 32

Use ``--batch_size`` to control inference batch size (default 32). EMA UNets are
loaded by default when ``ema_unet/`` exists under source/target checkpoints.

Usage (separate model checkpoints)::

    python -m examples.ddib.sample \
        --task sar2ir \
        --source_pretrained_path ./ckpt/ddib/source/sar2ir/checkpoint-epoch-100 \
        --target_pretrained_path ./ckpt/ddib/target/sar2ir/checkpoint-epoch-100 \
        --split test \
        --output_dir ./samples/ddib_sar2ir

When ``--pretrained_model_name_or_path`` points to a combined pipeline
directory (containing ``source_unet/``, ``target_unet/``, ``scheduler/``),
the script loads the entire pipeline via ``DDIBPipeline.from_pretrained``.
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

from src.schedulers import DDIBScheduler  # noqa: E402
from src.pipelines.ddib import DDIBPipeline  # noqa: E402

from .config import (  # noqa: E402
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from src.models.unet_ddib import DDIBUNet, create_model  # noqa: E402

# Reuse the DDBM dataset wrapper for loading paired data (source side only)
from examples.ddbm.dataset_wrapper import MavicTDDBMDataset  # noqa: E402

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Translate images using trained DDIB models.")
    parser.add_argument("--task", type=str, required=True, choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to a combined DDIBPipeline directory (contains source_unet/, target_unet/, scheduler/).",
    )
    parser.add_argument(
        "--source_pretrained_path",
        type=str,
        default=None,
        help="Path to the source-domain pretrained directory or legacy .pt/.safetensors file.",
    )
    parser.add_argument(
        "--target_pretrained_path",
        type=str,
        default=None,
        help="Path to the target-domain pretrained directory or legacy .pt/.safetensors file.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=250)
    parser.add_argument("--clip_denoised", type=bool, default=True)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic DDIM sampling (eta=0). "
        "DDIM is typically deterministic by default; eta=0 ensures no stochasticity.",
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
    if args.deterministic:
        # eta=0: fully deterministic DDIM (DDIB default is already 0)
        args.eta = 0.0
    # Normalize device: default single device, or list when multi-GPU
    if args.device is None:
        args.device = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    args.primary_device = args.device[0] if isinstance(args.device, list) else args.device
    args.use_multi_gpu = len(args.device) > 1 and all(d.startswith("cuda") for d in args.device)
    return args


def _load_unet(pretrained_path: str, cfg: TaskConfig, in_channels: int) -> DDIBUNet:
    """Load a single DDIB UNet from a pretrained directory or legacy file."""
    path = Path(pretrained_path)

    if path.is_dir():
        logger.info("Loading UNet from pretrained directory: %s", path)
        subfolder = "ema_unet" if (path / "ema_unet").is_dir() else "unet"
        if subfolder == "ema_unet":
            logger.info("Loading EMA UNet from %s", path / "ema_unet")
        return DDIBUNet.from_pretrained(pretrained_path, subfolder=subfolder)

    # ---- legacy single-file checkpoint ----
    logger.info("Loading UNet from legacy checkpoint: %s", path)
    model = create_model(
        image_size=cfg.resolution,
        in_channels=in_channels,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
        dropout=0.0,
        learn_sigma=cfg.learn_sigma,
        channel_mult=cfg.channel_mult,
    )
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        ckpt = load_file(str(path))
    else:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    return model


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

    # Load pipeline
    if args.pretrained_model_name_or_path:
        # ---- combined pipeline directory ----
        logger.info("Loading DDIBPipeline from: %s", args.pretrained_model_name_or_path)
        pipeline = DDIBPipeline.from_pretrained(args.pretrained_model_name_or_path)
        pp = Path(args.pretrained_model_name_or_path)
        if (pp / "source_ema_unet").is_dir():
            logger.info("Loading source EMA UNet")
            pipeline.source_unet = DDIBUNet.from_pretrained(str(pp), subfolder="source_ema_unet")
        if (pp / "target_ema_unet").is_dir():
            logger.info("Loading target EMA UNet")
            pipeline.target_unet = DDIBUNet.from_pretrained(str(pp), subfolder="target_ema_unet")
        pipeline = pipeline.to(args.primary_device)
        if device_ids is not None and len(device_ids) > 1:
            pipeline.source_unet = torch.nn.DataParallel(pipeline.source_unet, device_ids=device_ids)
            pipeline.target_unet = torch.nn.DataParallel(pipeline.target_unet, device_ids=device_ids)
            logger.info("Using DataParallel on GPUs %s", device_ids)
    elif args.source_pretrained_path and args.target_pretrained_path:
        # ---- separate source/target paths ----
        source_model = _load_unet(args.source_pretrained_path, cfg, cfg.source_channels)
        source_model = source_model.to(args.primary_device).eval()

        target_model = _load_unet(args.target_pretrained_path, cfg, cfg.target_channels)
        target_model = target_model.to(args.primary_device).eval()

        if device_ids is not None and len(device_ids) > 1:
            source_model = torch.nn.DataParallel(source_model, device_ids=device_ids)
            target_model = torch.nn.DataParallel(target_model, device_ids=device_ids)
            logger.info("Using DataParallel on GPUs %s", device_ids)

        # Prefer loading scheduler from a pretrained directory if available
        source_path = Path(args.source_pretrained_path)
        if source_path.is_dir() and (source_path / "scheduler").is_dir():
            scheduler = DDIBScheduler.from_pretrained(args.source_pretrained_path, subfolder="scheduler")
        else:
            scheduler = DDIBScheduler(
                num_train_timesteps=cfg.diffusion_steps,
                noise_schedule=cfg.noise_schedule,
                learn_sigma=cfg.learn_sigma,
                predict_xstart=cfg.predict_xstart,
                rescale_timesteps=cfg.rescale_timesteps,
            )
        pipeline = DDIBPipeline(
            source_unet=source_model,
            target_unet=target_model,
            scheduler=scheduler,
        )
    else:
        raise ValueError(
            "Provide either --pretrained_model_name_or_path for a combined pipeline "
            "or both --source_pretrained_path and --target_pretrained_path."
        )

    # Load evaluation data (source side only)
    dataset = MavicTDDBMDataset(
        task=args.task,
        split=args.split,
        resolution=cfg.resolution,
        model_channels=cfg.source_channels,
        with_target=False,
    )
    if args.skip_existing:
        indices_to_process = [
            i for i in range(len(dataset))
            if not (output_dir / f"sample_{i:05d}.png").exists()
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
    if isinstance(dataset, Subset):
        def original_indices(i):
            return dataset.indices[i]
    else:
        def original_indices(i):
            return i

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="DDIB Translation"):
            # batch = (zeros_target, source)
            source = batch[1].to(args.primary_device) * 2 - 1  # [0,1] → [-1,1]

            result = pipeline(
                source_image=source,
                num_inference_steps=args.num_inference_steps,
                clip_denoised=args.clip_denoised,
                eta=args.eta,
                output_type="pt",
            )
            images = result.images  # (B, C, H, W) in [-1, 1]
            images_uint8 = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            images_uint8 = images_uint8.permute(0, 2, 3, 1).cpu().numpy()

            for i, img_arr in enumerate(images_uint8):
                out_idx = original_indices(sample_idx + i)
                if img_arr.shape[2] == 1:
                    img_arr = img_arr.squeeze(2)
                img = Image.fromarray(img_arr)
                img.save(output_dir / f"sample_{out_idx:05d}.png")
            sample_idx += len(images_uint8)
            all_samples.append(images_uint8)

    all_samples = np.concatenate(all_samples, axis=0)

    if args.save_npz:
        np.savez(output_dir / f"samples_{len(all_samples)}.npz", arr_0=all_samples)
        logger.info(f"Saved NPZ with {len(all_samples)} samples.")

    logger.info(f"Sampling complete – {sample_idx} images saved to {output_dir}")


if __name__ == "__main__":
    main()
