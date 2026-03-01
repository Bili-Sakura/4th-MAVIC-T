#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Run inference on misue-vram checkpoint (first 4 test samples).

Matches the validation logic in examples/ddbm/trainer.py log_validation.
Auto-detects pixel vs latent from checkpoint weights; uses DDBMPipeline for
pixel or DDBMLatentPipeline for latent.

Usage:
    python scripts/run_misue_vram_inference.py \
        --checkpoint ./ckpt/misue-vram/misue-vram_rgb2ir/ddbm/rgb2ir/checkpoint-7000 \
        --output_dir ./ckpt/misue-vram/misue-vram_rgb2ir/ddbm/rgb2ir/test_results/step-007000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from src.pipelines.ddbm import DDBMPipeline, DDBMLatentPipeline
from src.schedulers.scheduling_ddbm import DDBMScheduler
from src.models.unet_ddbm import create_model
from examples.ddbm.dataset_wrapper import MavicTDDBMDataset


def load_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def infer_model_channels(state_dict: dict) -> int:
    """Infer in_channels from checkpoint conv_in shape. (C_in * 2 for concat) -> C_in."""
    w = state_dict["unet.conv_in.weight"]
    in_ch = w.shape[1]
    return in_ch // 2  # concat mode doubles channels


def main():
    parser = argparse.ArgumentParser(description="Inference on misue-vram checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./ckpt/misue-vram/misue-vram_rgb2ir/ddbm/rgb2ir/checkpoint-7000",
        help="Path to checkpoint directory (containing config.yaml and model.safetensors)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for samples (default: checkpoint_dir/test_results/inference)",
    )
    parser.add_argument(
        "--latent_vae_path",
        type=str,
        default="./models/BiliSakura/VAEs/FLUX2-VAE",
        help="VAE path for latent checkpoints (required when checkpoint is latent)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_resolution",
        type=int,
        default=None,
        help="Load & infer at this resolution (e.g. 1024 when trained at 512). Overrides config.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    cfg = load_config(checkpoint_dir)

    task_name = cfg["task_name"]
    resolution = cfg["resolution"]
    model_channels = cfg["model_channels"]
    source_channels = cfg["source_channels"]
    target_channels = cfg["target_channels"]

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load state dict to infer architecture
    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    from safetensors.torch import load_file
    state_dict = load_file(str(model_path))
    inferred_channels = infer_model_channels(state_dict)
    use_latent = inferred_channels == 32  # latent uses 32 ch
    in_channels = inferred_channels

    # Build model with inferred architecture
    # For latent, resolution is spatial compression (e.g. 1024/8=128)
    image_size = resolution // 8 if use_latent else resolution
    model = create_model(
        image_size=image_size,
        in_channels=in_channels,
        num_channels=cfg["num_channels"],
        num_res_blocks=cfg["num_res_blocks"],
        unet_type=cfg.get("unet_type", "adm"),
        attention_resolutions=cfg["attention_resolutions"],
        dropout=cfg.get("dropout", 0.0),
        condition_mode=cfg.get("condition_mode", "concat"),
        channel_mult=cfg["channel_mult"],
    )
    model.load_state_dict(state_dict, strict=True)

    scheduler = DDBMScheduler(
        sigma_min=cfg["sigma_min"],
        sigma_max=cfg["sigma_max"],
        sigma_data=cfg["sigma_data"],
        beta_d=cfg["beta_d"],
        beta_min=cfg["beta_min"],
        pred_mode=cfg["pred_mode"],
        num_train_timesteps=cfg["num_inference_steps"],
    )

    if use_latent:
        from src.utils.latent_target import LatentTargetEncoder
        vae_encoder = LatentTargetEncoder(args.latent_vae_path)
        pipeline = DDBMLatentPipeline(unet=model, scheduler=scheduler, vae=vae_encoder.vae)
    else:
        pipeline = DDBMPipeline(unet=model, scheduler=scheduler)
    pipeline = pipeline.to(device, dtype=torch.bfloat16)

    # Load test dataset (first 4 samples)
    # Use output_resolution for loading when set (512-trained model → 1024 inference)
    load_resolution = args.output_resolution or cfg.get("output_resolution") or resolution
    val_ds = MavicTDDBMDataset(
        task=task_name,
        split="test",
        resolution=load_resolution,
        source_channels=source_channels,
        target_channels=target_channels,
        with_target=False,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "test_results" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {task_name} (first 4 test samples) …")
    saved = 0

    with torch.no_grad():
        for batch in val_loader:
            _zeros, source = batch
            source_01 = source.to(device)
            source_inp = source_01 * 2 - 1  # [0,1] → [-1,1]

            pipeline_kwargs = {
                "source_image": source_inp,
                "num_inference_steps": cfg["num_inference_steps"],
                "guidance": cfg["guidance"],
                "churn_step_ratio": cfg["churn_step_ratio"],
                "output_type": "pt",
            }
            if use_latent:
                pipeline_kwargs["target_channels"] = target_channels
            result = pipeline(**pipeline_kwargs)
            generated = (result.images + 1) * 0.5  # [-1,1] → [0,1]

            # Match channel counts for visualization (same as trainer)
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
                Image.fromarray(concat).save(output_dir / f"sample_{saved:02d}.png")
                saved += 1

            if saved >= 4:
                break

    print(f"Saved {saved} test samples to {output_dir}")


if __name__ == "__main__":
    main()
