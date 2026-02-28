#!/usr/bin/env python
"""Sample (inference) script for Text2Earth InstructPix2Pix-style SAR2RGB.

Runs a trained checkpoint on the MAVIC-T test set and saves outputs in
submission format (PNG per input, matching stem names).

Usage:
    python -m examples.text2earth_sar2rgb.sample_sar2rgb \
        --checkpoint_path /path/to/checkpoint-20000 \
        --split test \
        --output_dir ./outputs/text2earth_sar2rgb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, PretrainedConfig

from examples.ddbm.dataset_wrapper import MavicTDDBMDataset, PairedValDataset, resolve_paired_val_manifest
from src.utils.paths import get_project_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Text2Earth InstructPix2Pix SAR2RGB inference on MAVIC-T test set."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/data/projects/4th-MAVIC-T/ckpt/EXP_0225_text2earth_sar2rgb_instructpix2pix/checkpoint-20000",
        help="Path to the trained checkpoint directory (e.g. checkpoint-20000).",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/lcybuaa/Text2Earth",
        help="Path to Text2Earth base model (for VAE, text encoder, scheduler).",
    )
    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default="models/lcybuaa/Text2Earth/vae",
        help="Path to VAE.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to run on.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: datasets/BiliSakura/MACIV-T-2025-Submissions/sar2rgb/<checkpoint_name>.",
    )
    parser.add_argument(
        "--submission_root",
        type=str,
        default=None,
        help="Root for submissions. Default: datasets/BiliSakura/MACIV-T-2025-Submissions.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="text2earth_instructpix2pix",
        help="Model name for output folder.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Input/output resolution. Model was trained at 512; 1024 may generalize.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="18_GOOGLE_LEVEL_ a satellite optical image",
        help="Text prompt (must match training).",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=0,
        help="Text2Earth resolution class label.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=200,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples whose output file already exists.",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to paired_val manifest (input\\ttarget per line). When set, uses PairedValDataset instead of split.",
    )
    parser.add_argument(
        "--save_input_gt",
        action="store_true",
        help="When used with --manifest, save input/ generation/ gt/ subdirs (for temp/ comparisons).",
    )
    return parser.parse_args()


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return get_project_root() / raw


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    raise ValueError(f"{model_class} is not supported.")


@torch.no_grad()
def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    root = get_project_root()
    ckpt_path = _resolve_path(args.checkpoint_path)
    pretrained_path = _resolve_path(args.pretrained_model_name_or_path)
    vae_path = _resolve_path(args.vae_model_name_or_path)

    if not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        sub_root = args.submission_root or str(root / "datasets/BiliSakura/MACIV-T-2025-Submissions")
        output_dir = Path(sub_root) / "sar2rgb" / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_input_gt:
        (output_dir / "input").mkdir(exist_ok=True)
        (output_dir / "gt").mkdir(exist_ok=True)
        gen_dir = output_dir / "generation"
        gen_dir.mkdir(exist_ok=True)
    else:
        gen_dir = output_dir

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load base components from Text2Earth
    tokenizer = AutoTokenizer.from_pretrained(
        str(pretrained_path),
        subfolder="tokenizer",
        use_fast=False,
    )
    vae = AutoencoderKL.from_pretrained(str(vae_path)).to(device, dtype=dtype)
    noise_scheduler = DDIMScheduler.from_pretrained(
        str(pretrained_path),
        subfolder="scheduler",
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(str(pretrained_path))
    text_encoder = text_encoder_cls.from_pretrained(
        str(pretrained_path),
        subfolder="text_encoder",
    ).to(device, dtype=dtype)

    # Load fine-tuned UNet from checkpoint
    unet_subfolder = "ema_unet" if (ckpt_path / "ema_unet").is_dir() else "unet"
    unet = UNet2DConditionModel.from_pretrained(
        str(ckpt_path),
        subfolder=unet_subfolder,
        torch_dtype=dtype,
    ).to(device)
    unet.eval()

    has_class_embedding = getattr(unet, "class_embedding", None) is not None

    # Dataset: use PairedValDataset when manifest given, else MavicTDDBMDataset
    if args.manifest:
        manifest_resolved = resolve_paired_val_manifest(args.manifest)
        manifest_path = manifest_resolved or Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        dataset = PairedValDataset(
            manifest_path=manifest_path,
            resolution=args.resolution,
            source_channels=3,
            target_channels=3,
            return_order="target_source",  # match MavicTDDBMDataset: (target, source)
        )
    else:
        dataset = MavicTDDBMDataset(
            task="sar2rgb",
            split=args.split,
            resolution=args.resolution,
            source_channels=3,
            target_channels=3,
            with_target=False,
        )

    def get_output_name(ds, idx: int) -> str:
        if hasattr(ds, "get_output_name"):
            return ds.get_output_name(idx)
        stem = Path(ds._pairs[idx][0]).stem
        return f"{stem}.png"

    class _SubsetWithOutputNames(Subset):
        def get_output_name(self, idx: int) -> str:
            base_idx = self.indices[idx]
            return get_output_name(self.dataset, base_idx)

    if args.skip_existing:
        indices = [i for i in range(len(dataset)) if not (gen_dir / get_output_name(dataset, i)).exists()]
        if not indices:
            print(f"All {len(dataset)} outputs already exist in {gen_dir}. Nothing to do.")
            return
        dataset = _SubsetWithOutputNames(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Tokenize caption once
    text_inputs = tokenizer(
        args.caption,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]
    # Expand for batches: (1, seq, dim) -> repeat for first batch, then we'll index
    def get_emb(batch_size):
        return encoder_hidden_states.repeat(batch_size, 1, 1)

    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
    sample_idx = 0

    split_or_manifest = f"manifest={args.manifest}" if args.manifest else f"split={args.split}"
    print(f"Running inference: {split_or_manifest}, {len(dataset)} images -> {gen_dir}")
    print(f"Checkpoint: {ckpt_path}, steps={args.num_inference_steps}")

    for batch in tqdm(dataloader, desc="Sampling"):
        # batch is (target, source) from MavicTDDBMDataset or PairedValDataset(return_order=target_source)
        source = batch[1]  # (B, 3, H, W) in [0, 1]
        target = batch[0]
        B = source.shape[0]
        source = source.to(device, dtype=dtype)
        # Convert [0,1] to [-1,1] for VAE (matches training)
        source_norm = 2.0 * source - 1.0
        # Encode SAR to latent (3ch input for VAE)
        sar_latents = vae.encode(source_norm).latent_dist.sample()
        sar_latents = sar_latents * vae.config.scaling_factor

        # Start from noise
        latents = torch.randn_like(sar_latents, device=device, dtype=dtype)

        emb = get_emb(B)

        for t in noise_scheduler.timesteps:
            latent_input = torch.cat([latents, sar_latents], dim=1)
            unet_kwargs = {"encoder_hidden_states": emb, "return_dict": False}
            if has_class_embedding:
                unet_kwargs["class_labels"] = torch.full(
                    (B,), args.class_label, dtype=torch.long, device=device
                )
            noise_pred = unet(latent_input, t, **unet_kwargs)[0]
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        pred_pixels = vae.decode(latents / vae.config.scaling_factor).sample
        pred_pixels = ((pred_pixels / 2) + 0.5).clamp(0, 1)

        for i in range(B):
            idx = sample_idx + i
            out_name = get_output_name(dataset, idx)
            arr = (pred_pixels[i].float().cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            Image.fromarray(arr).save(gen_dir / out_name)
            if args.save_input_gt:
                src_arr = (source[i].float().cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
                Image.fromarray(src_arr).save(output_dir / "input" / out_name)
                tgt_arr = (target[i].float().cpu().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
                Image.fromarray(tgt_arr).save(output_dir / "gt" / out_name)
        sample_idx += B

    print(f"Done. Saved {sample_idx} images to {gen_dir}")


if __name__ == "__main__":
    main()
