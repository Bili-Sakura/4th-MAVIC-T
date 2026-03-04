#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Sample (inference) script for a trained StegoGAN model on any MAVIC-T task.

Usage::

    python -m examples.stegogan.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./ckpt/stegogan/sar2ir/checkpoint-epoch-100 \
        --split test \
        --output_dir ./samples/stegogan_sar2ir \
        --batch_size 32
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .config import (  # noqa: E402
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from .dataset_wrapper import MavicTStegoGANDataset  # noqa: E402
from src.models.gan.stegogan_model import StegoGANGeneratorA, create_generator_a  # noqa: E402
from src.pipelines.stegogan import StegoGANPipeline  # noqa: E402
from src.utils.readme_utils import (  # noqa: E402
    load_checkpoint_config,
    build_detailed_description,
    write_readme,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


def _get_output_name(dataset_obj, idx: int) -> str:
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
    parser = argparse.ArgumentParser(description="Sample from a trained StegoGAN generator.")
    parser.add_argument("--task", type=str, required=True, choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--extra_data", action="store_true")
    parser.add_argument("--readme_description", type=str, default="")
    parser.add_argument("--device", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_npz", action="store_true")
    args = parser.parse_args()
    if args.device is None:
        args.device = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    args.primary_device = args.device[0] if isinstance(args.device, list) else args.device
    args.use_multi_gpu = len(args.device) > 1 and all(d.startswith("cuda") for d in args.device)
    return args


def _load_pipeline(pretrained_path: str, cfg: TaskConfig, primary_device: str, device_ids=None):
    path = Path(pretrained_path)

    if path.is_dir():
        logger.info("Loading pipeline from pretrained directory: %s", path)
        pipeline = StegoGANPipeline.from_pretrained(pretrained_path, torch_dtype=torch.bfloat16)
    else:
        logger.info("Loading generator from legacy checkpoint: %s", path)
        netG = create_generator_a(
            input_nc=cfg.source_channels,
            output_nc=cfg.target_channels,
            ngf=cfg.ngf,
            n_blocks=cfg.n_blocks,
            norm_type=cfg.normG,
            use_dropout=not cfg.no_dropout,
            resnet_layer=cfg.resnet_layer,
            use_fusion_block=cfg.use_fusion_block,
            init_type=cfg.init_type,
            init_gain=cfg.init_gain,
        )
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(str(path))
        else:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        netG.load_state_dict(ckpt)
        pipeline = StegoGANPipeline(generator=netG)

    pipeline = pipeline.to(primary_device, dtype=torch.bfloat16)
    if device_ids is not None and len(device_ids) > 1:
        pipeline.generator = torch.nn.DataParallel(pipeline.generator, device_ids=device_ids)
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
        device_ids = [int(d.split(":")[1]) if ":" in d else i for i, d in enumerate(args.device)]
    pipeline = _load_pipeline(args.pretrained_model_name_or_path, cfg, args.primary_device, device_ids)

    infer_resolution = args.resolution if args.resolution is not None else cfg.resolution
    dataset = MavicTStegoGANDataset(
        task=args.task,
        split=args.split,
        resolution=infer_resolution,
        source_channels=cfg.source_channels,
        target_channels=cfg.target_channels,
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
            logger.info("All %d outputs already exist. Nothing to do.", n_total)
            return
        if len(indices_to_process) < n_total:
            dataset = Subset(dataset, indices_to_process)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info(f"Generating samples for {args.task} ({args.split}), {len(dataset)} inputs …")
    all_samples = []
    sample_idx = 0
    total_start = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Sampling"):
            source = batch[0].to(args.primary_device) * 2 - 1
            result = pipeline(source_image=source, output_type="pt")
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

    total_elapsed = time.perf_counter() - total_start
    runtime_per_image = total_elapsed / len(dataset) if len(dataset) > 0 else 0.0
    use_gpu = args.primary_device.startswith("cuda")

    checkpoint_config = load_checkpoint_config(args.pretrained_model_name_or_path)
    extra_sampling = [
        "StegoGAN: single-step deterministic generator forward (no diffusion steps)",
    ]
    detailed_description = build_detailed_description(
        model_name="StegoGAN",
        model_description="StegoGAN - steganographic GAN for non-bijective image-to-image translation. CVPR 2024.",
        checkpoint_path=args.pretrained_model_name_or_path,
        args=args,
        cfg=cfg,
        checkpoint_config=checkpoint_config,
        runtime_per_image=runtime_per_image,
        extra_sampling_lines=extra_sampling,
    )
    write_readme(
        output_dir,
        runtime_per_image=runtime_per_image,
        use_gpu=use_gpu,
        extra_data=args.extra_data,
        description=detailed_description,
    )

    all_samples = np.concatenate(all_samples, axis=0)
    if args.save_npz:
        np.savez(output_dir / f"samples_{len(all_samples)}.npz", arr_0=all_samples)

    logger.info(
        f"Sampling complete – {sample_idx} images saved to {output_dir} "
        f"(runtime: {total_elapsed:.1f}s total, {runtime_per_image:.2f}s/image)"
    )


if __name__ == "__main__":
    main()
