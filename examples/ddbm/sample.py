#!/usr/bin/env python
"""Sample (inference) script for a trained DDBM model on any MAVIC-T task.

Outputs are saved under MACIV-T-2025-Submissions/<task>/<model_name>/ following
the official submission format (see official-docs/submission.md). Images are
named by input stem (e.g. 0.png, 1041.png for sar2eo).

Usage (pretrained directory — recommended)::

    nohup python -m examples.ddbm.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./ckpt/exp3/stage1_sar2ir/ddbm/sar2ir/checkpoint-10000 \
        --split test \
        --model_name ddbm \
        --batch_size 8 \
        --deterministic \
        --num_inference_steps 250 \
        --device cuda:1 &

Use ``--batch_size`` to control inference batch size (default 32). Larger values
are faster but require more GPU memory.

Usage (legacy ``.pt`` file)::

    python -m examples.ddbm.sample \
        --task sar2ir \
        --pretrained_model_name_or_path ./outputs/ddbm_sar2ir/model_epoch_100.pt \
        --split test \
        --model_name ddbm

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

from src.schedulers import DDBMScheduler  # noqa: E402
from src.pipelines.ddbm import DDBMPipeline  # noqa: E402

from .config import (  # noqa: E402
    TaskConfig,
    sar2eo_config,
    rgb2ir_config,
    sar2ir_config,
    sar2rgb_config,
)
from .dataset_wrapper import MavicTDDBMDataset  # noqa: E402
from src.models.unet_ddbm import DDBMUNet, create_model  # noqa: E402

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SUBMISSION_ROOT = Path("/data/projects/4th-MAVIC-T/datasets/BiliSakura/MACIV-T-2025-Submissions")

_TASK_CONFIG_MAP = {
    "sar2eo": sar2eo_config,
    "rgb2ir": rgb2ir_config,
    "sar2ir": sar2ir_config,
    "sar2rgb": sar2rgb_config,
}


class _SubsetWithOutputNames(Subset):
    """Subset that delegates get_output_name to the underlying dataset with original indices."""

    def get_output_name(self, idx: int) -> str:
        return self.dataset.get_output_name(self.indices[idx])


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained DDBM model.")
    parser.add_argument("--task", type=str, required=True, choices=list(_TASK_CONFIG_MAP.keys()))
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to a diffusers-style checkpoint directory or a legacy .pt/.safetensors file.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory. Default: {DEFAULT_SUBMISSION_ROOT}/<task>/<model_name>/",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ddbm",
        help="Model name for folder structure (e.g. ddbm). Output: <submission_root>/<task>/<model_name>/",
    )
    parser.add_argument(
        "--submission_root",
        type=str,
        default=str(DEFAULT_SUBMISSION_ROOT),
        help="Root directory for submissions.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference (larger = faster).")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1000,
        help="Number of denoising steps (1000 for best quality).",
    )
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--churn_step_ratio", type=float, default=0.33)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic sampling (churn_step_ratio=0, faster, no stochastic churn). "
        "Note: Can cause large quality degradation vs default; use for speed/reproducibility only.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip samples whose output file already exists (default: True, for resumability).",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Overwrite existing output files.",
    )
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument(
        "--extra_data",
        action="store_true",
        help="Set to 1 in readme: models trained with extra data beyond challenge data.",
    )
    parser.add_argument(
        "--readme_description",
        type=str,
        default="DDBM diffusion-based image translation. PyTorch implementation.",
        help="Additional description for readme.txt",
    )
    args = parser.parse_args()
    if args.deterministic:
        # churn=0: faster (~33% fewer model calls) but reported large quality degradation
        args.churn_step_ratio = 0.0
    return args


def _load_pipeline(pretrained_path: str, cfg: TaskConfig, device: str) -> DDBMPipeline:
    """Load the DDBM pipeline from a pretrained directory or legacy file.

    Parameters
    ----------
    pretrained_path : str
        Either a diffusers-style checkpoint directory (containing ``unet/``
        and ``scheduler/`` sub-folders) or a legacy ``.pt`` / ``.safetensors``
        single-file checkpoint.
    cfg : TaskConfig
        Task-specific configuration (used only for legacy loading).
    device : str
        Target device.
    """
    path = Path(pretrained_path)

    if path.is_dir():
        # ---- diffusers from_pretrained path ----
        logger.info("Loading pipeline from pretrained directory: %s", path)
        pipeline = DDBMPipeline.from_pretrained(pretrained_path)
        # Prefer ema_unet when available (default for best inference quality)
        ema_unet_dir = path / "ema_unet"
        if ema_unet_dir.is_dir():
            logger.info("Loading EMA UNet from %s", ema_unet_dir)
            pipeline.unet = DDBMUNet.from_pretrained(pretrained_path, subfolder="ema_unet")
    else:
        # ---- legacy single-file checkpoint ----
        logger.info("Loading model from legacy checkpoint: %s", path)
        model = create_model(
            image_size=cfg.resolution,
            in_channels=cfg.model_channels,
            num_channels=cfg.num_channels,
            num_res_blocks=cfg.num_res_blocks,
            unet_type=cfg.unet_type,
            attention_resolutions=cfg.attention_resolutions,
            dropout=0.0,
            condition_mode=cfg.condition_mode,
            channel_mult=cfg.channel_mult,
        )
        if str(path).endswith(".safetensors"):
            from safetensors.torch import load_file
            ckpt = load_file(str(path))
        else:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)

        scheduler = DDBMScheduler(
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sigma_data=cfg.sigma_data,
            beta_d=cfg.beta_d,
            beta_min=cfg.beta_min,
            pred_mode=cfg.pred_mode,
        )
        pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

    pipeline = pipeline.to(device)
    return pipeline


def _write_readme(
    output_dir: Path,
    runtime_per_image: float,
    use_gpu: bool,
    extra_data: bool,
    description: str,
) -> None:
    """Write readme.txt per official submission format."""
    readme_path = output_dir / "readme.txt"
    content = f"""runtime per image [s] : {runtime_per_image:.2f}
CPU[1] / GPU[0] : {0 if use_gpu else 1}
Extra Data [1] / No Extra Data [0] : {1 if extra_data else 0}
Other description : {description}
"""
    readme_path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s", readme_path)


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    cfg: TaskConfig = _TASK_CONFIG_MAP[args.task]()
    submission_root = Path(args.submission_root)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = submission_root / args.task / args.model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _load_pipeline(args.pretrained_model_name_or_path, cfg, args.device)

    # Load evaluation data
    dataset = MavicTDDBMDataset(
        task=args.task,
        split=args.split,
        resolution=cfg.resolution,
        model_channels=cfg.model_channels,
        with_target=False,
    )
    if args.skip_existing:
        indices_to_process = [
            i for i in range(len(dataset))
            if not (output_dir / dataset.get_output_name(i)).exists()
        ]
        n_total = len(dataset)
        if not indices_to_process:
            logger.info("All %d outputs already exist in %s. Nothing to do.", n_total, output_dir)
            return
        if len(indices_to_process) < n_total:
            dataset = _SubsetWithOutputNames(dataset, indices_to_process)
            logger.info("Skipping %d existing, processing %d remaining.", n_total - len(indices_to_process), len(indices_to_process))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    churn_info = " (deterministic)" if args.deterministic else ""
    logger.info(
        "Generating samples for %s (%s), %d inputs, batch_size=%d, steps=%d%s → %s",
        args.task,
        args.split,
        len(dataset),
        args.batch_size,
        args.num_inference_steps,
        churn_info,
        output_dir,
    )

    all_samples = []
    total_start = time.perf_counter()
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Sampling"):
            # batch = (zeros_target, source)
            source = batch[1].to(args.device) * 2 - 1  # [0,1] → [-1,1]

            result = pipeline(
                source_image=source,
                num_inference_steps=args.num_inference_steps,
                guidance=args.guidance,
                churn_step_ratio=args.churn_step_ratio,
                output_type="pt",
            )
            images = result.images  # (B, C, H, W) in [-1, 1]
            images_uint8 = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            images_uint8 = images_uint8.permute(0, 2, 3, 1).cpu().numpy()

            for i, img_arr in enumerate(images_uint8):
                out_name = dataset.get_output_name(sample_idx + i)
                if img_arr.shape[2] == 1:
                    img_arr = img_arr.squeeze(2)
                img = Image.fromarray(img_arr)
                img.save(output_dir / out_name)

            sample_idx += len(images_uint8)
            all_samples.append(images_uint8)

    total_elapsed = time.perf_counter() - total_start
    runtime_per_image = total_elapsed / len(dataset) if len(dataset) > 0 else 0.0
    use_gpu = args.device.startswith("cuda")

    # Write readme.txt per official submission format
    _write_readme(
        output_dir,
        runtime_per_image=runtime_per_image,
        use_gpu=use_gpu,
        extra_data=args.extra_data,
        description=args.readme_description,
    )

    all_samples = np.concatenate(all_samples, axis=0)

    if args.save_npz:
        np.savez(output_dir / f"samples_{len(all_samples)}.npz", arr_0=all_samples)
        logger.info("Saved NPZ with %d samples.", len(all_samples))

    logger.info(
        "Sampling complete – %d images saved to %s (runtime: %.1fs total, %.2fs/image)",
        sample_idx,
        output_dir,
        total_elapsed,
        runtime_per_image,
    )


if __name__ == "__main__":
    main()
