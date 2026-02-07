#!/usr/bin/env python3
"""Quick VAE reconstruction sanity check for SAR / IR / EO imagery.

The script loads a diffusers ``AutoencoderKL`` from ``./models`` (defaults to
``./models/BiliSakura/VAEs``), expands single-channel inputs to three channels
as expected by the VAE, reconstructs them, averages the three output channels
back to one channel for SAR/IR inputs, and reports MAE / PSNR / SSIM.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

DEFAULT_VAE_PATH = "./models/BiliSakura/VAEs"
DEFAULT_EXTS = (".png", ".tif", ".tiff")


def _normalise_image_array(arr: np.ndarray) -> np.ndarray:
    """Normalise a H×W×C image array to [0, 1] float32."""
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        if arr.dtype == np.float32 and arr.max() > 255.0:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0
    return arr


def load_image_to_three_channels(
    path: Path, resolution: int, device: torch.device
) -> Tuple[torch.Tensor, int]:
    """Load an image, resize, and expand to three channels for the VAE.

    Returns
    -------
    tensor : torch.Tensor
        Tensor of shape (1, 3, H, W) in [0, 1] on the requested device.
    orig_channels : int
        Number of channels in the source image before expansion.
    """
    img = Image.open(path)
    if resolution:
        img = img.resize((resolution, resolution), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = _normalise_image_array(arr)

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    orig_channels = arr.shape[2]

    if orig_channels < 3:
        arr = np.repeat(arr, 3, axis=2)
    elif orig_channels > 3:
        arr = arr[:, :, :3]

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, orig_channels


def reduce_reconstruction_channels(recon: torch.Tensor, orig_channels: int) -> torch.Tensor:
    """Convert a 3-channel reconstruction back to the original channel layout."""
    if orig_channels == 1:
        return recon.mean(dim=1, keepdim=True)
    return recon[:, :3]


def compute_reconstruction_metrics(
    recon: torch.Tensor, target: torch.Tensor
) -> Dict[str, float]:
    """Compute MAE, PSNR, and SSIM for a single reconstructed sample."""
    recon = recon.clamp(0, 1)
    target = target.clamp(0, 1)
    _, _, h, w = recon.shape
    kernel = min(h, w, 11)
    if kernel % 2 == 0:
        kernel = max(1, kernel - 1)
    use_gaussian = min(h, w) >= 11

    mae = torch.mean(torch.abs(recon - target)).item()
    psnr = peak_signal_noise_ratio(recon, target, data_range=1.0).item()
    ssim = structural_similarity_index_measure(
        recon, target, data_range=1.0, kernel_size=kernel, gaussian_kernel=use_gaussian
    ).item()
    return {"mae": float(mae), "psnr": float(psnr), "ssim": float(ssim)}


def _iter_image_paths(root: Path, exts: Sequence[str]) -> List[Path]:
    """Return a sorted list of image paths under ``root``."""
    wanted = {ext.lower() for ext in exts}
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in wanted
    )


def _save_reconstruction(
    path: Path, recon: torch.Tensor, orig_channels: int, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    recon = recon.squeeze(0).detach().cpu()
    if orig_channels == 1:
        recon_np = recon[0].clamp(0, 1).numpy()
        img = Image.fromarray((recon_np * 255).astype(np.uint8), mode="L")
    else:
        recon_np = recon.permute(1, 2, 0).clamp(0, 1).numpy()
        img = Image.fromarray((recon_np * 255).astype(np.uint8))
    img.save(out_dir / path.name)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vae_path = Path(args.vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE path not found: {vae_path}")

    vae = AutoencoderKL.from_pretrained(str(vae_path)).to(device)
    vae.eval()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))

    image_paths = _iter_image_paths(Path(args.input_dir), args.extensions)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(
            f"No images with extensions {args.extensions} found under {args.input_dir}"
        )

    results: List[Dict[str, float]] = []
    for path in image_paths:
        input_tensor, orig_channels = load_image_to_three_channels(
            path, args.resolution, device
        )
        vae_input = input_tensor * 2 - 1

        with torch.no_grad():
            posterior = vae.encode(vae_input).latent_dist
            latents = posterior.mean * scaling_factor
            decoded = vae.decode(latents / scaling_factor).sample

        recon = (decoded.clamp(-1, 1) + 1) / 2
        recon_for_metrics = reduce_reconstruction_channels(recon, orig_channels)
        target_for_metrics = (
            input_tensor[:, :1] if orig_channels == 1 else input_tensor[:, :3]
        )

        metrics = compute_reconstruction_metrics(recon_for_metrics, target_for_metrics)
        results.append(metrics)

        print(
            f"[{path.name}] MAE: {metrics['mae']:.4f} | "
            f"PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}"
        )

        if args.output_dir:
            _save_reconstruction(path, recon_for_metrics, orig_channels, Path(args.output_dir))

    # Summary
    mean_mae = float(np.mean([m["mae"] for m in results]))
    mean_psnr = float(np.mean([m["psnr"] for m in results]))
    mean_ssim = float(np.mean([m["ssim"] for m in results]))
    print(
        f"\nAveraged over {len(results)} image(s): "
        f"MAE={mean_mae:.4f}, PSNR={mean_psnr:.2f}, SSIM={mean_ssim:.4f}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing SAR/IR/EO images to reconstruct.",
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        default=DEFAULT_VAE_PATH,
        help="Path to the diffusers AutoencoderKL (defaults to ./models/BiliSakura/VAEs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device string (default: "cuda" if available else "cpu").',
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resize resolution fed to the VAE (square).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTS),
        help="Image extensions to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to write reconstructed outputs.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optionally limit the number of images processed.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(args=argv)
    run(args)


if __name__ == "__main__":
    main()
