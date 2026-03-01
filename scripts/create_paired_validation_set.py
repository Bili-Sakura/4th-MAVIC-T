#!/usr/bin/env python3
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Create paired validation manifests using MaRS embedding-based selection.

Selection policy:
* ``sar2eo`` uses ``16 * N`` pairs.
* ``sar2ir`` uses ``N`` pairs.
* ``sar2rgb`` uses ``N`` pairs.
* ``rgb2ir`` uses ``N`` pairs.

Encoder policy:
* ``sar2eo``, ``sar2ir``, ``sar2rgb`` -> MaRS-Base-SAR (encode input images).
* ``rgb2ir`` -> MaRS-Base-RGB (encode input images).

The selector is deterministic and balances representativeness + diversity:
1) start from the sample nearest to embedding centroid,
2) greedily add samples maximizing ``min_dist_to_selected - lambda * dist_to_centroid``.

Writes manifest files under ``dataset_root/manifests/paired_val_<task>.txt``.
Each line: ``input_path<TAB>target_path`` (absolute paths for portability).

Usage:
    python scripts/create_paired_validation_set.py
    python scripts/create_paired_validation_set.py --dataset_root ./datasets/BiliSakura/MACIV-T-2025-Structure-Refined
    python scripts/create_paired_validation_set.py --n 4 --batch_size 8
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

# Standalone constants to avoid heavy imports (torch, datasets, etc.)
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFINED_ROOT = _REPO_ROOT / "datasets/BiliSakura/MACIV-T-2025-Structure-Refined"
REFINED_MANIFEST_NAMES = ("refined_manifest.csv", "refined_manifest_crop_aug.csv")
TASKS = ("sar2eo", "rgb2ir", "sar2ir", "sar2rgb", "rgb2ir_crop_aug", "sar2ir_crop_aug", "sar2rgb_crop_aug")
TASK_TARGET_COUNTS_MULTIPLIER = {
    "sar2eo": 16,
    "sar2ir": 1,
    "sar2rgb": 1,
    "rgb2ir": 1,
}
TASK_ENCODER_KIND = {
    "sar2eo": "sar",
    "sar2ir": "sar",
    "sar2rgb": "sar",
    "rgb2ir": "rgb",
}


def _resolve_dataset_root(root: Path) -> Path:
    if root.exists():
        return root
    root_str = str(root)
    if root_str.startswith("/mnt/data/"):
        alt = Path("/data") / Path(root_str).relative_to("/mnt/data")
        if alt.exists():
            return alt
    if root_str.startswith("/data/"):
        alt = Path("/mnt/data") / Path(root_str).relative_to("/data")
        if alt.exists():
            return alt
    return root


# Base tasks only (exclude _crop_aug for deduplication; we sample from base + crop_aug manifests)
BASE_TASKS = ("sar2eo", "rgb2ir", "sar2ir", "sar2rgb")
DEFAULT_MARS_SAR_PATH = _REPO_ROOT / "models/BiliSakura/MaRS-Base-SAR"
DEFAULT_MARS_RGB_PATH = _REPO_ROOT / "models/BiliSakura/MaRS-Base-RGB"


class MaRSEmbedder:
    """Frozen MaRS feature extractor for ranking validation candidates."""

    def __init__(
        self,
        model_path: Path,
        kind: str,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if kind not in {"sar", "rgb"}:
            raise ValueError(f"Unknown embedder kind: {kind}")
        self.kind = kind
        self.model_path = model_path
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        processor_kwargs = {"do_convert_rgb": False} if kind == "sar" else {}
        self.processor = AutoImageProcessor.from_pretrained(str(model_path), **processor_kwargs)
        self.model = AutoModel.from_pretrained(str(model_path), trust_remote_code=False)
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.model.eval()

    def _load_pil(self, path: str) -> Image.Image:
        img = Image.open(path)
        if self.kind == "sar":
            if img.mode != "L":
                img = img.convert("L")
        else:
            if img.mode != "RGB":
                img = img.convert("RGB")
        return img

    def encode_paths(self, image_paths: Iterable[str], batch_size: int = 8) -> np.ndarray:
        torch = self.torch
        paths = list(image_paths)
        if not paths:
            return np.zeros((0, 1), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in tqdm(
                range(0, len(paths), batch_size),
                desc="Encoding",
                unit="batch",
                total=(len(paths) + batch_size - 1) // batch_size,
            ):
                batch_paths = paths[start:start + batch_size]
                batch_images = [self._load_pil(p) for p in batch_paths]
                pixel_values = self.processor(images=batch_images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                outputs = self.model(pixel_values)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    emb = outputs.pooler_output
                else:
                    hidden = outputs.last_hidden_state
                    if hidden.ndim == 4:
                        emb = hidden.mean(dim=[2, 3])
                    else:
                        emb = hidden.mean(dim=1)
                emb = torch.nn.functional.normalize(emb, dim=-1)
                all_embeddings.append(emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)


def _pair_count_for_task(task: str, n: int) -> int:
    mul = TASK_TARGET_COUNTS_MULTIPLIER.get(task)
    if mul is None:
        raise KeyError(f"Missing task multiplier for {task}")
    return mul * n


def _select_by_hybrid_kcenter(
    embeddings: np.ndarray,
    k: int,
    lambda_center: float,
) -> list[int]:
    """Select indices with centroid-anchored farthest-point strategy."""
    n = embeddings.shape[0]
    if n == 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))

    centroid = embeddings.mean(axis=0, keepdims=True)
    centroid_dist = np.linalg.norm(embeddings - centroid, axis=1)

    # Start from the most representative sample (nearest to centroid).
    selected: list[int] = [int(np.argmin(centroid_dist))]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[selected[0]] = True

    min_dist = np.linalg.norm(embeddings - embeddings[selected[0]], axis=1)
    min_dist[selected[0]] = -math.inf

    while len(selected) < k:
        # Balance coverage/diversity and avoid extreme outliers.
        score = min_dist - (lambda_center * centroid_dist)
        score[selected_mask] = -math.inf
        nxt = int(np.argmax(score))
        selected.append(nxt)
        selected_mask[nxt] = True
        min_dist = np.minimum(min_dist, np.linalg.norm(embeddings - embeddings[nxt], axis=1))
        min_dist[nxt] = -math.inf

    return selected


def load_train_pairs(
    refined_root: Path,
    task: str,
    exclude_paths: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Load all train (input, target) path pairs for a task from refined manifests."""
    pairs: list[tuple[str, str]] = []
    exclude = exclude_paths or set()

    # Include both base task and crop_aug variant to match full train set
    task_variants = [task]
    if f"{task}_crop_aug" in TASKS:
        task_variants.append(f"{task}_crop_aug")

    manifest_paths = [
        refined_root / "manifests" / name
        for name in REFINED_MANIFEST_NAMES
        if (refined_root / "manifests" / name).is_file()
    ]
    if not manifest_paths:
        raise FileNotFoundError(
            f"No manifest files found under {refined_root}/manifests "
            f"(expected one of: {', '.join(REFINED_MANIFEST_NAMES)})"
        )

    for manifest_path in manifest_paths:
        with manifest_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if (row.get("split") or "").lower() != "train":
                    continue
                row_task = (row.get("task") or "").lower()
                if row_task not in task_variants:
                    continue

                input_path = str((refined_root / row["input"]).resolve())
                target_path = str((refined_root / row["target"]).resolve())

                if input_path in exclude or target_path in exclude:
                    continue
                pairs.append((input_path, target_path))

    return pairs


def load_exclude_set(exclude_file: Path | None) -> set[str]:
    """Load absolute paths to exclude (e.g. bad_samples.txt)."""
    if not exclude_file or not exclude_file.is_file():
        return set()
    out: set[str] = set()
    with exclude_file.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.add(str(Path(line).resolve()))
    return out


def create_paired_validation_set(
    dataset_root: Path,
    n: int = 4,
    exclude_file: Path | None = None,
    mars_sar_path: Path = DEFAULT_MARS_SAR_PATH,
    mars_rgb_path: Path = DEFAULT_MARS_RGB_PATH,
    batch_size: int = 8,
    lambda_center: float = 0.35,
    device: str | None = None,
) -> dict[str, Path]:
    """Select paired validation sets via MaRS embeddings and save manifests."""
    refined_root = _resolve_dataset_root(dataset_root)
    manifests_dir = refined_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    exclude = load_exclude_set(exclude_file) if exclude_file else set()
    created: dict[str, Path] = {}
    embedders: dict[str, MaRSEmbedder] = {}

    if not mars_sar_path.exists():
        raise FileNotFoundError(f"MaRS-SAR model path not found: {mars_sar_path}")
    if not mars_rgb_path.exists():
        raise FileNotFoundError(f"MaRS-RGB model path not found: {mars_rgb_path}")

    for task in tqdm(BASE_TASKS, desc="Tasks", unit="task"):
        pairs = load_train_pairs(refined_root, task, exclude)
        n_pairs = _pair_count_for_task(task, n)

        if len(pairs) < n_pairs:
            chosen = pairs
            print(f"  Warning: {task} has only {len(pairs)} pairs, using all")
        else:
            encoder_kind = TASK_ENCODER_KIND[task]
            embedder = embedders.get(encoder_kind)
            if embedder is None:
                model_path = mars_sar_path if encoder_kind == "sar" else mars_rgb_path
                embedder = MaRSEmbedder(model_path=model_path, kind=encoder_kind, device=device)
                embedders[encoder_kind] = embedder

            input_paths = [inp for inp, _ in pairs]
            embeddings = embedder.encode_paths(input_paths, batch_size=batch_size)
            selected_idx = _select_by_hybrid_kcenter(
                embeddings=embeddings,
                k=n_pairs,
                lambda_center=lambda_center,
            )
            chosen = [pairs[i] for i in selected_idx]

        out_path = manifests_dir / f"paired_val_{task}.txt"
        with out_path.open("w") as fh:
            for inp, tgt in chosen:
                fh.write(f"{inp}\t{tgt}\n")

        created[task] = out_path
        print(f"  {task}: wrote {len(chosen)} pairs to {out_path}")

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Create paired validation set via MaRS embedding selection"
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_REFINED_ROOT,
        help="Path to refined dataset root (contains manifests/)",
    )
    parser.add_argument("--n", type=int, default=4, help="Base count N (default: 4)")
    parser.add_argument(
        "--exclude_file",
        type=Path,
        default=None,
        help="Optional path to bad_samples.txt to exclude from sampling",
    )
    parser.add_argument(
        "--mars_sar_path",
        type=Path,
        default=DEFAULT_MARS_SAR_PATH,
        help="Path to MaRS-Base-SAR model directory",
    )
    parser.add_argument(
        "--mars_rgb_path",
        type=Path,
        default=DEFAULT_MARS_RGB_PATH,
        help="Path to MaRS-Base-RGB model directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Embedding batch size (default: 8)",
    )
    parser.add_argument(
        "--lambda_center",
        type=float,
        default=0.35,
        help="Outlier penalty in selector (higher -> more representative)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device for embedding extraction (e.g. "cuda", "cpu"). Default: auto',
    )
    args = parser.parse_args()

    exclude = args.exclude_file
    if exclude is None:
        default_exclude = args.dataset_root / "manifests" / "bad_samples.txt"
        if default_exclude.exists():
            exclude = default_exclude

    counts_text = ", ".join(
        f"{task}={_pair_count_for_task(task, args.n)}" for task in BASE_TASKS
    )
    print(f"Creating paired validation set with N={args.n} ({counts_text})...")
    create_paired_validation_set(
        dataset_root=args.dataset_root,
        n=args.n,
        exclude_file=exclude,
        mars_sar_path=args.mars_sar_path,
        mars_rgb_path=args.mars_rgb_path,
        batch_size=args.batch_size,
        lambda_center=args.lambda_center,
        device=args.device,
    )
    print("Done.")


if __name__ == "__main__":
    main()
