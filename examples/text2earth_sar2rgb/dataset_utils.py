"""Dataset utilities for Text2Earth SAR2RGB training.

Supports MAVIC-T paired manifest format (input_path\\ttarget_path) and
HuggingFace datasets / imagefolder.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Project root for resolving relative paths
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_manifest_path(raw: str, manifest_path: Path) -> str:
    """Resolve a path from a manifest (may be relative to dataset root)."""
    p = Path(raw.strip())
    if p.is_absolute():
        return str(p.resolve())
    dataset_root = manifest_path.resolve().parent.parent
    return str((dataset_root / raw).resolve())


def load_paired_manifest(
    manifest_path: str | Path,
    *,
    refined_root: Optional[str | Path] = None,
) -> list[dict]:
    """Load (input_path, target_path) pairs from a paired manifest.

    Manifest format: one line per pair, input_path\\ttarget_path (tab-separated).
    Paths may be absolute or relative to the dataset root (manifest's parent.parent).
    """
    path = Path(manifest_path)
    if not path.is_file():
        # Try project root
        candidate = _PROJECT_ROOT / path
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            inp, tgt = parts[0].strip(), parts[1].strip()
            if inp and tgt:
                inp_resolved = _resolve_manifest_path(inp, path)
                tgt_resolved = _resolve_manifest_path(tgt, path)
                records.append({"input_path": inp_resolved, "target_path": tgt_resolved})
    return records


def load_sar2rgb_train_records(
    refined_root: Optional[str | Path] = None,
    sar2rgb_sup_manifest: Optional[str] = None,
    exclude_file: Optional[str] = None,
    paired_val_manifest: Optional[str] = None,
    use_augmented: bool = True,
) -> list[dict]:
    """Load SAR2RGB training records from MAVIC-T dataset.

    Reads refined_manifest.csv directly to avoid HuggingFace datasets import
    (project has local datasets/ folder that shadows the package).
    """
    import csv

    refined_root = Path(refined_root or _PROJECT_ROOT / "datasets/BiliSakura/MACIV-T-2025-Structure-Refined")
    records = []

    # refined_manifest.csv has task=sar2rgb; refined_manifest_crop_aug.csv has task=sar2rgb_crop_aug
    manifest_tasks = [("refined_manifest.csv", "sar2rgb")]
    if use_augmented:
        manifest_tasks.append(("refined_manifest_crop_aug.csv", "sar2rgb_crop_aug"))

    for manifest_name, task_filter in manifest_tasks:
        manifest_path = refined_root / "manifests" / manifest_name
        if not manifest_path.is_file():
            continue
        with manifest_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if (row.get("split") or "").lower() != "train":
                    continue
                if (row.get("task") or "").lower() != task_filter:
                    continue
                input_path = str((refined_root / row["input"]).resolve())
                target_path = str((refined_root / row["target"]).resolve())
                records.append({"input_path": input_path, "target_path": target_path})

    # Add sar2rgb_sup pairs
    if sar2rgb_sup_manifest:
        try:
            sup = load_paired_manifest(sar2rgb_sup_manifest)
            records.extend(sup)
        except FileNotFoundError:
            pass

    # Exclude set
    exclude = set()
    if exclude_file:
        ep = Path(exclude_file)
        if not ep.is_absolute():
            ep = _PROJECT_ROOT / ep
        if ep.is_file():
            with ep.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        exclude.add(str(Path(line).resolve()))
    if paired_val_manifest:
        try:
            val_records = load_paired_manifest(paired_val_manifest)
            for r in val_records:
                exclude.add(r["input_path"])
                exclude.add(r["target_path"])
        except FileNotFoundError:
            pass

    if exclude:
        records = [
            r for r in records
            if r["input_path"] not in exclude and r["target_path"] not in exclude
        ]
    return records


def _load_image(path: str, channels: int, resolution: int, crop_pos=None) -> np.ndarray:
    """Load image as (H, W, C) float32 in [0, 1].
    crop_pos: (x, y) for random/center crop. Resolution is crop size only, never resize."""
    with Image.open(path) as img:
        if crop_pos is not None:
            x, y = crop_pos
            if img.width >= x + resolution and img.height >= y + resolution:
                img = img.crop((x, y, x + resolution, y + resolution))
            else:
                raise ValueError(
                    f"Image {path} size ({img.width}x{img.height}) too small for "
                    f"resolution {resolution} crop at ({x},{y})."
                )
        else:
            # Center crop when no random crop (e.g. validation)
            if img.width >= resolution and img.height >= resolution:
                x = (img.width - resolution) // 2
                y = (img.height - resolution) // 2
                img = img.crop((x, y, x + resolution, y + resolution))
            else:
                raise ValueError(
                    f"Image {path} size ({img.width}x{img.height}) smaller than "
                    f"resolution {resolution}; crop only, no resize."
                )
        arr = np.array(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0 if arr.max() <= 255 else arr / 65535.0
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    c = arr.shape[2]
    if c < channels:
        arr = np.repeat(arr, channels // c + 1, axis=2)[:, :, :channels]
    elif c > channels:
        arr = arr[:, :, :channels]
    return arr


class MavicTSAR2RGBDataset(Dataset):
    """PyTorch Dataset for SAR2RGB pairs from MAVIC-T manifest format.

    Returns dict with:
    - pixel_values: RGB target (3, H, W) in [-1, 1] for VAE
    - conditioning_pixel_values: SAR as 3ch grayscale (3, H, W) for ControlNet
    - sar_pixel_values: SAR (1, H, W) in [-1, 1] for InstructPix2Pix latent concat
    - input_ids: tokenized caption
    """

    def __init__(
        self,
        records: list[dict],
        resolution: int = 512,
        caption: str = "18_GOOGLE_LEVEL_ a satellite optical image",
        use_random_crop: bool = True,
        use_horizontal_flip: bool = False,
        use_vertical_flip: bool = False,
        tokenizer=None,
    ):
        self.records = records
        self.resolution = resolution
        self.caption = caption
        self.use_random_crop = use_random_crop
        self.use_horizontal_flip = use_horizontal_flip
        self.use_vertical_flip = use_vertical_flip
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def _sample_crop_pos(self, idx: int) -> Optional[tuple]:
        if not self.use_random_crop:
            return None
        rec = self.records[idx]
        try:
            with Image.open(rec["input_path"]) as img:
                w, h = img.size
        except Exception:
            return None
        if w < self.resolution or h < self.resolution:
            return None
        x = random.randint(0, w - self.resolution)
        y = random.randint(0, h - self.resolution)
        return (x, y)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        crop_pos = self._sample_crop_pos(idx)

        # When random crop requested but image too small, skip to another sample
        if self.use_random_crop and crop_pos is None:
            if len(self.records) > 1:
                return self.__getitem__(random.randint(0, len(self.records) - 1))
            raise ValueError(
                f"All images smaller than resolution {self.resolution}; "
                "resolution is for random crop only, not resize."
            )

        # Load SAR (1ch) and RGB (3ch)
        sar_arr = _load_image(rec["input_path"], 1, self.resolution, crop_pos)
        rgb_arr = _load_image(rec["target_path"], 3, self.resolution, crop_pos)

        # Augmentations (same for both)
        if self.use_horizontal_flip and random.random() > 0.5:
            sar_arr = np.fliplr(sar_arr).copy()
            rgb_arr = np.fliplr(rgb_arr).copy()
        if self.use_vertical_flip and random.random() > 0.5:
            sar_arr = np.flipud(sar_arr).copy()
            rgb_arr = np.flipud(rgb_arr).copy()

        # RGB: (3, H, W) in [-1, 1] for VAE / target
        rgb_tensor = torch.from_numpy(rgb_arr).permute(2, 0, 1).float()
        rgb_tensor = 2.0 * rgb_tensor - 1.0

        # SAR as 3ch grayscale for ControlNet conditioning (no normalize to [0,1] then *2-1)
        sar_3ch = np.repeat(sar_arr, 3, axis=2)  # (H, W, 3)
        conditioning_tensor = torch.from_numpy(sar_3ch).permute(2, 0, 1).float()
        # ControlNet typically expects [0, 1] for conditioning image
        # (diffusers controlnet normalizes internally in some cases; we keep [0,1])

        # SAR (1, H, W) in [-1, 1] for InstructPix2Pix latent concat
        sar_1ch = torch.from_numpy(sar_arr).permute(2, 0, 1).float()  # (1, H, W)
        sar_1ch = 2.0 * sar_1ch - 1.0

        out = {
            "pixel_values": rgb_tensor,
            "conditioning_pixel_values": conditioning_tensor,
            "sar_pixel_values": sar_1ch,
        }
        if self.tokenizer is not None:
            inputs = self.tokenizer(
                self.caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            out["input_ids"] = inputs.input_ids.squeeze(0)
        return out
