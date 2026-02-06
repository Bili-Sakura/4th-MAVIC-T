"""
Prepare refined training datasets for MAVIC-T 4 image-to-image tasks.

Outputs to: datasets/BiliSakura/MACIV-T-2025-Structure-Refined/
  sar2eo/train/{input,target}/    256x256 PNG symlinks
  rgb2ir/train/{input,target}/    1024x1024 TIFF (input=RGB 3-band, target=IR 1-band)
  sar2ir/train/{input,target}/    1024x1024 TIFF (input=SAR 1-band, target=IR 1-band)
  sar2rgb/train/{input,target}/   1024x1024 TIFF (input=SAR 1-band, target=RGB 3-band)
  manifests/refined_manifest.csv

Rules:
- sar2eo:  256x256 paired PNGs from EO/SAR dataset (symlinked, no resize).
- rgb2ir:  from city tiles, pair *_rgb.tiff with *_ir.tiff; resize both to 1024x1024.
- sar2ir:  from city tiles, pair each SAR scene with the tile's IR mosaic; resize to 1024.
- sar2rgb: from city tiles, pair each SAR scene with the tile's RGB mosaic; resize to 1024.
- Full set, no split.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin


SRC_ROOT = Path("/mnt/data/projects/4th-MAVIC-T/datasets/BiliSakura/MAVIC-T-2025")
DST_ROOT = Path("/mnt/data/projects/4th-MAVIC-T/datasets/BiliSakura/MACIV-T-2025-Structure-Refined")

CITIES = [
    "Train_Data_Bingham_SAR_IR_RGB",
    "Train_Data_Centerfield_SAR_IR_RGB",
    "Train_Data_Manhattan_SAR_IR_RGB",
    "Train_Data_UC_Davis_SAR_IR_RGB",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_hidden(name: str) -> bool:
    return name.startswith(".") or name.startswith("._")


def resample_to_square(src_path: Path, size: int) -> Tuple[np.ndarray, dict]:
    """Read image and resample to size x size, dropping georeference."""
    with rasterio.open(src_path) as src:
        data = src.read(
            out_shape=(src.count, size, size),
            resampling=Resampling.bilinear,
        )
        profile = src.profile.copy()
    # Normalize uint16 → uint8 if needed
    if profile["dtype"] == "uint16":
        data = (data.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
        profile["dtype"] = "uint8"
    profile.update(
        height=size,
        width=size,
        transform=from_origin(0, size, 1, 1),
        crs=None,
        driver="GTiff",
        tiled=False,
    )
    return data, profile


def write_tiff(out_path: Path, data: np.ndarray, profile: dict) -> None:
    ensure_dir(out_path.parent)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def symlink_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Pairing helpers for city tile data
# ---------------------------------------------------------------------------

def collect_rgb_ir_pairs(tile_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Return {base_name: {"rgb": Path, "ir": Path}} for files in a tile."""
    pairs: Dict[str, Dict[str, Path]] = {}
    for f in tile_dir.iterdir():
        if not f.is_file() or is_hidden(f.name):
            continue
        low = f.name.lower()
        stem = f.stem  # without extension
        if low.endswith("_rgb.tiff") or low.endswith("_rgb.tif"):
            base = stem.rsplit("_rgb", 1)[0]
            pairs.setdefault(base, {})["rgb"] = f
        elif low.endswith("_ir.tiff") or low.endswith("_ir.tif"):
            base = stem.rsplit("_ir", 1)[0]
            pairs.setdefault(base, {})["ir"] = f
        elif low.startswith("mosiac_rgb"):
            pairs.setdefault("mosiac", {})["rgb"] = f
        elif low.startswith("mosiac_ir"):
            pairs.setdefault("mosiac", {})["ir"] = f
    return pairs


def pick_mosaic(tile_dir: Path, kind: str) -> Path | None:
    """Pick the mosaic target file (IR or RGB) for SAR pairing."""
    key = kind  # "rgb" or "ir"
    for f in tile_dir.iterdir():
        if f.is_file() and not is_hidden(f.name):
            if f.name.lower().startswith(f"mosiac_{key}"):
                return f
    # fallback: first *_{key}.tiff
    for f in tile_dir.iterdir():
        if f.is_file() and not is_hidden(f.name):
            if f.name.lower().endswith(f"_{key}.tiff") or f.name.lower().endswith(f"_{key}.tif"):
                return f
    return None


def sar_files(tile_dir: Path) -> List[Path]:
    out = []
    for f in tile_dir.iterdir():
        if f.is_file() and not is_hidden(f.name):
            low = f.name.lower()
            if "_rgb" in low or "_ir" in low or low.startswith("mosiac"):
                continue
            if low.endswith((".tif", ".tiff")):
                out.append(f)
    return sorted(out)


# ---------------------------------------------------------------------------
# Task processors
# ---------------------------------------------------------------------------

def process_sar2eo(writer: csv.writer) -> int:
    """Symlink EO/SAR paired 256x256 PNGs."""
    dst_input = DST_ROOT / "sar2eo" / "train" / "input"
    dst_target = DST_ROOT / "sar2eo" / "train" / "target"
    count = 0
    eo_sar_root = SRC_ROOT / "Train_Data_and_Validation_EO_SAR"
    for split_name in ("train", "validation"):
        sar_dir = eo_sar_root / "SAR" / split_name
        eo_dir = eo_sar_root / "EO" / split_name
        if not sar_dir.is_dir() or not eo_dir.is_dir():
            print(f"  [WARN] Missing dir: {sar_dir} or {eo_dir}")
            continue
        fnames = sorted(f for f in os.listdir(sar_dir) if f.endswith(".png") and not is_hidden(f))
        for fname in fnames:
            sar_path = sar_dir / fname
            eo_path = eo_dir / fname
            if not eo_path.exists():
                continue
            symlink_or_copy(sar_path, dst_input / fname)
            symlink_or_copy(eo_path, dst_target / fname)
            writer.writerow([
                "sar2eo", "train",
                f"sar2eo/train/input/{fname}",
                f"sar2eo/train/target/{fname}",
                split_name, "eo_sar",
            ])
            count += 1
        print(f"  sar2eo/{split_name}: {len(fnames)} files linked")
    return count


def process_rgb2ir(writer: csv.writer) -> int:
    """Pair RGB→IR from city tiles, resample to 1024x1024."""
    dst_input = DST_ROOT / "rgb2ir" / "train" / "input"
    dst_target = DST_ROOT / "rgb2ir" / "train" / "target"
    count = 0
    for city in CITIES:
        city_dir = SRC_ROOT / city
        tiles = sorted(d for d in city_dir.iterdir() if d.is_dir() and not is_hidden(d.name))
        city_count = 0
        for tile in tiles:
            pairs = collect_rgb_ir_pairs(tile)
            for base, paths in pairs.items():
                if "rgb" not in paths or "ir" not in paths:
                    continue
                data_rgb, prof_rgb = resample_to_square(paths["rgb"], 1024)
                data_ir, prof_ir = resample_to_square(paths["ir"], 1024)
                # Filename encodes city + tile + base + modality suffix
                in_name = f"{city}__{tile.name}__{base}__rgb.tif"
                tg_name = f"{city}__{tile.name}__{base}__ir.tif"
                write_tiff(dst_input / in_name, data_rgb, prof_rgb)
                write_tiff(dst_target / tg_name, data_ir, prof_ir)
                writer.writerow([
                    "rgb2ir", "train",
                    f"rgb2ir/train/input/{in_name}",
                    f"rgb2ir/train/target/{tg_name}",
                    tile.name, city,
                ])
                city_count += 1
                count += 1
        print(f"  rgb2ir/{city}: {city_count} pairs")
    return count


def process_sar_to_optical(kind: str, writer: csv.writer) -> int:
    """Pair SAR→IR or SAR→RGB from city tiles, resample to 1024x1024."""
    assert kind in ("ir", "rgb")
    task = f"sar2{kind}"
    dst_input = DST_ROOT / task / "train" / "input"
    dst_target = DST_ROOT / task / "train" / "target"
    count = 0
    for city in CITIES:
        city_dir = SRC_ROOT / city
        tiles = sorted(d for d in city_dir.iterdir() if d.is_dir() and not is_hidden(d.name))
        city_count = 0
        for tile in tiles:
            target_path = pick_mosaic(tile, kind)
            if target_path is None:
                continue
            target_data, target_profile = resample_to_square(target_path, 1024)
            for sar_f in sar_files(tile):
                sar_data, sar_profile = resample_to_square(sar_f, 1024)
                in_name = f"{city}__{tile.name}__{sar_f.stem}__sar.tif"
                tg_name = f"{city}__{tile.name}__{target_path.stem}__{kind}.tif"
                write_tiff(dst_input / in_name, sar_data, sar_profile)
                write_tiff(dst_target / tg_name, target_data, target_profile)
                writer.writerow([
                    task, "train",
                    f"{task}/train/input/{in_name}",
                    f"{task}/train/target/{tg_name}",
                    tile.name, city,
                ])
                city_count += 1
                count += 1
        print(f"  {task}/{city}: {city_count} pairs")
    return count


def main() -> None:
    t0 = time.time()
    ensure_dir(DST_ROOT / "manifests")
    manifest_path = DST_ROOT / "manifests" / "refined_manifest.csv"

    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "split", "input", "target", "tile", "source_city"])

        print("[1/4] Processing sar2eo (256x256 symlinks)...")
        n = process_sar2eo(writer)
        print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

        print("[2/4] Processing rgb2ir (1024x1024 TIFF)...")
        n = process_rgb2ir(writer)
        print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

        print("[3/4] Processing sar2ir (1024x1024 TIFF)...")
        n = process_sar_to_optical("ir", writer)
        print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

        print("[4/4] Processing sar2rgb (1024x1024 TIFF)...")
        n = process_sar_to_optical("rgb", writer)
        print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
