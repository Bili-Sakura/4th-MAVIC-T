"""
Prepare refined training datasets for MAVIC-T tasks.

Outputs to: datasets/BiliSakura/MACIV-T-2025-Structure-Refined

Tasks:
- sar2eo  : keep native 256x256 PNG pairs (symlink, no resize).
- rgb2ir  : resize both RGB/IR to 1024x1024.
- sar2ir  : resize SAR->IR to 1024x1024 (pairs SAR scenes to tile IR mosaic).
- sar2rgb : resize SAR->RGB to 1024x1024 (pairs SAR scenes to tile RGB mosaic).

Notes:
- Uses full datasets (no split). If you want splits, add filtering before writing.
- City data pairing rules:
  * For rgb2ir: pairs files in each tile whose names share the same base before _rgb/_ir (plus mosiac_*).
  * For sar2ir/rgb: pairs every SAR scene in a tile with the tile mosaic IR/RGB (mosiac_* preferred;
    if missing, the first *_ir / *_rgb is used).
- Resampling uses bilinear for all bands; georeferencing is dropped.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin


SRC_ROOT = Path("/data/projects/4th-MAVIC-T/datasets/BiliSakura/MAVIC-T-2025")
DST_ROOT = Path("/data/projects/4th-MAVIC-T/datasets/BiliSakura/MACIV-T-2025-Structure-Refined")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_hidden(name: str) -> bool:
    return name.startswith(".") or name.startswith("._")


def resample_to_square(src_path: Path, size: int) -> Tuple[np.ndarray, dict]:
    """Read image and resample to size x size, dropping georeference."""
    with rasterio.open(src_path) as src:
        data = src.read(out_shape=(src.count, size, size), resampling=Resampling.bilinear)
        profile = src.profile
    profile.update(
        height=size,
        width=size,
        transform=from_origin(0, size, 1, 1),  # dummy geotransform
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
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def collect_city_pairs(tile_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Collect RGB/IR per tile by base name before _rgb/_ir, include mosiac."""
    files = [f for f in tile_dir.iterdir() if f.is_file() and not is_hidden(f.name)]
    pairs: Dict[str, Dict[str, Path]] = {}
    for f in files:
        name = f.name.lower()
        stem = f.stem
        if name.endswith("_rgb.tiff") or name.endswith("_rgb.tif"):
            base = stem[:-4] if stem.endswith("_rgb") else stem
            pairs.setdefault(base, {})["rgb"] = f
        elif name.endswith("_ir.tiff") or name.endswith("_ir.tif"):
            base = stem[:-3] if stem.endswith("_ir") else stem
            pairs.setdefault(base, {})["ir"] = f
        elif "mosiac_rgb" in name:
            pairs.setdefault("mosiac", {})["rgb"] = f
        elif "mosiac_ir" in name:
            pairs.setdefault("mosiac", {})["ir"] = f
    return pairs


def pick_mosaic(tile_dir: Path, kind: str) -> Path | None:
    """Pick mosaic file for SAR pairing."""
    candidates = [f for f in tile_dir.iterdir() if f.is_file() and not is_hidden(f.name)]
    key = "rgb" if kind == "rgb" else "ir"
    # prefer mosiac_
    for f in candidates:
        if f.name.lower().startswith("mosiac_") and key in f.name.lower():
            return f
    # fallback: first *_rgb or *_ir
    suffix = f"_{key}"
    for f in candidates:
        if suffix in f.stem.lower():
            return f
    return None


def sar_files(tile_dir: Path) -> List[Path]:
    out = []
    for f in tile_dir.iterdir():
        if f.is_file() and not is_hidden(f.name):
            name = f.name.lower()
            if ("_rgb" in name) or ("_ir" in name):
                continue
            if name.endswith((".tif", ".tiff")):
                out.append(f)
    return out


def process_sar2eo(manifest: csv.writer) -> None:
    dst_input = DST_ROOT / "sar2eo/train/input"
    dst_target = DST_ROOT / "sar2eo/train/target"
    src_splits = [
        SRC_ROOT / "Train_Data_and_Validation_EO_SAR/SAR/train",
        SRC_ROOT / "Train_Data_and_Validation_EO_SAR/SAR/validation",
    ]
    for split_dir in src_splits:
        for fname in sorted(os.listdir(split_dir)):
            if is_hidden(fname) or not fname.endswith(".png"):
                continue
            sar_path = split_dir / fname
            eo_path = sar_path.parent.parent / "EO" / split_dir.name / fname
            if not eo_path.exists():
                continue
            out_in = dst_input / fname
            out_tg = dst_target / fname
            symlink_or_copy(sar_path, out_in)
            symlink_or_copy(eo_path, out_tg)
            manifest.writerow(["sar2eo", "train", out_in.relative_to(DST_ROOT), out_tg.relative_to(DST_ROOT), split_dir.name, "eo_sar"])


def process_rgb2ir(manifest: csv.writer) -> None:
    dst_input = DST_ROOT / "rgb2ir/train/input"
    dst_target = DST_ROOT / "rgb2ir/train/target"
    cities = [
        "Train_Data_Bingham_SAR_IR_RGB",
        "Train_Data_Centerfield_SAR_IR_RGB",
        "Train_Data_Manhattan_SAR_IR_RGB",
        "Train_Data_UC_Davis_SAR_IR_RGB",
    ]
    for city in cities:
        city_dir = SRC_ROOT / city
        for tile in sorted(d for d in city_dir.iterdir() if d.is_dir() and not is_hidden(d.name)):
            pairs = collect_city_pairs(tile)
            for base, paths in pairs.items():
                if "rgb" in paths and "ir" in paths:
                    data_rgb, profile_rgb = resample_to_square(paths["rgb"], 1024)
                    data_ir, profile_ir = resample_to_square(paths["ir"], 1024)
                    out_name = f"{city}_{tile.name}_{base}.tif"
                    out_in = dst_input / out_name
                    out_tg = dst_target / out_name
                    write_tiff(out_in, data_rgb, profile_rgb)
                    write_tiff(out_tg, data_ir, profile_ir)
                    manifest.writerow(["rgb2ir", "train", out_in.relative_to(DST_ROOT), out_tg.relative_to(DST_ROOT), tile.name, city])


def process_sar_to_optical(kind: str, manifest: csv.writer) -> None:
    assert kind in ("ir", "rgb")
    dst_input = DST_ROOT / f"sar2{kind}/train/input"
    dst_target = DST_ROOT / f"sar2{kind}/train/target"
    cities = [
        "Train_Data_Bingham_SAR_IR_RGB",
        "Train_Data_Centerfield_SAR_IR_RGB",
        "Train_Data_Manhattan_SAR_IR_RGB",
        "Train_Data_UC_Davis_SAR_IR_RGB",
    ]
    for city in cities:
        city_dir = SRC_ROOT / city
        for tile in sorted(d for d in city_dir.iterdir() if d.is_dir() and not is_hidden(d.name)):
            target_path = pick_mosaic(tile, kind)
            if target_path is None:
                continue
            target_data, target_profile = resample_to_square(target_path, 1024)
            target_cached = (target_data, target_profile)
            for sar in sar_files(tile):
                sar_data, sar_profile = resample_to_square(sar, 1024)
                out_name = f"{city}_{tile.name}_{sar.stem}_to_{target_path.stem}.tif"
                out_in = dst_input / out_name
                out_tg = dst_target / out_name
                write_tiff(out_in, sar_data, sar_profile)
                write_tiff(out_tg, target_cached[0], target_cached[1])
                manifest.writerow([f"sar2{kind}", "train", out_in.relative_to(DST_ROOT), out_tg.relative_to(DST_ROOT), tile.name, city])


def main() -> None:
    ensure_dir(DST_ROOT / "manifests")
    manifest_path = DST_ROOT / "manifests" / "refined_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "split", "input", "target", "tile", "source_city"])
        process_sar2eo(writer)
        process_rgb2ir(writer)
        process_sar_to_optical("ir", writer)
        process_sar_to_optical("rgb", writer)
    print(f"Done. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
