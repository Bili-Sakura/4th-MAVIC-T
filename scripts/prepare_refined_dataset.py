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
- sar2ir:  from city tiles, pair each SAR scene with the tile's IR mosaic.
           Both are reprojected into a common CRS and cropped to their geographic
           overlap before resampling to 1024x1024 (spatially aligned).
- sar2rgb: same georef-aligned approach as sar2ir.
- Full set, no split.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject, transform_bounds

# Add project root for path_from_root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import path_from_root  # noqa: E402

SRC_ROOT = path_from_root("datasets/BiliSakura/MAVIC-T-2025")
DST_ROOT = path_from_root("datasets/BiliSakura/MACIV-T-2025-Structure-Refined")

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


def warp_aligned_pair(
    sar_path: Path,
    target_path: Path,
    size: int,
) -> Tuple[np.ndarray, np.ndarray, dict, dict] | None:
    """Reproject SAR & target into their geographic overlap at *size x size*.

    1. Compute overlapping bounds in the target's CRS.
    2. Warp both images into that bounding box at the requested pixel size.
    3. Normalize uint16 → uint8 if needed.

    Returns (sar_data, target_data, sar_profile, target_profile) or None if
    the overlap is too small (< 50 % of the SAR footprint).
    """
    with rasterio.open(sar_path) as sar_src, rasterio.open(target_path) as tgt_src:
        # Use the target CRS as the common CRS.
        dst_crs = tgt_src.crs

        # Reproject SAR bounds into target CRS.
        sar_bounds_in_tgt = transform_bounds(sar_src.crs, dst_crs, *sar_src.bounds)
        tgt_bounds = tgt_src.bounds

        # Compute overlap.
        left = max(sar_bounds_in_tgt[0], tgt_bounds.left)
        bottom = max(sar_bounds_in_tgt[1], tgt_bounds.bottom)
        right = min(sar_bounds_in_tgt[2], tgt_bounds.right)
        top = min(sar_bounds_in_tgt[3], tgt_bounds.top)

        if right <= left or top <= bottom:
            return None

        overlap_area = (right - left) * (top - bottom)
        sar_area = (sar_bounds_in_tgt[2] - sar_bounds_in_tgt[0]) * (
            sar_bounds_in_tgt[3] - sar_bounds_in_tgt[1]
        )
        if sar_area <= 0 or overlap_area / sar_area < 0.5:
            return None

        # Build common transform for the overlap region at *size x size*.
        dst_transform = from_bounds(left, bottom, right, top, size, size)

        def _warp(src, band_count):
            out = np.zeros((band_count, size, size), dtype=np.float32)
            for b in range(band_count):
                reproject(
                    source=rasterio.band(src, b + 1),
                    destination=out[b],
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
            return out

        sar_data = _warp(sar_src, sar_src.count)
        tgt_data = _warp(tgt_src, tgt_src.count)

    def _normalize(data: np.ndarray, dtype_str: str):
        if dtype_str == "uint16":
            data = (data / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
            return data, "uint8"
        return data.clip(0, 255).astype(np.uint8), "uint8"

    with rasterio.open(sar_path) as s:
        sar_dtype = s.dtypes[0]
        sar_count = s.count
    with rasterio.open(target_path) as t:
        tgt_dtype = t.dtypes[0]
        tgt_count = t.count

    sar_data, sar_out_dtype = _normalize(sar_data, sar_dtype)
    tgt_data, tgt_out_dtype = _normalize(tgt_data, tgt_dtype)

    base_profile = dict(
        driver="GTiff",
        height=size,
        width=size,
        transform=from_origin(0, size, 1, 1),
        crs=None,
        tiled=False,
    )
    sar_profile = {**base_profile, "count": sar_count, "dtype": sar_out_dtype}
    tgt_profile = {**base_profile, "count": tgt_count, "dtype": tgt_out_dtype}

    return sar_data, tgt_data, sar_profile, tgt_profile


def sar_opt_worker(args: Tuple) -> Tuple[List[str] | None, bool]:
    city, tile_name, sar_path, target_path, task, kind, size = args
    sar_path = Path(sar_path)
    target_path = Path(target_path)
    result = warp_aligned_pair(sar_path, target_path, size)
    if result is None:
        return None, True
    sar_data, tgt_data, sar_profile, tgt_profile = result

    dst_input = DST_ROOT / task / "train" / "input"
    dst_target = DST_ROOT / task / "train" / "target"
    in_name = f"{city}__{tile_name}__{sar_path.stem}__sar.tif"
    tg_name = f"{city}__{tile_name}__{sar_path.stem}__{target_path.stem}__{kind}.tif"
    write_tiff(dst_input / in_name, sar_data, sar_profile)
    write_tiff(dst_target / tg_name, tgt_data, tgt_profile)

    row = [
        task,
        "train",
        f"{task}/train/input/{in_name}",
        f"{task}/train/target/{tg_name}",
        tile_name,
        city,
    ]
    return row, False


def process_sar_to_optical(kind: str, writer: csv.writer, workers: int) -> int:
    """Pair SAR→IR or SAR→RGB from city tiles, spatially aligned to 1024x1024."""
    assert kind in ("ir", "rgb")
    task = f"sar2{kind}"
    count = 0
    skipped = 0
    tasks: List[Tuple] = []
    for city in CITIES:
        city_dir = SRC_ROOT / city
        tiles = sorted(d for d in city_dir.iterdir() if d.is_dir() and not is_hidden(d.name))
        for tile in tiles:
            target_path = pick_mosaic(tile, kind)
            if target_path is None:
                continue
            for sar_f in sar_files(tile):
                tasks.append((city, tile.name, sar_f, target_path, task, kind, 1024))

    if workers <= 1:
        for args in tasks:
            row, was_skipped = sar_opt_worker(args)
            if was_skipped:
                skipped += 1
                continue
            writer.writerow(row)
            count += 1
        print(f"  {task}: {count} pairs")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(sar_opt_worker, args) for args in tasks]
            for future in as_completed(futures):
                row, was_skipped = future.result()
                if was_skipped:
                    skipped += 1
                    continue
                writer.writerow(row)
                count += 1
        print(f"  {task}: {count} pairs (workers={workers})")

    if skipped:
        print(f"  [INFO] {task}: skipped {skipped} pairs with < 50% overlap")
    return count


ALL_TASKS = ("sar2eo", "rgb2ir", "sar2ir", "sar2rgb")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare refined MAVIC-T datasets")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=f"Tasks to run (default: all). Choices: {', '.join(ALL_TASKS)}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="Number of worker processes for SAR alignment (default: min(32, cpu_count)).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing manifest instead of overwriting.",
    )
    args = parser.parse_args()

    tasks = [t.lower() for t in args.tasks] if args.tasks else list(ALL_TASKS)
    for t in tasks:
        if t not in ALL_TASKS:
            parser.error(f"Unknown task '{t}'. Choose from: {', '.join(ALL_TASKS)}")

    t0 = time.time()
    ensure_dir(DST_ROOT / "manifests")
    manifest_path = DST_ROOT / "manifests" / "refined_manifest.csv"

    # When appending, read existing rows for tasks we are NOT re-running.
    existing_rows: List[List[str]] = []
    if args.append and manifest_path.is_file():
        with manifest_path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[0] not in tasks:
                    existing_rows.append(row)

    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "split", "input", "target", "tile", "source_city"])
        for row in existing_rows:
            writer.writerow(row)

        step = 0
        total = len(tasks)

        if "sar2eo" in tasks:
            step += 1
            print(f"[{step}/{total}] Processing sar2eo (256x256 symlinks)...")
            n = process_sar2eo(writer)
            print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

        if "rgb2ir" in tasks:
            step += 1
            print(f"[{step}/{total}] Processing rgb2ir (1024x1024 TIFF)...")
            n = process_rgb2ir(writer)
            print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

        if "sar2ir" in tasks:
            step += 1
            print(f"[{step}/{total}] Processing sar2ir (1024x1024 aligned TIFF)...")
            n = process_sar_to_optical("ir", writer, workers=args.workers)
            print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

        if "sar2rgb" in tasks:
            step += 1
            print(f"[{step}/{total}] Processing sar2rgb (1024x1024 aligned TIFF)...")
            n = process_sar_to_optical("rgb", writer, workers=args.workers)
            print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
