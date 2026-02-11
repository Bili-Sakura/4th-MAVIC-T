"""
Prepare refined training datasets for MAVIC-T crop-aug tasks.

Outputs to: datasets/BiliSakura/MACIV-T-2025-Structure-Refined/
  rgb2ir_crop_aug/train/{input,target}/    1024x1024 TIFF (input=RGB 3-band, target=IR 1-band)
  sar2ir_crop_aug/train/{input,target}/    1024x1024 TIFF (input=SAR 1-band, target=IR 1-band)
  sar2rgb_crop_aug/train/{input,target}/   1024x1024 TIFF (input=SAR 1-band, target=RGB 3-band)
  manifests/refined_manifest_crop_aug.csv

Rules:
- rgb2ir:  from city tiles, pair *_rgb.tiff with *_ir.tiff; crop to 1024x1024.
           RGB and IR share the same CRS/transform, so pixel-level crops are valid.
- sar2ir:  from city tiles, pair each SAR scene with the tile's IR mosaic.
           Both are reprojected into the mosaic's CRS, warped to their geographic
           overlap, then sliding-window cropped at 1024x1024 (spatially aligned).
- sar2rgb: same georef-aligned approach as sar2ir.
- If the aligned overlap is smaller than crop size, fall back to resize-to-square.
- Full set, no split.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window


SRC_ROOT = Path("/mnt/data/projects/4th-MAVIC-T/datasets/BiliSakura/MAVIC-T-2025")
DST_ROOT = Path("/mnt/data/projects/4th-MAVIC-T/datasets/BiliSakura/MACIV-T-2025-Structure-Refined")

CITIES = [
    "Train_Data_Bingham_SAR_IR_RGB",
    "Train_Data_Centerfield_SAR_IR_RGB",
    "Train_Data_Manhattan_SAR_IR_RGB",
    "Train_Data_UC_Davis_SAR_IR_RGB",
]

# Crop configuration
CROP_SIZE = 1024
CROP_STRIDE = 512  # overlap of 512 px for denser augmentation
MAX_CROPS_PER_PAIR = None
TASK_SUFFIX = "_crop_aug"
WORKERS = int(os.environ.get("CROP_WORKERS", "32"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_hidden(name: str) -> bool:
    return name.startswith(".") or name.startswith("._")


def normalize_array(data: np.ndarray, dtype_str: str) -> Tuple[np.ndarray, str]:
    if dtype_str == "uint16":
        data = (data.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
        return data, "uint8"
    return data.clip(0, 255).astype(np.uint8), "uint8"


def out_profile(count: int, size: int, dtype: str = "uint8") -> dict:
    return dict(
        driver="GTiff",
        count=count,
        height=size,
        width=size,
        dtype=dtype,
        transform=from_origin(0, size, 1, 1),
        crs=None,
        tiled=False,
    )


def write_tiff(out_path: Path, data: np.ndarray, profile: dict) -> None:
    ensure_dir(out_path.parent)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def iter_crop_windows(width: int, height: int, size: int, stride: int) -> List[Window]:
    if width < size or height < size:
        return []
    xs = list(range(0, width - size + 1, stride))
    ys = list(range(0, height - size + 1, stride))
    if xs and xs[-1] != width - size:
        xs.append(width - size)
    if ys and ys[-1] != height - size:
        ys.append(height - size)
    return [Window(x, y, size, size) for y in ys for x in xs]


def crop_suffix(window: Window, idx: int) -> str:
    return f"crop{idx:03d}_x{int(window.col_off)}_y{int(window.row_off)}"


# ---------------------------------------------------------------------------
# rgb2ir worker  (same CRS, pixel-level crops are valid)
# ---------------------------------------------------------------------------

def rgb2ir_worker(args: Tuple) -> List[List[str]]:
    city, tile_name, base, rgb_path, ir_path, crop_size, crop_stride, max_crops, task_name, dst_root = args
    dst_root = Path(dst_root)
    dst_input = dst_root / task_name / "train" / "input"
    dst_target = dst_root / task_name / "train" / "target"
    rows: List[List[str]] = []

    with rasterio.open(rgb_path) as rgb_src, rasterio.open(ir_path) as ir_src:
        min_w = min(rgb_src.width, ir_src.width)
        min_h = min(rgb_src.height, ir_src.height)
        windows = iter_crop_windows(min_w, min_h, crop_size, crop_stride)
        if max_crops is not None:
            windows = windows[:max_crops]

        if not windows:
            # Fallback: resize both to crop_size
            rgb_data = rgb_src.read(out_shape=(rgb_src.count, crop_size, crop_size), resampling=Resampling.bilinear)
            ir_data = ir_src.read(out_shape=(ir_src.count, crop_size, crop_size), resampling=Resampling.bilinear)
            rgb_data, rgb_dt = normalize_array(rgb_data, rgb_src.dtypes[0])
            ir_data, ir_dt = normalize_array(ir_data, ir_src.dtypes[0])
            in_name = f"{city}__{tile_name}__{base}__rgb.tif"
            tg_name = f"{city}__{tile_name}__{base}__ir.tif"
            write_tiff(dst_input / in_name, rgb_data, out_profile(rgb_src.count, crop_size, rgb_dt))
            write_tiff(dst_target / tg_name, ir_data, out_profile(ir_src.count, crop_size, ir_dt))
            rows.append([task_name, "train", f"{task_name}/train/input/{in_name}", f"{task_name}/train/target/{tg_name}", tile_name, city])
            return rows

        for idx, window in enumerate(windows):
            rgb_data = rgb_src.read(window=window)
            ir_data = ir_src.read(window=window)
            rgb_data, rgb_dt = normalize_array(rgb_data, rgb_src.dtypes[0])
            ir_data, ir_dt = normalize_array(ir_data, ir_src.dtypes[0])
            suffix = crop_suffix(window, idx)
            in_name = f"{city}__{tile_name}__{base}__{suffix}__rgb.tif"
            tg_name = f"{city}__{tile_name}__{base}__{suffix}__ir.tif"
            write_tiff(dst_input / in_name, rgb_data, out_profile(rgb_src.count, crop_size, rgb_dt))
            write_tiff(dst_target / tg_name, ir_data, out_profile(ir_src.count, crop_size, ir_dt))
            rows.append([task_name, "train", f"{task_name}/train/input/{in_name}", f"{task_name}/train/target/{tg_name}", tile_name, city])
    return rows


# ---------------------------------------------------------------------------
# sar2opt worker  (different CRS -- georef-align first, then crop)
# ---------------------------------------------------------------------------

def _warp_to_grid(src, dst_crs, dst_transform, dst_size: int) -> np.ndarray:
    """Reproject all bands of *src* into a (count, dst_size, dst_size) array."""
    out = np.zeros((src.count, dst_size, dst_size), dtype=np.float32)
    for b in range(src.count):
        reproject(
            source=rasterio.band(src, b + 1),
            destination=out[b],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return out


def sar2opt_worker(args: Tuple) -> List[List[str]]:
    """Georef-aligned crop worker for SAR→optical pairs."""
    city, tile_name, sar_path, target_path, kind, crop_size, crop_stride, max_crops, task_name, dst_root = args
    dst_root = Path(dst_root)
    sar_path = Path(sar_path)
    target_path = Path(target_path)
    dst_input = dst_root / task_name / "train" / "input"
    dst_target = dst_root / task_name / "train" / "target"
    rows: List[List[str]] = []

    with rasterio.open(sar_path) as sar_src, rasterio.open(target_path) as tgt_src:
        dst_crs = tgt_src.crs

        # Reproject SAR bounds into target CRS
        sar_bounds = transform_bounds(sar_src.crs, dst_crs, *sar_src.bounds)
        tgt_bounds = tgt_src.bounds

        # Compute overlap
        left = max(sar_bounds[0], tgt_bounds.left)
        bottom = max(sar_bounds[1], tgt_bounds.bottom)
        right = min(sar_bounds[2], tgt_bounds.right)
        top = min(sar_bounds[3], tgt_bounds.top)

        if right <= left or top <= bottom:
            return rows

        overlap_area = (right - left) * (top - bottom)
        sar_area = (sar_bounds[2] - sar_bounds[0]) * (sar_bounds[3] - sar_bounds[1])
        if sar_area <= 0 or overlap_area / sar_area < 0.5:
            return rows

        # Determine the native resolution of the TARGET (finer grid).
        tgt_res_x = abs(tgt_src.transform.a)
        tgt_res_y = abs(tgt_src.transform.e)

        # Compute pixel dimensions of the overlap at native target resolution.
        overlap_w = int(round((right - left) / tgt_res_x))
        overlap_h = int(round((top - bottom) / tgt_res_y))

        # Build transform for the full overlap region at native resolution.
        overlap_transform = from_bounds(left, bottom, right, top, overlap_w, overlap_h)

        # Warp both into this common grid at native target resolution.
        sar_aligned = np.zeros((sar_src.count, overlap_h, overlap_w), dtype=np.float32)
        for b in range(sar_src.count):
            reproject(
                source=rasterio.band(sar_src, b + 1),
                destination=sar_aligned[b],
                dst_transform=overlap_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        tgt_aligned = np.zeros((tgt_src.count, overlap_h, overlap_w), dtype=np.float32)
        for b in range(tgt_src.count):
            reproject(
                source=rasterio.band(tgt_src, b + 1),
                destination=tgt_aligned[b],
                dst_transform=overlap_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

        sar_count = sar_src.count
        tgt_count = tgt_src.count
        sar_dtype = sar_src.dtypes[0]
        tgt_dtype = tgt_src.dtypes[0]

    # Normalize
    sar_aligned, sar_out_dt = normalize_array(sar_aligned, sar_dtype)
    tgt_aligned, tgt_out_dt = normalize_array(tgt_aligned, tgt_dtype)

    # Slide crop windows over the aligned arrays
    windows = iter_crop_windows(overlap_w, overlap_h, crop_size, crop_stride)
    if max_crops is not None:
        windows = windows[:max_crops]

    # Use SAR stem in BOTH input and target names to avoid race conditions
    # when multiple SAR scenes share the same mosaic target.
    sar_stem = sar_path.stem

    if not windows:
        # Overlap smaller than crop_size: resize to crop_size using numpy interp
        def _resize_array(arr: np.ndarray, h: int, w: int) -> np.ndarray:
            """Simple bilinear-ish resize via rasterio reproject in-memory."""
            from rasterio.transform import from_bounds as _fb
            from rasterio.warp import reproject as _rp
            c = arr.shape[0]
            out = np.zeros((c, h, w), dtype=arr.dtype)
            src_t = _fb(0, 0, arr.shape[2], arr.shape[1], arr.shape[2], arr.shape[1])
            dst_t = _fb(0, 0, arr.shape[2], arr.shape[1], w, h)
            for b in range(c):
                _rp(arr[b], out[b], src_transform=src_t, dst_transform=dst_t,
                    src_crs="EPSG:4326", dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear)
            return out
        sar_resized = _resize_array(sar_aligned, crop_size, crop_size)
        tgt_resized = _resize_array(tgt_aligned, crop_size, crop_size)
        in_name = f"{city}__{tile_name}__{sar_stem}__sar.tif"
        tg_name = f"{city}__{tile_name}__{sar_stem}__{kind}.tif"
        write_tiff(dst_input / in_name, sar_resized, out_profile(sar_count, crop_size, sar_out_dt))
        write_tiff(dst_target / tg_name, tgt_resized, out_profile(tgt_count, crop_size, tgt_out_dt))
        rows.append([task_name, "train", f"{task_name}/train/input/{in_name}", f"{task_name}/train/target/{tg_name}", tile_name, city])
        return rows

    for idx, window in enumerate(windows):
        x = int(window.col_off)
        y = int(window.row_off)
        sar_crop = sar_aligned[:, y:y + crop_size, x:x + crop_size].copy()
        tgt_crop = tgt_aligned[:, y:y + crop_size, x:x + crop_size].copy()
        suffix = crop_suffix(window, idx)
        in_name = f"{city}__{tile_name}__{sar_stem}__{suffix}__sar.tif"
        tg_name = f"{city}__{tile_name}__{sar_stem}__{suffix}__{kind}.tif"
        write_tiff(dst_input / in_name, sar_crop, out_profile(sar_count, crop_size, sar_out_dt))
        write_tiff(dst_target / tg_name, tgt_crop, out_profile(tgt_count, crop_size, tgt_out_dt))
        rows.append([task_name, "train", f"{task_name}/train/input/{in_name}", f"{task_name}/train/target/{tg_name}", tile_name, city])

    return rows


# ---------------------------------------------------------------------------
# Pairing helpers for city tile data
# ---------------------------------------------------------------------------

def collect_rgb_ir_pairs(tile_dir: Path) -> Dict[str, Dict[str, Path]]:
    pairs: Dict[str, Dict[str, Path]] = {}
    for f in tile_dir.iterdir():
        if not f.is_file() or is_hidden(f.name):
            continue
        low = f.name.lower()
        stem = f.stem
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
    key = kind
    for f in tile_dir.iterdir():
        if f.is_file() and not is_hidden(f.name):
            if f.name.lower().startswith(f"mosiac_{key}"):
                return f
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

def process_rgb2ir(writer: csv.writer, workers: int) -> int:
    task = f"rgb2ir{TASK_SUFFIX}"
    count = 0
    work_items: List[Tuple] = []
    for city in CITIES:
        city_dir = SRC_ROOT / city
        tiles = sorted(d for d in city_dir.iterdir() if d.is_dir() and not is_hidden(d.name))
        for tile in tiles:
            pairs = collect_rgb_ir_pairs(tile)
            for base, paths in pairs.items():
                if "rgb" not in paths or "ir" not in paths:
                    continue
                work_items.append((
                    city, tile.name, base,
                    str(paths["rgb"]), str(paths["ir"]),
                    CROP_SIZE, CROP_STRIDE, MAX_CROPS_PER_PAIR,
                    task, str(DST_ROOT),
                ))
    if workers <= 1:
        for item in tqdm(work_items, desc=f"  {task}", unit="tile"):
            for row in rgb2ir_worker(item):
                writer.writerow(row)
                count += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(rgb2ir_worker, item) for item in work_items]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {task}", unit="tile"):
                for row in future.result():
                    writer.writerow(row)
                    count += 1
    print(f"  {task}: {count} pairs")
    return count


def process_sar_to_optical(kind: str, writer: csv.writer, workers: int) -> int:
    assert kind in ("ir", "rgb")
    task = f"sar2{kind}{TASK_SUFFIX}"
    count = 0
    skipped = 0
    work_items: List[Tuple] = []
    for city in CITIES:
        city_dir = SRC_ROOT / city
        tiles = sorted(d for d in city_dir.iterdir() if d.is_dir() and not is_hidden(d.name))
        for tile in tiles:
            target_path = pick_mosaic(tile, kind)
            if target_path is None:
                continue
            for sar_f in sar_files(tile):
                work_items.append((
                    city, tile.name,
                    str(sar_f), str(target_path), kind,
                    CROP_SIZE, CROP_STRIDE, MAX_CROPS_PER_PAIR,
                    task, str(DST_ROOT),
                ))
    if workers <= 1:
        for item in tqdm(work_items, desc=f"  {task}", unit="pair"):
            rows = sar2opt_worker(item)
            if not rows:
                skipped += 1
                continue
            for row in rows:
                writer.writerow(row)
                count += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(sar2opt_worker, item) for item in work_items]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {task}", unit="pair"):
                rows = future.result()
                if not rows:
                    skipped += 1
                    continue
                for row in rows:
                    writer.writerow(row)
                    count += 1
    print(f"  {task}: {count} pairs (workers={workers})")
    if skipped:
        print(f"  [INFO] {task}: skipped {skipped} pairs with insufficient overlap")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare crop-aug refined datasets (georef-aligned).")
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help=f"Number of worker processes (default: {WORKERS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers = max(1, int(args.workers))
    t0 = time.time()
    ensure_dir(DST_ROOT / "manifests")
    manifest_path = DST_ROOT / "manifests" / "refined_manifest_crop_aug.csv"

    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "split", "input", "target", "tile", "source_city"])

        print(f"[1/3] Processing rgb2ir_crop_aug ({CROP_SIZE}x{CROP_SIZE} crops, stride={CROP_STRIDE}, workers={workers})...")
        n = process_rgb2ir(writer, workers)
        print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

        print(f"[2/3] Processing sar2ir_crop_aug ({CROP_SIZE}x{CROP_SIZE} aligned crops, stride={CROP_STRIDE}, workers={workers})...")
        n = process_sar_to_optical("ir", writer, workers)
        print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

        print(f"[3/3] Processing sar2rgb_crop_aug ({CROP_SIZE}x{CROP_SIZE} aligned crops, stride={CROP_STRIDE}, workers={workers})...")
        n = process_sar_to_optical("rgb", writer, workers)
        print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
