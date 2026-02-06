"""
Prepare refined training datasets for MAVIC-T crop-aug tasks.

Outputs to: datasets/BiliSakura/MACIV-T-2025-Structure-Refined/
  rgb2ir_crop_aug/train/{input,target}/    1024x1024 TIFF (input=RGB 3-band, target=IR 1-band)
  sar2ir_crop_aug/train/{input,target}/    1024x1024 TIFF (input=SAR 1-band, target=IR 1-band)
  sar2rgb_crop_aug/train/{input,target}/   1024x1024 TIFF (input=SAR 1-band, target=RGB 3-band)
  manifests/refined_manifest_crop_aug.csv

Rules:
- rgb2ir:  from city tiles, pair *_rgb.tiff with *_ir.tiff; crop to 1024x1024.
- sar2ir:  from city tiles, pair each SAR scene with the tile's IR mosaic; crop to 1024.
- sar2rgb: from city tiles, pair each SAR scene with the tile's RGB mosaic; crop to 1024.
- If an image is smaller than crop size, fall back to resize-to-square.
- Full set, no split.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window


SRC_ROOT = Path("/mnt/data/projects/4th-MAVIC-T/datasets/BiliSakura/MAVIC-T-2025")
DST_ROOT = Path("/mnt/data/projects/4th-MAVIC-T/datasets/BiliSakura/MACIV-T-2025-Structure-Refined")

CITIES = [
    "Train_Data_Bingham_SAR_IR_RGB",
    "Train_Data_Centerfield_SAR_IR_RGB",
    "Train_Data_Manhattan_SAR_IR_RGB",
    "Train_Data_UC_Davis_SAR_IR_RGB",
]

# Crop configuration (adjust as needed)
CROP_SIZE = 1024
CROP_STRIDE = 1024  # lower than size gives overlap
MAX_CROPS_PER_PAIR = None  # set to int to cap crops per pair
TASK_SUFFIX = "_crop_aug"
WORKERS = int(os.environ.get("CROP_WORKERS", "1"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_hidden(name: str) -> bool:
    return name.startswith(".") or name.startswith("._")


def normalize_dtype(data: np.ndarray, profile: dict) -> Tuple[np.ndarray, dict]:
    if profile.get("dtype") == "uint16":
        data = (data.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
        profile["dtype"] = "uint8"
    return data, profile


def profile_for_size(src_profile: dict, size: int) -> dict:
    profile = src_profile.copy()
    profile.update(
        height=size,
        width=size,
        transform=from_origin(0, size, 1, 1),
        crs=None,
        driver="GTiff",
        tiled=False,
    )
    return profile


def resample_to_square(src_path: Path, size: int) -> Tuple[np.ndarray, dict]:
    """Read image and resample to size x size, dropping georeference."""
    with rasterio.open(src_path) as src:
        data = src.read(
            out_shape=(src.count, size, size),
            resampling=Resampling.bilinear,
        )
        profile = profile_for_size(src.profile, size)
    data, profile = normalize_dtype(data, profile)
    return data, profile


def read_crop(src: rasterio.io.DatasetReader, window: Window) -> Tuple[np.ndarray, dict]:
    """Read a crop window and return data/profile with georef dropped."""
    data = src.read(window=window)
    profile = profile_for_size(src.profile, int(window.height))
    data, profile = normalize_dtype(data, profile)
    return data, profile


def write_tiff(out_path: Path, data: np.ndarray, profile: dict) -> None:
    ensure_dir(out_path.parent)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)


def write_tiff_if_missing(out_path: Path, data: np.ndarray, profile: dict) -> None:
    if out_path.exists():
        return
    write_tiff(out_path, data, profile)


def symlink_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


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


def limit_windows(windows: List[Window], max_crops: int | None) -> List[Window]:
    if max_crops is None or len(windows) <= max_crops:
        return windows
    return windows[:max_crops]


def crop_suffix(window: Window, idx: int) -> str:
    return f"crop{idx:03d}_x{int(window.col_off)}_y{int(window.row_off)}"


def run_tasks(
    tasks: Iterable[Tuple],
    worker_fn,
    workers: int,
) -> Iterable[List[List[str]]]:
    if workers <= 1:
        for task in tasks:
            yield worker_fn(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(worker_fn, tasks, chunksize=1):
            yield result


def rgb2ir_worker(task: Tuple) -> List[List[str]]:
    (
        city,
        tile_name,
        base,
        rgb_path,
        ir_path,
        crop_size,
        crop_stride,
        max_crops,
        task_name,
        dst_root,
    ) = task
    dst_root = Path(dst_root)
    dst_input = dst_root / task_name / "train" / "input"
    dst_target = dst_root / task_name / "train" / "target"
    rows: List[List[str]] = []
    with rasterio.open(rgb_path) as rgb_src, rasterio.open(ir_path) as ir_src:
        min_w = min(rgb_src.width, ir_src.width)
        min_h = min(rgb_src.height, ir_src.height)
        windows = iter_crop_windows(min_w, min_h, crop_size, crop_stride)
        windows = limit_windows(windows, max_crops)
        if not windows:
            data_rgb, prof_rgb = resample_to_square(Path(rgb_path), crop_size)
            data_ir, prof_ir = resample_to_square(Path(ir_path), crop_size)
            in_name = f"{city}__{tile_name}__{base}__rgb.tif"
            tg_name = f"{city}__{tile_name}__{base}__ir.tif"
            write_tiff(dst_input / in_name, data_rgb, prof_rgb)
            write_tiff(dst_target / tg_name, data_ir, prof_ir)
            rows.append([
                task_name, "train",
                f"{task_name}/train/input/{in_name}",
                f"{task_name}/train/target/{tg_name}",
                tile_name, city,
            ])
            return rows
        for idx, window in enumerate(windows):
            data_rgb, prof_rgb = read_crop(rgb_src, window)
            data_ir, prof_ir = read_crop(ir_src, window)
            suffix = crop_suffix(window, idx)
            in_name = f"{city}__{tile_name}__{base}__{suffix}__rgb.tif"
            tg_name = f"{city}__{tile_name}__{base}__{suffix}__ir.tif"
            write_tiff(dst_input / in_name, data_rgb, prof_rgb)
            write_tiff(dst_target / tg_name, data_ir, prof_ir)
            rows.append([
                task_name, "train",
                f"{task_name}/train/input/{in_name}",
                f"{task_name}/train/target/{tg_name}",
                tile_name, city,
            ])
    return rows


def sar2opt_worker(task: Tuple) -> List[List[str]]:
    (
        city,
        tile_name,
        sar_path,
        target_path,
        kind,
        crop_size,
        crop_stride,
        max_crops,
        task_name,
        dst_root,
    ) = task
    dst_root = Path(dst_root)
    dst_input = dst_root / task_name / "train" / "input"
    dst_target = dst_root / task_name / "train" / "target"
    rows: List[List[str]] = []
    with rasterio.open(target_path) as target_src, rasterio.open(sar_path) as sar_src:
        min_w = min(sar_src.width, target_src.width)
        min_h = min(sar_src.height, target_src.height)
        windows = iter_crop_windows(min_w, min_h, crop_size, crop_stride)
        windows = limit_windows(windows, max_crops)
        if not windows:
            sar_data, sar_profile = resample_to_square(Path(sar_path), crop_size)
            target_data, target_profile = resample_to_square(Path(target_path), crop_size)
            in_name = f"{city}__{tile_name}__{Path(sar_path).stem}__sar.tif"
            tg_name = f"{city}__{tile_name}__{Path(target_path).stem}__{kind}.tif"
            write_tiff(dst_input / in_name, sar_data, sar_profile)
            write_tiff_if_missing(dst_target / tg_name, target_data, target_profile)
            rows.append([
                task_name, "train",
                f"{task_name}/train/input/{in_name}",
                f"{task_name}/train/target/{tg_name}",
                tile_name, city,
            ])
            return rows
        for idx, window in enumerate(windows):
            sar_data, sar_profile = read_crop(sar_src, window)
            target_data, target_profile = read_crop(target_src, window)
            suffix = crop_suffix(window, idx)
            in_name = f"{city}__{tile_name}__{Path(sar_path).stem}__{suffix}__sar.tif"
            tg_name = f"{city}__{tile_name}__{Path(target_path).stem}__{suffix}__{kind}.tif"
            write_tiff(dst_input / in_name, sar_data, sar_profile)
            write_tiff_if_missing(dst_target / tg_name, target_data, target_profile)
            rows.append([
                task_name, "train",
                f"{task_name}/train/input/{in_name}",
                f"{task_name}/train/target/{tg_name}",
                tile_name, city,
            ])
    return rows

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


def process_rgb2ir(writer: csv.writer, workers: int) -> int:
    """Pair RGB→IR from city tiles, crop to CROP_SIZE."""
    task = f"rgb2ir{TASK_SUFFIX}"
    count = 0
    tasks: List[Tuple] = []
    for city in CITIES:
        city_dir = SRC_ROOT / city
        tiles = sorted(d for d in city_dir.iterdir() if d.is_dir() and not is_hidden(d.name))
        for tile in tiles:
            pairs = collect_rgb_ir_pairs(tile)
            for base, paths in pairs.items():
                if "rgb" not in paths or "ir" not in paths:
                    continue
                tasks.append((
                    city,
                    tile.name,
                    base,
                    str(paths["rgb"]),
                    str(paths["ir"]),
                    CROP_SIZE,
                    CROP_STRIDE,
                    MAX_CROPS_PER_PAIR,
                    task,
                    str(DST_ROOT),
                ))
    for rows in run_tasks(tasks, rgb2ir_worker, workers):
        for row in rows:
            writer.writerow(row)
        count += len(rows)
    print(f"  rgb2ir: {count} pairs")
    return count


def process_sar_to_optical(kind: str, writer: csv.writer, workers: int) -> int:
    """Pair SAR→IR or SAR→RGB from city tiles, crop to CROP_SIZE."""
    assert kind in ("ir", "rgb")
    task = f"sar2{kind}{TASK_SUFFIX}"
    count = 0
    tasks: List[Tuple] = []
    for city in CITIES:
        city_dir = SRC_ROOT / city
        tiles = sorted(d for d in city_dir.iterdir() if d.is_dir() and not is_hidden(d.name))
        for tile in tiles:
            target_path = pick_mosaic(tile, kind)
            if target_path is None:
                continue
            for sar_f in sar_files(tile):
                tasks.append((
                    city,
                    tile.name,
                    str(sar_f),
                    str(target_path),
                    kind,
                    CROP_SIZE,
                    CROP_STRIDE,
                    MAX_CROPS_PER_PAIR,
                    task,
                    str(DST_ROOT),
                ))
    for rows in run_tasks(tasks, sar2opt_worker, workers):
        for row in rows:
            writer.writerow(row)
        count += len(rows)
    print(f"  {task}: {count} pairs")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare crop-aug refined datasets.")
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="Number of worker processes (default: env CROP_WORKERS or 1).",
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

        print(f"[1/3] Processing rgb2ir ({CROP_SIZE}x{CROP_SIZE} crops, workers={workers})...")
        n = process_rgb2ir(writer, workers)
        print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

        print(f"[2/3] Processing sar2ir ({CROP_SIZE}x{CROP_SIZE} crops, workers={workers})...")
        n = process_sar_to_optical("ir", writer, workers)
        print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

        print(f"[3/3] Processing sar2rgb ({CROP_SIZE}x{CROP_SIZE} crops, workers={workers})...")
        n = process_sar_to_optical("rgb", writer, workers)
        print(f"  => {n} pairs  ({time.time()-t0:.0f}s)\n")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
