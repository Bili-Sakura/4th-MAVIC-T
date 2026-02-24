#!/usr/bin/env python3
"""Quick statistics of MACIV-T-2025-Structure-Refined training data.
Random select 10 pairs per task, report min, max, mean, std for input and output.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets/BiliSakura/MACIV-T-2025-Structure-Refined"
SEED = 42


def _resolve_path(root: Path, rel_path: str) -> Path:
    p = (root / rel_path).resolve()
    if not p.exists() and str(root).startswith("/data/"):
        alt = Path("/mnt/data") / Path(str(root)).relative_to("/data")
        p2 = (alt / rel_path).resolve()
        if p2.exists():
            return p2
    if not p.exists() and str(root).startswith("/mnt/"):
        alt = Path("/data") / Path(str(root)).relative_to("/mnt/data")
        p2 = (alt / rel_path).resolve()
        if p2.exists():
            return p2
    return p


def _load_image_stats(path: Path) -> dict | None:
    try:
        with Image.open(path) as img:
            arr = np.array(img, dtype=np.float64)
    except Exception as e:
        return {"error": str(e)}

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    # Normalize to [0,1] for reporting
    if arr.max() > 1.0:
        if arr.max() > 255:
            arr = arr / 65535.0
        else:
            arr = arr / 255.0

    flat = arr.flatten()
    return {
        "shape": arr.shape,
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "dtype": str(arr.dtype),
    }


def main():
    random.seed(SEED)
    root = DATASET_ROOT.resolve()
    if not root.exists():
        print(f"Dataset root not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Collect pairs per core task (merge crop_aug into base)
    task_pairs: dict[str, list[tuple[str, str]]] = {}

    for manifest_name in ["refined_manifest.csv", "refined_manifest_crop_aug.csv"]:
        mp = root / "manifests" / manifest_name
        if not mp.exists():
            continue
        with mp.open(newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("split") or "").lower() != "train":
                    continue
                task = (row.get("task") or "").lower()
                core = task.replace("_crop_aug", "")
                inp, tgt = row.get("input", ""), row.get("target", "")
                if inp and tgt:
                    task_pairs.setdefault(core, []).append((inp, tgt))

    # Add sar2rgb_sup
    sup_path = root / "manifests" / "paired_sar2rgb_sup.txt"
    if sup_path.exists():
        pairs = []
        with sup_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    inp, tgt = parts[0].strip(), parts[1].strip()
                    if inp.startswith("sar2rgb_sup/"):
                        pairs.append((inp, tgt))
        if pairs:
            task_pairs.setdefault("sar2rgb", []).extend(pairs)

    # Deduplicate by input path
    for t in task_pairs:
        seen = set()
        unique = []
        for inp, tgt in task_pairs[t]:
            if inp not in seen:
                seen.add(inp)
                unique.append((inp, tgt))
        task_pairs[t] = unique

    # Summary counts
    print("=" * 60)
    print("MACIV-T-2025-Structure-Refined Training Data Summary")
    print("=" * 60)
    for task in sorted(task_pairs.keys()):
        n = len(task_pairs[task])
        print(f"  {task}: {n:,} pairs")
    print(f"  Total: {sum(len(v) for v in task_pairs.values()):,} pairs")
    print()

    # Random 10 per task, compute stats
    print("=" * 60)
    print("Random 10 samples per task – Input & Output Statistics")
    print("=" * 60)

    for task in sorted(task_pairs.keys()):
        pairs = task_pairs[task]
        n_sample = min(10, len(pairs))
        sampled = random.sample(pairs, n_sample)

        print(f"\n### {task} (n={n_sample}) ###")

        inp_stats_list, tgt_stats_list = [], []
        for i, (inp_rel, tgt_rel) in enumerate(sampled):
            inp_path = _resolve_path(root, inp_rel)
            tgt_path = _resolve_path(root, tgt_rel)

            inp_s = _load_image_stats(inp_path)
            tgt_s = _load_image_stats(tgt_path)

            if "error" in inp_s or "error" in tgt_s:
                print(f"  [{i+1}] input: {inp_s.get('error', 'ok')} | target: {tgt_s.get('error', 'ok')}")
                continue

            inp_stats_list.append(inp_s)
            tgt_stats_list.append(tgt_s)

        if not inp_stats_list:
            continue

        # Aggregate over 10 samples
        def agg(stats_list, name):
            mins = [s["min"] for s in stats_list]
            maxs = [s["max"] for s in stats_list]
            means = [s["mean"] for s in stats_list]
            stds = [s["std"] for s in stats_list]
            shapes = [s["shape"] for s in stats_list]
            return {
                "min": (min(mins), max(mins)),
                "max": (min(maxs), max(maxs)),
                "mean": (min(means), max(means), np.mean(means), np.std(means)),
                "std": (min(stds), max(stds), np.mean(stds), np.std(stds)),
                "shapes": shapes,
            }

        inp_agg = agg(inp_stats_list, "input")
        tgt_agg = agg(tgt_stats_list, "target")

        print("  INPUT:")
        print(f"    shape: {inp_agg['shapes'][0]} (all same: {len(set(str(s) for s in inp_agg['shapes']))==1})")
        print(f"    min:   {inp_agg['min'][0]:.6f} ~ {inp_agg['min'][1]:.6f}")
        print(f"    max:   {inp_agg['max'][0]:.6f} ~ {inp_agg['max'][1]:.6f}")
        print(f"    mean:  {inp_agg['mean'][2]:.6f} ± {inp_agg['mean'][3]:.6f}")
        print(f"    std:   {inp_agg['std'][2]:.6f} ± {inp_agg['std'][3]:.6f}")

        print("  OUTPUT:")
        print(f"    shape: {tgt_agg['shapes'][0]} (all same: {len(set(str(s) for s in tgt_agg['shapes']))==1})")
        print(f"    min:   {tgt_agg['min'][0]:.6f} ~ {tgt_agg['min'][1]:.6f}")
        print(f"    max:   {tgt_agg['max'][0]:.6f} ~ {tgt_agg['max'][1]:.6f}")
        print(f"    mean:  {tgt_agg['mean'][2]:.6f} ± {tgt_agg['mean'][3]:.6f}")
        print(f"    std:   {tgt_agg['std'][2]:.6f} ± {tgt_agg['std'][3]:.6f}")

        print("  Sampled pairs:")
        for i, (inp, tgt) in enumerate(sampled[:10]):
            print(f"    [{i+1}] {Path(inp).name} <-> {Path(tgt).name}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
