# Quick Start

Use this short guide to get a baseline training run, export predictions, and measure MAVIC-T metrics.

## 1. Environment

- Create the provided conda environment:  
  `conda env create -f environment.yaml && conda activate rsgen`
- Place the refined dataset at `datasets/BiliSakura/MACIV-T-2025-Structure-Refined`.
- Note: the challenge is **MAVIC-T**, but the released folder is spelled **MACIV-T-2025-Structure-Refined**—keep that exact name (see `docs/dataset.md` for layout).
- Validation/test inputs live under the same refined root.

## 2. Train a model (Pix2Pix-Turbo example)

Each task has a dedicated launcher under `examples/img2img_turbo/`:

```bash
# Single GPU sar2ir training
python -m src.img2img_turbo.train_sar2ir --output_dir ./outputs/turbo_sar2ir --train_batch_size 2

# Multi-GPU (accelerate)
accelerate launch -m src.img2img_turbo.train_sar2ir --train_batch_size 4
```

Every config field can be overridden on the command line (see `examples/img2img_turbo/config.py`). Checkpoints are written to the chosen `--output_dir`.

## 3. Run inference

After training, generate predictions for the val/test inputs:

```bash
python -m src.img2img_turbo.sample \
  --task sar2ir \
  --model_path ./outputs/turbo_sar2ir/checkpoints/model_final.pkl \
  --split test \
  --output_dir ./samples/turbo_sar2ir
```

The script reads the evaluation inputs, runs the model, and saves PNGs under `--output_dir`.

## 4. Evaluate locally

For a quick sanity check, compute LPIPS/L1 (and FID when `torchvision` is available) on a slice of the training split (the only split with targets available locally):

```python
import torch
from torch.utils.data import DataLoader
from src.metrics import MetricCalculator
from src.img2img_turbo.dataset_wrapper import MavicTTurboDataset
from src.img2img_turbo.models import Pix2PixTurbo
from src.img2img_turbo.config import sar2ir_config

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = sar2ir_config()
model_path = "./outputs/turbo_sar2ir/checkpoints/model_final.pkl"  # same as the inference example
model = Pix2PixTurbo(
    pretrained_path=model_path,
    pretrained_model_name_or_path=cfg.pretrained_model_name_or_path,
).to(device).eval()
dataset = MavicTTurboDataset(
    task="sar2ir",
    split="train",
    resolution=cfg.resolution,
    model_channels=cfg.model_channels,
    with_target=True,
)
loader = DataLoader(dataset, batch_size=2, shuffle=False)
calc = MetricCalculator(device=device, compute_fid=False)

with torch.no_grad():
    for batch in loader:
        src = batch["conditioning_pixel_values"].to(device)     # [0, 1]
        tgt = (batch["output_pixel_values"].to(device) + 1) / 2 # back to [0, 1]
        preds = (model(src * 2 - 1) + 1) / 2                    # model outputs [-1, 1]
        calc.update(preds, tgt)

print(calc.compute())  # -> MetricResults(lpips=..., fid=None, l1=..., score=...)
```

This offers a quick sanity check on your generated images before packaging a competition submission.

## 5. Filter bad samples & curated fine-tuning

Satellite imagery often contains tiles that are entirely black or filled with N/A values. Use the provided filtering script to detect them and then fine-tune on the curated (clean) dataset.

### Step 1 — Find bad images

```bash
# Scan all tasks (writes bad_samples.txt in the project root)
python scripts/filter_bad_samples.py

# Scan specific tasks only
python scripts/filter_bad_samples.py --tasks sar2ir sar2rgb

# Custom output path and black-pixel threshold
python scripts/filter_bad_samples.py --output ./filtered_paths.txt --black_thresh 1e-6
```

The script checks every training image (input **and** target) for all-zero or all-NaN pixels and writes the absolute paths of the bad files to a text file (one path per line).

### Step 2 — Fine-tune with curated data

Pass the exclude list to any baseline trainer with `--exclude_file`. Combine it with `--num_epochs 1` (or `--n_epochs 1` for CUT) and `--resume_from_checkpoint` for a quick curated fine-tuning run:

```bash
# Pix2Pix-Turbo: 1-epoch fine-tune on clean sar2ir data
python -m src.img2img_turbo.train_sar2ir \
  --exclude_file ./bad_samples.txt \
  --resume_from_checkpoint ./outputs/turbo_sar2ir/checkpoints/model_final.pkl \
  --num_epochs 1 \
  --output_dir ./outputs/turbo_sar2ir_curated

# CUT baseline: 1-epoch fine-tune on clean sar2ir data
python -m src.cut_baseline.train_sar2ir \
  --exclude_file ./bad_samples.txt \
  --resume_from_checkpoint latest \
  --n_epochs 1 \
  --output_dir ./outputs/cut_sar2ir_curated

# DDBM baseline: 1-epoch fine-tune on clean sar2ir data
python -m src.ddbm_baseline.train_sar2ir \
  --exclude_file ./bad_samples.txt \
  --resume_from_checkpoint latest \
  --num_epochs 1 \
  --output_dir ./outputs/ddbm_sar2ir_curated
```

The `--exclude_file` flag is supported by all three baselines and works for every task.
