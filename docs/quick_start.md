# Quick Start

Use this short guide to get a baseline training run, export predictions, and measure MAVIC-T metrics.

## 1. Environment

- Create the provided conda environment:  
  `conda env create -f environment.yaml && conda activate rsgen`
- Place the refined dataset at `datasets/BiliSakura/MACIV-T-2025-Structure-Refined` (see `docs/dataset.md` for layout). The validation/test inputs live under the same root.

## 2. Train a model (Pix2Pix-Turbo example)

Each task has a dedicated launcher under `src/img2img_turbo/`:

```bash
# Single GPU sar2ir training
python -m src.img2img_turbo.train_sar2ir --output_dir ./outputs/turbo_sar2ir --train_batch_size 2

# Multi-GPU (accelerate)
accelerate launch -m src.img2img_turbo.train_sar2ir --train_batch_size 4
```

Every config field can be overridden on the command line (see `src/img2img_turbo/config.py`). Checkpoints are written to the chosen `--output_dir`.

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

You can compute LPIPS/L1 (and FID when `torchvision` is available) directly while looping over a dataset split:

```python
import torch
from torch.utils.data import DataLoader
from src.metrics import MetricCalculator
from src.img2img_turbo.dataset_wrapper import MavicTTurboDataset
from src.img2img_turbo.models import Pix2PixTurbo
from src.img2img_turbo.config import sar2ir_config

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = sar2ir_config()
model = Pix2PixTurbo(
    pretrained_path="./outputs/turbo_sar2ir/checkpoints/model_final.pkl",
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
