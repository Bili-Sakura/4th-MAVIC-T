# UniDB Integration

This document describes the integration of [UniDB++](https://github.com/2769433owo/UniDB-plusplus) into this repository.

## Overview

UniDB uses a stochastic optimal control formulation for image-to-image translation (e.g., deraining, super-resolution, inpainting). The model predicts noise; the scheduler supports multiple solvers:

- **euler**: Standard Euler-Maruyama (sde, mean-ode, pf-ode)
- **noise-solver-1**: UniDB++ reduced-step noise predictor
- **noise-solver-2**: Higher-order noise predictor
- **data-solver-1**: Data prediction with reduced steps
- **data-solver-2**: Higher-order data predictor

## Components

| Component | Path | Description |
|-----------|------|-------------|
| Scheduler | `src/schedulers/scheduling_unidb.py` | `UniDBScheduler` with cosine/linear/constant schedules |
| UNet | `src/models/unet_unidb.py` | `UniDBConditionalUNet` (noise predictor) |
| Pipeline | `src/pipelines/unidb/pipeline_unidb.py` | `UniDBPipeline` for inference |

## Usage

### Load checkpoint

UniDB pretrained checkpoints are available from [Google Drive](https://drive.google.com/drive/folders/1UUeeQREpX98bwHdtmOqBqExlr_QmN4bZ?usp=drive_link). The checkpoint format is a `.pth` file with keys like `module.xxx` (DataParallel) or `xxx` (single GPU).

### Inference

```python
from src.pipelines.unidb import UniDBPipeline, UniDBPipelineOutput
from src.models import UniDBConditionalUNet
from src.schedulers import UniDBScheduler

# Load checkpoint (adapt path if needed)
unet = UniDBConditionalUNet(in_channels=3, out_channels=3, nf=64, depth=4)
state = torch.load("path/to/pretrain_model_G.pth", map_location="cpu")
# Remove 'module.' prefix if present (DataParallel)
state = {k.replace("module.", ""): v for k, v in state.items()}
unet.load_state_dict(state, strict=False)

scheduler = UniDBScheduler(
    lambda_square=30,
    gamma=1e7,
    num_train_timesteps=100,
    solver_step=20,  # e.g. 5, 10, 20, 25, 50, 100 for UniDB++
    method="euler",    # or noise-solver-1, data-solver-1, etc.
    solver_type="mean-ode",  # or sde, pf-ode
)

pipe = UniDBPipeline(unet=unet, scheduler=scheduler)
pipe = pipe.to("cuda")

# image: PIL Image or list of PIL Images
output = pipe(image, num_inference_steps=100, output_type="pil")
images = output.images
```

### Parameters

- **lambda_square**: Diffusion coefficient (default 30 for deraining; scaled by 1/255 if >= 1)
- **gamma**: Terminal penalty weight (1e6, 1e7, 1e8,...). Must match pretrained checkpoint.
- **solver_step**: For UniDB++: 5, 10, 20, 25, 50, or 100. For euler: fixed at num_train_timesteps.
- **method**: `euler`, `noise-solver-1`, `noise-solver-2`, `data-solver-1`, `data-solver-2`
- **solver_type**: `sde`, `mean-ode`, or `pf-ode` (for euler and data-solver-1)

## Checkpoint compatibility

UniDB checkpoints use `ConditionalUNet` with `in_nc=3, out_nc=3, nf=64, depth=4`. Our `UniDBConditionalUNet` matches this architecture. If your checkpoint uses different keys (e.g. `module.` prefix), strip them before loading.

## Training

Training examples are in `examples/unidb/`:

```bash
# SAR-to-RGB
python -m examples.unidb.train_sar2rgb --max_train_steps 10000

# SAR-to-IR
python -m examples.unidb.train_sar2ir --max_train_steps 10000

# SAR-to-EO
python -m examples.unidb.train_sar2eo --max_train_steps 10000

# RGB-to-IR
python -m examples.unidb.train_rgb2ir --max_train_steps 10000
```

Override any config field via `--field_name value`, e.g.:

```bash
python -m examples.unidb.train_sar2rgb \
  --lambda_square 30 \
  --gamma 1e7 \
  --train_batch_size 16 \
  --learning_rate 1e-4
```

## References

- [UniDB++ GitHub](https://github.com/2769433owo/UniDB-plusplus)
- [UniDB pretrained checkpoints](https://drive.google.com/drive/folders/1UUeeQREpX98bwHdtmOqBqExlr_QmN4bZ?usp=drive_link)
