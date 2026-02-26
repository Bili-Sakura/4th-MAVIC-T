# Roadmap and Experiments

> Take our previous results as a quick start point, see [staled-experiments](./staled-roadmaps-experiments/experiment_observations.md) and [staled-roadmap](./staled-roadmaps-experiments/roadmap.md).

## Pre-trained models (default paths)

| Model       | Default path                          | Used by                                      |
|-------------|---------------------------------------|----------------------------------------------|
| **Text2Earth** | `models/lcybuaa/Text2Earth`           | EXP-0225 (SAR2RGB InstructPix2Pix)          |
| **Flux2-VAE**  | `models/BiliSakura/VAEs/FLUX2-VAE`    | Latent-space baselines (i2sb, ddib, ddbm, cut, bibbdm, img2img_turbo) |

Ensure these models are available before running experiments that depend on them.

## DDBM-Pixel-Medium (2026/02/13)


**Pipeline**: `DDBMPipeline` (DDBM in pixel space, no VAE).

| Task      | Pixel shape    | DDBM tier | ~Params |
|-----------|----------------|-----------|---------|
| RGB → IR  | `(512, 512)`   | medium    | ~120 M  |
| SAR → IR  | `(512, 512)`   | medium    | ~120 M  |
| SAR → RGB | `(512, 512)`   | medium    | ~120 M  |
| SAR → EO  | `(256, 256)`   | small     | ~20 M   |

```bash
bash scripts/DDBM_Pixel_Medium-0213/train_rgb2ir.sh
bash scripts/DDBM_Pixel_Medium-0213/train_sar2ir.sh
bash scripts/DDBM_Pixel_Medium-0213/train_sar2rgb.sh
bash scripts/DDBM_Pixel_Medium-0213/train_sar2eo.sh
```

## DBIM-Pixel-Medium (2026/02/16)

**Pipeline**: `DBIMPipeline` (DBIM in pixel space, no VAE). Same config as DDBM-Pixel-Medium; DBIM uses improved implicit bridge samplers at inference.

| Task      | Pixel shape    | DBIM tier | ~Params |
|-----------|----------------|-----------|---------|
| RGB → IR  | `(512, 512)`   | medium    | ~120 M  |
| SAR → IR  | `(512, 512)`   | medium    | ~120 M  |
| SAR → RGB | `(512, 512)`   | medium    | ~120 M  |
| SAR → EO  | `(256, 256)`   | small     | ~20 M   |

```bash
bash scripts/DBIM_Pixel_Medium-0216/train_rgb2ir.sh
bash scripts/DBIM_Pixel_Medium-0216/train_sar2ir.sh
bash scripts/DBIM_Pixel_Medium-0216/train_sar2rgb.sh
bash scripts/DBIM_Pixel_Medium-0216/train_sar2eo.sh
```

## CUT SAR2EO medium/large (2026/02/17 and 2026/02/18)

**Note:** Conducted on simple GPUs (e.g. RTX 3090 / 4090), not on a heavy H800 cluster. **Task:** SAR → EO only, using **medium** and **large** CUT sizes from `configs/model_scaling_variants.yaml`. Logging is to **TensorBoard locally** (no SwanLab).

**Pipeline:** `CUTPipeline` (CUT in pixel space, 256×256).

| CUT size | Pixel shape    | ~Params (1ch) | Script                          |
|----------|----------------|---------------|----------------------------------|
| medium   | `(256, 256)`   | ~14.1 M       | `train_sar2eo_medium.sh`         |
| large    | `(256, 256)`   | ~56.5 M       | `train_sar2eo_large.sh`          |

View TensorBoard (after starting a run):

```bash
tensorboard --logdir ./ckpt/4th-MAVIC-T-ckpt-0217
```

```bash
bash scripts/CUT_SAR2EO_0217/train_sar2eo_medium.sh # save checkpoint as -0217
bash scripts/CUT_SAR2EO_0217/train_sar2eo_large.sh # save checkpoint as -0218
```

**Experiment results / observations (SAR2EO medium, paired val):**

Evaluated checkpoints `ckpt/4th-MAVIC-T-ckpt-0217/sar2eo_medium/cut/sar2eo/checkpoint-epoch-{1..10}` on `paired_val_sar2eo.txt` (64 pairs, 256×256) with:

```bash
python -m examples.cut.evaluate_metrics \
  --checkpoint_dir ckpt/4th-MAVIC-T-ckpt-0217/sar2eo_medium/cut/sar2eo/checkpoint-epoch-<N> \
  --manifest datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2eo.txt \
  --task sar2eo --batch_size 16
```

| Epoch | Steps  | LPIPS | L1   | FID     | Task score |
|-------|--------|-------|------|---------|------------|
| 1     | 11,106 | 0.6030 | 0.1362 | 304.30 | 0.5790 |
| 2     | 22,212 | 0.5985 | 0.1320 | 259.63 | 0.5760 |
| 3     | 33,318 | 0.5896 | 0.1289 | 251.78 | 0.5720 |
| 4     | 44,424 | 0.6086 | 0.1304 | 308.93 | 0.5790 |
| 5     | 55,530 | 0.5752 | 0.1412 | 227.79 | 0.5712 |
| 6     | 66,636 | 0.5685 | 0.1310 | 232.25 | **0.5656** |
| 7     | 77,742 | 0.5710 | 0.1335 | **223.21** | 0.5672 |
| 8     | 88,848 | 0.5670 | 0.1350 | 223.74 | 0.5664 |
| 9     | 99,954 | 0.5691 | 0.1357 | 228.89 | 0.5673 |
| 10    | 111,060 | 0.5688 | 0.1428 | 235.13 | 0.5697 |

- **Best task score:** epoch 6 (0.5656). Best FID: epoch 7 (223.21). Task score improves from epoch 1–3 and 4–8, with some fluctuation; epoch 6 is the best checkpoint by composite score.

## DBIM-Pixel-Scaled (2026/02/18)

Based on [DBIM-Pixel-Medium (02/16)](#dbim-pixel-medium-20260216) with **scaled model size** per [Stage 3 — Pixel-Space Modelling](staled-roadmaps-experiments/roadmap.md#stage-3--pixel-space-modelling-more-compute): 1024×1024 tasks use **huge** tier; SAR→EO stays 256×256 with **medium** . Pipeline remains `DBIMPipeline` (pixel space, no VAE).

> [!IMPORTANT]
> We change the [configuration](../configs/model_scaling_variants.yaml) in 2026/02/17.

| Task      | Pixel shape    | DBIM tier | ~Params     |
|-----------|----------------|-----------|-------------|
| RGB → IR  | `(1024, 1024)` | huge      | 416M |
| SAR → IR  | `(1024, 1024)` | huge      | 416M |
| SAR → RGB | `(1024, 1024)` | huge      | 416M |
| SAR → EO  | `(256, 256)`   | medium    | 74M |

```bash
# run on your new exisitng one H800 for GPU:0-7 
CUDA_VISIBLE_DEVICES=0,1 bash scripts/DBIM_Pixel_Scaled-0218/train_rgb2ir.sh
CUDA_VISIBLE_DEVICES=2,3 bash scripts/DBIM_Pixel_Scaled-0218/train_sar2ir.sh
CUDA_VISIBLE_DEVICES=4,5 bash scripts/DBIM_Pixel_Scaled-0218/train_sar2rgb.sh
CUDA_VISIBLE_DEVICES=6,7 bash scripts/DBIM_Pixel_Scaled-0218/train_sar2eo.sh
```

## CUT-Scaled (2026/02/18)

**Pipeline:** `CUTPipeline` (CUT in pixel space). Uses **large** tier for 256px task and **huge** tier for 1024px tasks per [model_scaling_variants.yaml](../configs/model_scaling_variants.yaml). Logging to **SwanLab** (same as DBIM-Pixel-Scaled).

| Task      | Pixel shape    | CUT tier | ~Params (1ch) |
|-----------|----------------|----------|---------------|
| SAR → EO  | `(256, 256)`   | large    | ~56.5 M       |
| RGB → IR  | `(1024, 1024)` | huge     | ~293 M        |
| SAR → IR  | `(1024, 1024)` | huge     | ~293 M        |
| SAR → RGB | `(1024, 1024)` | huge     | ~293 M        |

```bash
# SwanLab logs to ./ckpt/swanlog (workspace: EarthBridge)

# 256px task (large)
bash scripts/CUT_Scaled-0218/train_sar2eo.sh

# 1024px tasks (huge) — may need multi-GPU or smaller batch
bash scripts/CUT_Scaled-0218/train_rgb2ir.sh
bash scripts/CUT_Scaled-0218/train_sar2ir.sh
bash scripts/CUT_Scaled-0218/train_sar2rgb.sh
```

## EXP-0221 SAR2IR (2026/02/21)

**Pipeline**: `DBIMPipeline` (DBIM in pixel space). Focused recovery experiment for the failing `sar2ir` task. Uses **8-GPU training** by default, runtime random crop `1024 → 512`, and SAR-specific lighter architecture (`num_channels=96`, `channel_mult="1,1,2,2,4,4"`). **Result: failed.**

| Task      | Pixel shape    | Notes                    |
|-----------|----------------|--------------------------|
| SAR → IR  | `(512, 512)`   | SAR-lite, crop from 1024 |

```bash
bash scripts/EXP_0221_SAR2IR/train_dbim_sar2ir_8gpu.sh
```

The 512px checkpoint from this experiment feeds into EXP-0222 for direct 1024px tuning.

## EXP-0222 SAR2RGB MultiRes (2026/02/22)

**Pipeline**: `DBIMPipeline` (DBIM in pixel space). DBIM-only follow-up with multi-resolution training. All scripts default to `NGPU=8`. **sar2eo** has no script in this batch.

| Task      | Stage | Pixel shape    | Notes                                      |
|-----------|-------|----------------|--------------------------------------------|
| RGB → IR  | —     | `(1024, 1024)` | Quick 1024 tuning from known good 512 ckpt |
| SAR → IR  | —     | `(1024, 1024)` | 1024 tuning from EXP-0221 checkpoint       |
| SAR → RGB | A     | `(512, 512)`   | Runtime crop 1024→512, SAR-lite            |
| SAR → RGB | B     | `(1024, 1024)` | Direct 1024 fine-tune from Stage A         |

```bash
# RGB→IR: quick 1024 tuning (resume from known 512 checkpoint)
bash scripts/EXP_0222_SAR2RGB_MULTIRES/train_dbim_rgb2ir_1024_quick_from_512_8gpu.sh

# SAR→IR: 1024 tuning from EXP-0221 (run EXP-0221 first)
bash scripts/EXP_0222_SAR2RGB_MULTIRES/train_dbim_sar2ir_1024_from_0221_8gpu.sh

# SAR→RGB: Stage A (512 crop), then Stage B (1024 fine-tune)
bash scripts/EXP_0222_SAR2RGB_MULTIRES/train_dbim_sar2rgb_512_8gpu.sh
bash scripts/EXP_0222_SAR2RGB_MULTIRES/train_dbim_sar2rgb_1024_8gpu.sh
```

## EXP-0225 Text2Earth SAR2RGB InstructPix2Pix (2026/02/25)

**Pipeline**: InstructPix2Pix-style fine-tuning of **Text2Earth** for SAR→RGB. Direct image conditioning by concatenating SAR latent with noisy RGB latent (8-channel UNet input). Only the UNet is trained; VAE and text encoder are frozen. No ControlNet branch. Uses **8-GPU training** by default.

| Task      | Pixel shape    | Base model   | Notes                          |
|-----------|----------------|--------------|--------------------------------|
| SAR → RGB | `(512, 512)`   | Text2Earth   | 8ch UNet (noisy + SAR latent)  |

```bash
bash scripts/EXP_0225_Text2Earth_SAR2RGB/train_sar2rgb_instructpix2pix_8gpu.sh
```

## EXP-0225 SAR2IR/SAR2RGB Despeckle Recovery (2026/02/25)

**Pipeline**: `DBIMPipeline` (DBIM in pixel space). Focused recovery experiment for the failing SAR-conditioned tasks with **SAR-specific despeckling** and **CFG conditioning dropout** to reduce overfitting to SAR speckle patterns. Uses **8-GPU training** by default, runtime random crop `1024 → 512`, and SAR-focused no-attention architecture (`num_channels=96`, `channel_mult="1,1,2,2,4,4"`). For SAR→RGB, we keep `sar2rgb_sup` and enable REPA with MaRS-RGB.

| Task      | Pixel shape    | Notes |
|-----------|----------------|-------|
| SAR → IR  | `(512, 512)`   | Despeckle + CFG dropout; no attention |
| SAR → RGB | `(512, 512)`   | Despeckle + CFG dropout + REPA + `sar2rgb_sup` |

```bash
# SAR→IR (DBIM, despeckle recovery; flexible NGPU via NGPU=<N>)
bash scripts/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/train_dbim_sar2ir_despeckle.sh

# SAR→RGB (DBIM, despeckle recovery; flexible NGPU via NGPU=<N>)
bash scripts/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/train_dbim_sar2rgb_despeckle.sh

# CFG sweep inference on paired val set (default manifest from task config)
TASK=sar2ir bash scripts/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/run_cfg_inference_paired_val.sh
TASK=sar2rgb bash scripts/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/run_cfg_inference_paired_val.sh
```

<details>
<summary>SAR despeckle parameters (click to expand)</summary>

The despeckler uses a blended local mean filter: `out = (1 - strength) * x + strength * mean_filter(x)`.

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **SAR_DESPECKLE_KERNEL_SIZE** | 5 | odd int (3, 5, 7, 9…) | Window size for local average. Larger → more smoothing, more blur. Smaller → preserves edges, less speckle removal. Use 5 for 512px; consider 7–9 for 1024px. |
| **SAR_DESPECKLE_STRENGTH** | 0.6 | 0.0–1.0 | Blend ratio toward smoothed image. 0 = no change; 1 = fully blurred. 0.4–0.7 recommended. |

**Tuning:** If outputs show speckle-like artifacts → increase `STRENGTH` (e.g. 0.8) or `KERNEL_SIZE` (e.g. 7). If outputs are too smooth and lose structure → decrease `STRENGTH` (e.g. 0.4) or `KERNEL_SIZE` (e.g. 3).

</details>



## EXP-0226 SAR2IR/SAR2RGB Small (2026/02/25)

**Pipeline**: `CUTPipeline` (CUT in pixel space). Recovery experiment for the failing **SAR→IR** and **SAR→RGB** tasks only. Uses a **relatively small architecture** — CUT **medium** tier from [model_scaling_variants.yaml](../configs/model_scaling_variants.yaml): `ngf=64`, `ndf=64`, `netG=resnet_9blocks`, `n_layers_D=3`, ~14.1 M params. Same config that succeeded for [CUT SAR2EO (0217)](#cut-sar2eo-mediumlarge-20260217-and-20260218). Trains at **512×512** (crop from 1024) for these 1024px tasks.

| Task      | Pixel shape    | CUT tier | ~Params | Notes                          |
|-----------|----------------|----------|---------|--------------------------------|
| SAR → IR  | `(512, 512)`   | medium   | ~14.1 M | Crop from 1024; match SAR2EO   |
| SAR → RGB | `(512, 512)`   | medium   | ~14.1 M | Crop from 1024; match SAR2EO   |

```bash
# SAR→IR
bash scripts/EXP_0226_SAR2IR_Small/train_cut_sar2ir_small_8gpu.sh   # 8-GPU (default)
bash scripts/EXP_0226_SAR2IR_Small/train_cut_sar2ir_small_1gpu.sh   # 1-GPU

# SAR→RGB
bash scripts/EXP_0226_SAR2IR_Small/train_cut_sar2rgb_small_8gpu.sh   # 8-GPU (default)
bash scripts/EXP_0226_SAR2IR_Small/train_cut_sar2rgb_small_1gpu.sh   # 1-GPU
```

## EXP-0226 CUT SAR2RGB/SAR2IR/SAR2EO Scaled (2026/02/26)

**Pipeline**: `CUTPipeline` (CUT in pixel space). Multi-tier scaling experiment for **SAR→RGB**, **SAR→IR**, and **SAR→EO** tasks. SAR→RGB and SAR→IR train at **512×512** (crop from 1024) across **medium**, **large**, and **huge** CUT tiers; SAR→EO trains at **256×256** with **medium** and **large** tiers. All runs use **1 GPU**. Logging to **SwanLab**.

| Task      | Pixel shape    | CUT tier | ~Params (1ch) | Script                                   |
|-----------|----------------|----------|---------------|------------------------------------------|
| SAR → IR  | `(512, 512)`   | medium   | ~14.1 M       | `train_cut_sar2ir_medium_1gpu.sh`        |
| SAR → IR  | `(512, 512)`   | large    | ~56.5 M       | `train_cut_sar2ir_large_1gpu.sh`         |
| SAR → IR  | `(512, 512)`   | huge     | ~292.9 M      | `train_cut_sar2ir_huge_1gpu.sh`          |
| SAR → RGB | `(512, 512)`   | medium   | ~14.1 M       | `train_cut_sar2rgb_medium_1gpu.sh`       |
| SAR → RGB | `(512, 512)`   | large    | ~56.5 M       | `train_cut_sar2rgb_large_1gpu.sh`        |
| SAR → RGB | `(512, 512)`   | huge     | ~292.9 M      | `train_cut_sar2rgb_huge_1gpu.sh`         |
|---|---|---|---|---|
| SAR → EO  | `(256, 256)`   | medium   | ~14.1 M       | `train_cut_sar2eo_medium_1gpu.sh`        |
| SAR → EO  | `(256, 256)`   | large    | ~56.5 M       | `train_cut_sar2eo_large_1gpu.sh`         |

```bash
# SAR→IR (1 GPU cuda:0)
bash scripts/EXP_0226_CUT_Scaled/train_cut_sar2ir_medium_1gpu.sh
# SAR→IR (1 GPU cuda:1)
bash scripts/EXP_0226_CUT_Scaled/train_cut_sar2ir_large_1gpu.sh
# SAR→IR (2 GPU cuda:2,3)
bash scripts/EXP_0226_CUT_Scaled/train_cut_sar2ir_huge_2gpu.sh

# SAR→RGB (1 GPU cuda:4)
bash scripts/EXP_0226_CUT_Scaled/train_cut_sar2rgb_medium_1gpu.sh
# SAR→RGB (1 GPU cuda:5)
bash scripts/EXP_0226_CUT_Scaled/train_cut_sar2rgb_large_1gpu.sh
# SAR→RGB (2 GPU cuda:6,7)
bash scripts/EXP_0226_CUT_Scaled/train_cut_sar2rgb_huge_2gpu.sh

# SAR→EO (1 GPU, later on if time allowed)
bash scripts/EXP_0226_CUT_Scaled/train_cut_sar2eo_medium_1gpu.sh
bash scripts/EXP_0226_CUT_Scaled/train_cut_sar2eo_large_1gpu.sh
```

Test-set inference (auto-selects latest checkpoint for the given tier):

```bash
# SAR→IR
TIER=medium bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2ir.sh
TIER=large  bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2ir.sh
TIER=huge   bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2ir.sh

# SAR→RGB
TIER=medium bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2rgb.sh
TIER=large  bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2rgb.sh
TIER=huge   bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2rgb.sh

# SAR→EO
TIER=medium bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2eo.sh
TIER=large  bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2eo.sh
```

## EXP-0227 CUT SAR2RGB/SAR2IR Medium, Large & Huge (2026/02/27)

**Pipeline**: `CUTPipeline` (CUT in pixel space). **4- and 8-GPU training** for **SAR→RGB** and **SAR→IR** on **medium** (~14.1 M), **large** (~56.5 M), and **huge** (~293 M) CUT tiers. Trains at **512×512** (crop from 1024). **Gradient clipping** (`max_grad_norm=1.0`) and NaN-skip logic are enabled in the CUT trainer to mitigate mode collapse and NaN loss.

**MAVIC loss (paired pixel supervision):** When `use_mavic_loss=true`, the generator is trained with an additional differentiable loss matching the MAVIC-T evaluation metric: `LPIPS + L1` between predicted and paired target images. This adds direct pixel-to-pixel supervision on top of the standard CUT losses (GAN + NCE). Config: `mavic_lpips_weight=1.0`, `mavic_l1_weight=1.0`, `mavic_loss_weight=0.1`. Enabled for SAR→RGB huge 8-GPU runs.

| Task      | Pixel shape    | CUT tier | ~Params (1ch) | GPUs |
|-----------|----------------|----------|---------------|------|
| SAR → RGB | `(512, 512)`   | medium   | ~14.1 M       | 4, 8 |
| SAR → RGB | `(512, 512)`   | large    | ~56.5 M       | 4, 8 |
| SAR → RGB | `(512, 512)`   | huge     | ~292.9 M      | 4, 8 |
| SAR → IR  | `(512, 512)`   | medium   | ~14.1 M       | 4, 8 |
| SAR → IR  | `(512, 512)`   | large    | ~56.5 M       | 4, 8 |
| SAR → IR  | `(512, 512)`   | huge     | ~292.9 M      | 4, 8 |

```bash
# SAR→RGB — medium (4 or 8 GPU)
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2rgb_medium_4gpu.sh
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2rgb_medium_8gpu.sh

# SAR→RGB — large (4 or 8 GPU)
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2rgb_large_4gpu.sh
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2rgb_large_8gpu.sh

# SAR→RGB — huge (4 or 8 GPU)
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2rgb_huge_4gpu.sh
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2rgb_huge_8gpu.sh

# SAR→IR — medium (4 or 8 GPU)
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2ir_medium_4gpu.sh
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2ir_medium_8gpu.sh

# SAR→IR — large (4 or 8 GPU)
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2ir_large_4gpu.sh
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2ir_large_8gpu.sh

# SAR→IR — huge (4 or 8 GPU)
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2ir_huge_4gpu.sh
bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2ir_huge_8gpu.sh
```