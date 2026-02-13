# Competition Roadmap

> **Baseline method**: DDBM (`diffusion_unet` in `configs/model_scaling_variants.yaml`).
> **Stage 1 main**: DDBM directly on pixel space (no VAE). Latent-space
> modelling with pre-trained VAE is kept as **supplementary** (see experiment
> observations).  Other baselines (BiBBDM, I2SB, DDIB, CUT, Img2Img-Turbo) are
> left for future exploration.

We progressively increase training compute across four stages. Each stage
builds on the previous one and the corresponding training scripts live in
`scripts/`.

---

## VAE Channel Handling

All pre-trained VAEs mentioned below expect **3-channel** (RGB) input and
produce **3-channel** output. To support single-channel modalities (IR, SAR,
EO) we apply a simple channel-repeat / channel-average technique:

| Step | Operation |
|------|-----------|
| **Encode** | Repeat the 1-channel pixel value across 3 channels → feed the resulting `(H, W, 3)` tensor to the VAE encoder. |
| **Decode** | The VAE decoder outputs a 3-channel image → average the 3 channels to obtain the final 1-channel result. |

In our previous study we found that a **frozen pre-trained VAE works well**
even without retraining on remote-sensing data.

---

## Latent-Space UNet Channel Note

> **✅ Resolved** — When modelling in latent space (`use_latent_target=True`)
> the UNet `in_channels` and `out_channels` now use the `latent_channels`
> config field (default 32, matching the VAE latent dimension) instead of the
> pixel channel count (1 or 3). Each baseline trainer's `build_model` checks
> `use_latent_target` and selects the appropriate channel count automatically.

---

## Tasks & Latent Sizes

| Task | Pixel size | Spatial compression (÷8) | Latent shape | VAE |
|------|-----------|--------------------------|--------------|-----|
| RGB → IR  | 1024 × 1024 | 128 × 128 | `(128, 128, 32)` | FLUX2-VAE / SD21-VAE |
| SAR → IR  | 1024 × 1024 | 128 × 128 | `(128, 128, 32)` | FLUX2-VAE / SD21-VAE |
| SAR → RGB | 1024 × 1024 | 128 × 128 | `(128, 128, 32)` | FLUX2-VAE / SD21-VAE |
| SAR → EO (EO2SAR) | 256 × 256 | 32 × 32 | `(32, 32, 32)` | FLUX2-VAE / SD21-VAE |

---

## Stage 1 — Pixel-Space DDBM (Main)

**Goal**: Quick iteration using DDBM directly on pixel space (no VAE). 1024 px
tasks train at 512 px crop for faster iteration.

**Pipeline**: `DDBMPipeline` (DDBM in pixel space, no VAE).

| Task | Pixel shape | DDBM config tier | ~Params |
|------|-------------|------------------|---------|
| RGB → IR  | `(512, 512)` crop | **medium** | ~120 M |
| SAR → IR  | `(512, 512)` crop | **medium** | ~120 M |
| SAR → RGB | `(512, 512)` crop | **medium** | ~120 M |
| SAR → EO  | `(256, 256)`      | **small**  | ~20 M  |

```bash
# Run all Stage-1 (main) training jobs — pixel-space DDBM
bash scripts/train_stage1_rgb2ir.sh
bash scripts/train_stage1_sar2ir.sh
bash scripts/train_stage1_sar2rgb.sh
bash scripts/train_stage1_sar2eo.sh
```

### DDBM Latent (Supplementary — deprecated)

**Goal**: Latent-space modelling with frozen pre-trained VAE (kept for reference;
see [experiment_observations.md](experiment_observations.md) — yielded pure noise
in initial runs).

**Pipeline**: `DDBMLatentPipeline` (DDBM + frozen VAE). To revert to latent:
set `USE_LATENT_TARGET=true`, `LATENT_VAE_PATH`, and remove `RESOLUTION` override.

### CUT Ablation (Stage 1)

**Goal**: Compare CUT (GAN-based, pixel-space) against DDBM. CUT params are much smaller
than DDBM, so we use **aggressive** tier choices matched to recommended resolution
(`configs/model_scaling_variants.yaml`). 1024 px tasks train at 512 px crop for faster iteration.

| Task | Pixel shape | CUT config tier | ~Params | Notes |
|------|-------------|----------------|---------|-------|
| RGB → IR  | `(512, 512)` crop | **large**  | ~56.5 M | large: 256–512 |
| SAR → IR  | `(512, 512)` crop | **large**  | ~56.5 M | large: 256–512 |
| SAR → RGB | `(512, 512)` crop | **large**  | ~56.5 M | large: 256–512 |
| SAR → EO  | `(256, 256)`      | **medium** | ~14.1 M | medium: 128–256 |

```bash
# Run all Stage-1 CUT ablation jobs
bash scripts/train_stage1_cut_rgb2ir.sh
bash scripts/train_stage1_cut_sar2ir.sh
bash scripts/train_stage1_cut_sar2rgb.sh
bash scripts/train_stage1_cut_sar2eo.sh
```

---

## Stage 2 — Pixel-Space DDBM with Scaled-Up Models (Main)

**Goal**: Increase model capacity while still operating in pixel space (no VAE).
1024 px tasks train at 512 px crop.

**Pipeline**: `DDBMPipeline` (DDBM in pixel space, no VAE).

| Task | Pixel shape | DDBM config tier | ~Params |
|------|-------------|------------------|---------|
| RGB → IR  | `(512, 512)` crop | **large**  | ~404 M |
| SAR → IR  | `(512, 512)` crop | **large**  | ~404 M |
| SAR → RGB | `(512, 512)` crop | **large**  | ~404 M |
| SAR → EO  | `(256, 256)`      | **medium** | ~120 M |

```bash
# Run all Stage-2 (main) training jobs — pixel-space DDBM
bash scripts/train_stage2_pixel_rgb2ir.sh
bash scripts/train_stage2_pixel_sar2ir.sh
bash scripts/train_stage2_pixel_sar2rgb.sh
bash scripts/train_stage2_pixel_sar2eo.sh
```

### DDBM Latent (Supplementary)

**Goal**: Latent-space modelling with frozen pre-trained VAE (kept for reference).

**Pipeline**: `DDBMLatentPipeline` (DDBM + frozen VAE).

| Task | Latent shape | DDBM config tier | ~Params |
|------|-------------|------------------|---------|
| RGB → IR  | `(128, 128, 32)` | **large**  | ~404 M |
| SAR → IR  | `(128, 128, 32)` | **large**  | ~404 M |
| SAR → RGB | `(128, 128, 32)` | **large**  | ~404 M |
| SAR → EO  | `(32, 32, 32)`   | **medium** | ~120 M |

```bash
# Run Stage-2 (sup) — latent-space DDBM (existing scripts preserved)
bash scripts/train_stage2_rgb2ir.sh
bash scripts/train_stage2_sar2ir.sh
bash scripts/train_stage2_sar2rgb.sh
bash scripts/train_stage2_sar2eo.sh
```

### CUT Ablation (Stage 2)

**Goal**: Scale up CUT to huge tier for 1024 px tasks; large for 256 px.

| Task | Pixel shape | CUT config tier | ~Params | Notes |
|------|-------------|----------------|---------|-------|
| RGB → IR  | `(512, 512)` crop | **huge**   | ~293 M | huge: 512–1024 |
| SAR → IR  | `(512, 512)` crop | **huge**   | ~293 M | huge: 512–1024 |
| SAR → RGB | `(512, 512)` crop | **huge**   | ~293 M | huge: 512–1024 |
| SAR → EO  | `(256, 256)`      | **large**  | ~56.5 M | large: 256–512 |

```bash
# Run all Stage-2 CUT ablation jobs
bash scripts/train_stage2_cut_rgb2ir.sh
bash scripts/train_stage2_cut_sar2ir.sh
bash scripts/train_stage2_cut_sar2rgb.sh
bash scripts/train_stage2_cut_sar2eo.sh
```

---

## Stage 3 — Pixel-Space Modelling (More Compute)

**Goal**: Remove the VAE bottleneck by modelling directly in pixel space with
the same architecture tiers as Stage 2.

**Pipeline**: `DDBMPipeline` (DDBM in pixel space, no VAE).

| Task | Pixel shape | DDBM config tier | ~Params |
|------|------------|------------------|---------|
| RGB → IR  | `(1024, 1024)` | **large**  | ~404 M |
| SAR → IR  | `(1024, 1024)` | **large**  | ~404 M |
| SAR → RGB | `(1024, 1024)` | **large**  | ~404 M |
| SAR → EO  | `(256, 256)`   | **medium** | ~120 M |

```bash
bash scripts/train_stage3_rgb2ir.sh
bash scripts/train_stage3_sar2ir.sh
bash scripts/train_stage3_sar2rgb.sh
bash scripts/train_stage3_sar2eo.sh
```

### CUT Ablation (Stage 3)

**Goal**: Full 1024 px resolution for 1024 tasks; huge tier for SAR→EO.

| Task | Pixel shape | CUT config tier | ~Params | Notes |
|------|-------------|----------------|---------|-------|
| RGB → IR  | `(1024, 1024)` | **huge**  | ~293 M | huge: 512–1024 |
| SAR → IR  | `(1024, 1024)` | **huge**  | ~293 M | huge: 512–1024 |
| SAR → RGB | `(1024, 1024)` | **huge**  | ~293 M | huge: 512–1024 |
| SAR → EO  | `(256, 256)`   | **huge**  | ~293 M | aggressive (huge: 512–1024) |

```bash
# Run all Stage-3 CUT ablation jobs
bash scripts/train_stage3_cut_rgb2ir.sh
bash scripts/train_stage3_cut_sar2ir.sh
bash scripts/train_stage3_cut_sar2rgb.sh
bash scripts/train_stage3_cut_sar2eo.sh
```

---

## Stage 4 — Unified Model with RS-VAE

**Goal**: Train a single image-to-image translation foundation model across
all four tasks using progressive resolution training.

**Pipeline**: `DDBMLatentPipeline` (DDBM + RS-VAE).

### Step 4a — RS-VAE (External)

The RS-VAE is **trained and borrowed from an external repository**. It is
initialised from SD21-VAE and fine-tuned on combined remote-sensing data
(SAR, EO, RGB, IR) so that a single VAE can effectively compress and
reconstruct all modalities. The RS-VAE still accepts 3-channel input and we
use the same channel-repeat / channel-average technique described above for
single-channel modalities.

> No local training script is needed — see `scripts/train_stage4a_rs_vae.sh`
> for a reference placeholder.

### Step 4b — 256px Base Model (Flagship for SAR→EO)

Train a unified DDBM with the **large** configuration at **256 px**
resolution on all 4 tasks combined:

- **SAR → EO**: use the native 256 × 256 images directly.
- **RGB → IR / SAR → IR / SAR → RGB**: random-crop from 1024 px to 256 px.

This base model serves as the **flagship checkpoint for the SAR → EO
competition task** at 256 px.

| Component | Config |
|-----------|--------|
| RS-VAE | Borrowed from external repo |
| DDBM | **large** (`diffusion_unet`) ~404 M |
| Resolution | 256 × 256 (random crop from 1024 px for 3 tasks) |
| Training data | All 4 tasks combined |

### Step 4c — 512px Fine-Tune (Flagship for RGB→IR / SAR→IR / SAR→RGB)

Fine-tune the 256 px base model (from Step 4b) at **512 px** resolution using
random crops from the three 1024 px tasks:

- **RGB → IR / SAR → IR / SAR → RGB**: random-crop from 1024 px to 512 px.

This checkpoint serves as the **flagship model for the three 1024 px
competition tasks**.

### Step 4d — 1024px Fine-Tune (Optional)

If sufficient time and compute are available, further fine-tune the 512 px
model at **full 1024 px** resolution on the three 1024 px tasks.

### Random Crop Implementation

For Steps 4b–4d we use a `MultiScaleCrop` augmentation (borrowed from an
external repo) that randomly crops large images to a target resolution during
training. Example: for 256 px training, 1024 × 1024 images are randomly
cropped to 256 × 256; for 512 px training they are cropped to 512 × 512.

```bash
# Step 4a: RS-VAE (external — placeholder only)
bash scripts/train_stage4a_rs_vae.sh

# Step 4b: 256px unified base model
bash scripts/train_stage4b_unified_256.sh

# Step 4c: 512px fine-tune
bash scripts/train_stage4c_unified_512.sh

# Step 4d: 1024px fine-tune (optional)
bash scripts/train_stage4d_unified_1024.sh
```

---

## Summary

| Stage | Space | Pipeline | SAR→EO | RGB→IR / SAR→IR / SAR→RGB | Key idea |
|-------|-------|----------|--------|---------------------------|----------|
| 1 | Pixel | `DDBMPipeline` | small | medium (512 crop) | Fast baseline (no VAE) |
| 1 (sup) | Latent (frozen VAE) | `DDBMLatentPipeline` | small | medium | Supplementary; see experiment_observations |
| 1 (CUT) | Pixel | CUT (ResNet + PatchGAN) | medium | large (512 crop) | GAN ablation, resolution-matched |
| 2 | Pixel | `DDBMPipeline` | medium | large (512 crop) | Scale model (no VAE) |
| 2 (sup) | Latent (frozen VAE) | `DDBMLatentPipeline` | medium | large | Supplementary |
| 2 (CUT) | Pixel | CUT (ResNet + PatchGAN) | large | huge (512 crop) | GAN ablation scaled |
| 3 | Pixel | `DDBMPipeline` | medium | large | Drop VAE bottleneck |
| 3 (CUT) | Pixel | CUT (ResNet + PatchGAN) | huge | huge | GAN ablation, full resolution |
| 4b | Latent (RS-VAE) | `DDBMLatentPipeline` | large (unified, 256px) | large (unified, 256px crop) | Base foundation model |
| 4c | Latent (RS-VAE) | `DDBMLatentPipeline` | — | large (unified, 512px crop) | Fine-tune |
| 4d | Latent (RS-VAE) | `DDBMLatentPipeline` | — | large (unified, 1024px) | Optional full-resolution fine-tune |

---

## Planned Ablation Studies

The following ablation studies are planned for future work.  Each study
isolates one axis of the design space to quantify its impact on translation
quality.

### 1. Full Baselines

Compare all six baseline methods (DDBM, BiBBDM, I2SB, DDIB, CUT,
Img2Img-Turbo) under identical settings for every task to establish a
comprehensive performance table.

### 2. Full Model Scaling

Sweep across four model-size tiers (small / medium / large / huge as
defined in `configs/model_scaling_variants.yaml`) and measure the
quality-vs-compute trade-off for the primary DDBM baseline on all tasks.

### 3. Latent Modelling vs. Pixel Modelling

Compare latent-space modelling (`DDBMLatentPipeline` — Stages 1–2 with frozen
VAE) against pixel-space modelling (`DDBMPipeline` — Stage 3) at matched model
capacity, to quantify the effect of the VAE bottleneck on reconstruction
fidelity and training efficiency.

### 4. Dataset Pruning / Distillation

Investigate data-efficiency strategies:

* **Dataset pruning** – remove redundant or low-quality training pairs and
  measure the effect on final translation quality.
* **Dataset distillation** – synthesise a compact training set that
  preserves the performance of the full dataset.

> ⚠️ *Placeholder — experimental design to be finalised.*

### 5. Model Distillation for Acceleration

Compress the large teacher model (Stage 2 or Stage 4) into a smaller,
faster student model via knowledge distillation to reduce inference cost
while retaining quality.

> ⚠️ *Placeholder — experimental design to be finalised.*

---

## Additional Future Plans

### Resolution Upscaling Strategy (512 px → 1024 px)

We currently **train on 512 px** and **perform inference on 1024 px**. Two paths
are planned depending on compute budget:

1. **Direct 1024 px fine-tuning** — If compute budget allows, we will
   fine-tune the model directly at 1024 px resolution.

2. **Training up-sample techniques** — If compute is constrained, we may
   adopt external feature/image upsampling methods to improve 1024 px
   inference without full-resolution training. Relevant works include:
   AnyUp (Wimmer et al., 2026), FeatUp (Fu et al., 2024), MultiDiffusion
   (Bar-Tal et al., 2023), JAFAR (Couairon et al., 2025). See bibtex below.

```bibtex
@inproceedings{bar-talMultiDiffusionFusingDiffusion2023a,
  title = {{{MultiDiffusion}}: {{Fusing Diffusion Paths}} for {{Controlled Image Generation}}},
  shorttitle = {{{MultiDiffusion}}},
  booktitle = {{{ICML}}},
  author = {{Bar-Tal}, Omer and Yariv, Lior and Lipman, Yaron and Dekel, Tali},
  year = 2023,
  month = jan,
  urldate = {2025-08-28}
}

@inproceedings{couaironJAFARJackAny2025,
  title = {{{JAFAR}}: {{Jack}} up {{Any Feature}} at {{Any Resolution}}},
  shorttitle = {{{JAFAR}}},
  booktitle = {The {{Thirty-ninth Annual Conference}} on {{Neural Information Processing Systems}}},
  author = {Couairon, Paul and Chambon, Loick and Serrano, Louis and Haugeard, Jean-Emmanuel and Cord, Matthieu and Thome, Nicolas},
  year = 2025,
  month = oct,
  urldate = {2026-02-12}
}

@inproceedings{fu2024featup,
  title = {{{FeatUp}}: {{A Model-Agnostic Framework}} for {{Features}} at {{Any Resolution}}},
  shorttitle = {{{FeatUp}}},
  booktitle = {The Twelfth International Conference on Learning Representations},
  author = {Fu, Stephanie and Hamilton, Mark and Brandt, Laura E. and Feldmann, Axel and Zhang, Zhoutong and Freeman, William T.},
  year = 2024,
  urldate = {2024-12-22}
}

@inproceedings{wimmerAnyUpUniversalFeature2026,
  title = {{{AnyUp}}: {{Universal Feature Upsampling}}},
  shorttitle = {{{AnyUp}}},
  booktitle = {The {{Fourteenth International Conference}} on {{Learning Representations}}},
  author = {Wimmer, Thomas and Truong, Prune and Rakotosaona, Marie-Julie and Oechsle, Michael and Tombari, Federico and Schiele, Bernt and Lenssen, Jan Eric},
  year = 2026,
  urldate = {2026-02-12}
}
```
