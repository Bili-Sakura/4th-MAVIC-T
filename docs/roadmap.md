# Competition Roadmap

> **Baseline method**: DDBM (`diffusion_unet` in `configs/model_scaling_variants.yaml`).
> Other baselines (BiBBDM, I2SB, DDIB, CUT, Img2Img-Turbo) are left for future exploration.

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

## Stage 1 — Latent-Space Modelling with Pre-trained VAE (Baseline)

**Goal**: Quick iteration using frozen pre-trained VAEs borrowed from the RGB
domain (FLUX2-VAE with spatial compression ratio 8, latent channels 32; or
SD21-VAE).

| Task | Latent shape | DDBM config tier | ~Params |
|------|-------------|------------------|---------|
| RGB → IR  | `(128, 128, 32)` | **medium** | ~120 M |
| SAR → IR  | `(128, 128, 32)` | **medium** | ~120 M |
| SAR → RGB | `(128, 128, 32)` | **medium** | ~120 M |
| SAR → EO  | `(32, 32, 32)`   | **small**  | ~20 M  |

```bash
# Run all Stage-1 training jobs
bash scripts/train_stage1_rgb2ir.sh
bash scripts/train_stage1_sar2ir.sh
bash scripts/train_stage1_sar2rgb.sh
bash scripts/train_stage1_sar2eo.sh
```

---

## Stage 2 — Latent-Space Modelling with Scaled-Up Models

**Goal**: Increase model capacity while still operating in latent space.

| Task | Latent shape | DDBM config tier | ~Params |
|------|-------------|------------------|---------|
| RGB → IR  | `(128, 128, 32)` | **large**  | ~404 M |
| SAR → IR  | `(128, 128, 32)` | **large**  | ~404 M |
| SAR → RGB | `(128, 128, 32)` | **large**  | ~404 M |
| SAR → EO  | `(32, 32, 32)`   | **medium** | ~120 M |

```bash
bash scripts/train_stage2_rgb2ir.sh
bash scripts/train_stage2_sar2ir.sh
bash scripts/train_stage2_sar2rgb.sh
bash scripts/train_stage2_sar2eo.sh
```

---

## Stage 3 — Pixel-Space Modelling (More Compute)

**Goal**: Remove the VAE bottleneck by modelling directly in pixel space with
the same architecture tiers as Stage 2.

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

---

## Stage 4 — Unified Model with RS-VAE

**Goal**: Train a single image-to-image translation foundation model across
all four tasks using progressive resolution training.

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

## Representation Alignment (REPA)

All four training stages now use **REPA** (REPresentation Alignment) by
default.  A frozen pre-trained encoder extracts features from the source
image, while a trainable projection head maps the translation model's
output features into the same embedding space.  A negative-cosine-similarity
loss encourages the model to preserve the semantic content captured by the
encoder.

| Task | Default encoder |
|------|-----------------|
| SAR → EO | MaRS-SAR (SwinV2) |
| SAR → IR | MaRS-SAR (SwinV2) |
| SAR → RGB | MaRS-SAR (SwinV2) |
| RGB → IR | MaRS-RGB (SwinV2) |

REPA is enabled via `--use_rep_alignment true` in every training script.
The encoder checkpoint path is pre-configured in each task's
`TaskConfig` preset.

---

## Summary

| Stage | Space | SAR→EO | RGB→IR / SAR→IR / SAR→RGB | Key idea |
|-------|-------|--------|---------------------------|----------|
| 1 | Latent (frozen VAE) | small | medium | Fast baseline + REPA |
| 2 | Latent (frozen VAE) | medium | large | Scale model + REPA |
| 3 | Pixel | medium | large | Drop VAE bottleneck + REPA |
| 4b | Latent (RS-VAE) | large (unified, 256px) | large (unified, 256px crop) | Base foundation model + REPA |
| 4c | Latent (RS-VAE) | — | large (unified, 512px crop) | Fine-tune + REPA |
| 4d | Latent (RS-VAE) | — | large (unified, 1024px) | Optional full-resolution fine-tune + REPA |

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

Compare latent-space modelling (Stages 1–2 with frozen VAE) against
pixel-space modelling (Stage 3) at matched model capacity, to quantify the
effect of the VAE bottleneck on reconstruction fidelity and training
efficiency.

### 4. Advanced REPA Variants

Evaluate improved representation alignment techniques beyond the baseline
negative-cosine-similarity REPA loss:

* **REG** (Representation-Enhanced Generation) — *placeholder*
* **Multi-scale REPA** — *placeholder*
* **Contrastive REPA** — *placeholder*

> ⚠️ Implementation details will be added once the baseline REPA results
> are available.

### 5. Dataset Pruning / Distillation

Investigate data-efficiency strategies:

* **Dataset pruning** – remove redundant or low-quality training pairs and
  measure the effect on final translation quality.
* **Dataset distillation** – synthesise a compact training set that
  preserves the performance of the full dataset.

> ⚠️ *Placeholder — experimental design to be finalised.*

### 6. Model Distillation for Acceleration

Compress the large teacher model (Stage 2 or Stage 4) into a smaller,
faster student model via knowledge distillation to reduce inference cost
while retaining quality.

> ⚠️ *Placeholder — experimental design to be finalised.*
