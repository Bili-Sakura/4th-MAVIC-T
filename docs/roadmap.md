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

> **⚠️ TODO** — When modelling in latent space the UNet `in_channels` and
> `out_channels` should match the VAE latent dimension (32) rather than the
> pixel channel count (1 or 3). This mismatch is **intentionally left
> unfixed** in the current config/code and will be updated in a follow-up
> patch.

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
all four tasks.

### Step 4a — Train RS-VAE

Fine-tune SD21-VAE on the combined remote-sensing dataset (SAR, EO, RGB, IR)
so that a single VAE can effectively compress and reconstruct all modalities.
The RS-VAE still accepts 3-channel input and we use the same channel-repeat /
channel-average technique described above for single-channel modalities.

### Step 4b — Train Unified DDBM

Use the RS-VAE to encode all four task pairs into a shared latent space, then
train a single DDBM with the **large** configuration on the combined dataset.

| Component | Config |
|-----------|--------|
| RS-VAE | Initialised from SD21-VAE, fine-tuned on SAR+EO+RGB+IR |
| DDBM | **large** (`diffusion_unet`) ~404 M |
| Training data | All 4 tasks combined (RGB→IR, SAR→IR, SAR→RGB, SAR→EO) |

```bash
# Step 4a: fine-tune RS-VAE
bash scripts/train_stage4a_rs_vae.sh

# Step 4b: unified DDBM training
bash scripts/train_stage4b_unified.sh
```

---

## Summary

| Stage | Space | SAR→EO | RGB→IR / SAR→IR / SAR→RGB | Key idea |
|-------|-------|--------|---------------------------|----------|
| 1 | Latent (frozen VAE) | small | medium | Fast baseline |
| 2 | Latent (frozen VAE) | medium | large | Scale model |
| 3 | Pixel | medium | large | Drop VAE bottleneck |
| 4 | Latent (RS-VAE) | large (unified) | large (unified) | Foundation model |
