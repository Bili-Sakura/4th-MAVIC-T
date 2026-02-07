# Pre-trained Models

This document describes the pre-trained models used in the MAVIC-T project (excluding models trained within this repository).

## Model Paths

All pre-trained models are stored under the `models/` directory with the following structure:

### 1. SARCLIP

**Path:** `models/BiliSakura/SARCLIP`

Vision-language model (ViT-L/14) fine-tuned for SAR (Synthetic Aperture Radar) imagery understanding. Used for SAR feature extraction and cross-modal representation alignment in SAR2EO, SAR2IR, and SAR2RGB tasks.

### 2. DINOv3-sat

**Path:** `models/BiliSakura/DINOv3-sat`

Self-supervised vision transformer (ViT-L) pre-trained on satellite imagery. Provides robust visual features for remote sensing tasks. Used for representation alignment in the RGB2IR task.

### 3. VAEs Collection

**Path:** `models/BiliSakura/VAEs`

A collection of Variational Autoencoders (VAEs) used for latent space encoding/decoding in various tasks. Contains multiple VAE checkpoints for different modalities and resolutions.

---

## Usage Notes

- These models are external dependencies and should be downloaded separately if not already present.
- Model paths are referenced relative to the repository root.
- Ensure proper model loading and initialization according to each model's specific requirements.

## Representation Alignment (REPA)

The representation alignment technique (inspired by [REPA](vendor/REPA/)) uses the
frozen pre-trained encoders above to inject semantic knowledge into the
translation model during training.  The technique is **architecture-agnostic**
and works with any baseline (Pix2Pix-Turbo, CUT, DDBM).

| Task | Encoder | Config field |
|------|---------|-------------|
| `sar2eo` | SARCLIP | `rep_alignment_model_path="./models/BiliSakura/SARCLIP"` |
| `sar2ir` | SARCLIP | `rep_alignment_model_path="./models/BiliSakura/SARCLIP"` |
| `sar2rgb` | SARCLIP | `rep_alignment_model_path="./models/BiliSakura/SARCLIP"` |
| `rgb2ir` | DINOv3-sat | `rep_alignment_model_path="./models/BiliSakura/DINOv3-sat"` |

Enable via `use_rep_alignment=True` in the task config.  The alignment loss
weight is controlled by `lambda_rep_alignment` (default 1.0).
