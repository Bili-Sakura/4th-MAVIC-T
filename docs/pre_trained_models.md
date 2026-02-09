# Pre-trained Models

This document describes the pre-trained models used in the MAVIC-T project (excluding models trained within this repository).

## Model Paths

All pre-trained models are stored under the `models/` directory with the following structure:

### 1. SARCLIP

**Path:** `models/BiliSakura/SARCLIP-ViT-L-14`

Vision-language model (ViT-L/14) fine-tuned for SAR (Synthetic Aperture Radar) imagery understanding. Can be used as an alternative encoder for representation alignment in SAR2EO, SAR2IR, and SAR2RGB tasks.

### 2. MaRS-SAR

**Path:** `models/BiliSakura/MaRS-Base-SAR`

SwinV2-based image encoder (swinv2_base_window8_256) pre-trained for SAR (Synthetic Aperture Radar) imagery. Used as the default encoder for representation alignment in SAR2EO, SAR2IR, and SAR2RGB tasks. Loaded via `transformers`.

### 3. MaRS-RGB

**Path:** `models/BiliSakura/MaRS-Base-RGB`

SwinV2-based image encoder (swinv2_base_window8_256) pre-trained for RGB imagery. Used as the default encoder for representation alignment in the RGB2IR task. Loaded via `transformers`.

### 4. DINOv3-sat

**Path:** `models/facebook/dinov3-vitl16-pretrain-sat493m`

Self-supervised vision transformer (ViT-L) pre-trained on satellite imagery. Provides robust visual features for remote sensing tasks. Can be used as an alternative encoder for representation alignment in the RGB2IR task.

### 5. VAEs Collection

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

| Task | Default Encoder | Config field |
|------|-----------------|-------------|
| `sar2eo` | MaRS-SAR | `rep_alignment_model_path="./models/BiliSakura/MaRS-Base-SAR"` |
| `sar2ir` | MaRS-SAR | `rep_alignment_model_path="./models/BiliSakura/MaRS-Base-SAR"` |
| `sar2rgb` | MaRS-SAR | `rep_alignment_model_path="./models/BiliSakura/MaRS-Base-SAR"` |
| `rgb2ir` | MaRS-RGB | `rep_alignment_model_path="./models/BiliSakura/MaRS-Base-RGB"` |

**Alternative encoders:** SARCLIP (`./models/BiliSakura/SARCLIP-ViT-L-14`) can be used for SAR tasks, and DINOv3-sat (`./models/facebook/dinov3-vitl16-pretrain-sat493m`) can be used for RGB2IR task by overriding the `rep_alignment_model_path` config field.

Enable via `use_rep_alignment=True` in the task config.  The alignment loss
weight is controlled by `lambda_rep_alignment` (default 1.0).
