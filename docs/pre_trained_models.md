# Pre-trained Models

This document describes the pre-trained models used in the MAVIC-T project (excluding models trained within this repository).

## Model Paths

All pre-trained models are stored under the `models/` directory with the following structure:

### 1. SARCLIP

**Path:** `models/BiliSakura/SARCLIP-ViT-L-14`

Vision-language model fine-tuned for SAR (Synthetic Aperture Radar) imagery understanding. Used for SAR feature extraction and cross-modal alignment.

### 2. DINOv3-sat

**Path:** `models/facebook/dinov3-vitl16-pretrain-sat493m`

Self-supervised vision transformer pre-trained on satellite imagery. Provides robust visual features for remote sensing tasks.

### 3. VAEs Collection

**Path:** `models/BiliSakura/VAEs`

A collection of Variational Autoencoders (VAEs) used for latent space encoding/decoding in various tasks. Contains multiple VAE checkpoints for different modalities and resolutions.

---

## Usage Notes

- These models are external dependencies and should be downloaded separately if not already present.
- Model paths are referenced relative to the repository root.
- Ensure proper model loading and initialization according to each model's specific requirements.
