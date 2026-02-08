#!/usr/bin/env bash
# Stage 4b — Unified DDBM base model at 256px
#
# Train a single DDBM on ALL 4 tasks at 256px resolution:
#   - SAR→EO:  native 256×256 images
#   - RGB→IR / SAR→IR / SAR→RGB:  random-crop from 1024px to 256px
#
# This checkpoint is the flagship model for the SAR→EO competition task.
# It can also be used for 1024px tasks but is further fine-tuned in 4c/4d.
#
# DDBM large config (~404 M params).
#
# Usage:
#   bash scripts/train_stage4b_unified_256.sh
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage4b_unified_256.sh
#
# TODO: The unified multi-task training loop is not yet implemented.
#   This script will be updated once the data loader supports mixed-task
#   batches with random crop and the RS-VAE checkpoint is available.

set -euo pipefail

NGPU="${NGPU:-1}"

# --- Large config from configs/model_scaling_variants.yaml ---
NUM_CHANNELS=256
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,2,4,4"

# --- Latent-space settings (RS-VAE from external repo) ---
USE_LATENT_TARGET=true
LATENT_VAE_PATH="./models/rs_vae"

# --- Resolution & crop ---
RESOLUTION=256   # all tasks trained at 256px (1024px tasks are random-cropped)

OUTPUT_DIR="./ckpt/stage4b_unified_256"

echo "=== Stage 4b: Unified DDBM — 256px base model ==="
echo "  UNet config : large (num_channels=${NUM_CHANNELS})"
echo "  Resolution  : ${RESOLUTION}px (random crop from 1024px for 3 tasks)"
echo "  RS-VAE      : ${LATENT_VAE_PATH}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo ""
echo "  Tasks: SAR→EO (native 256px) + RGB→IR, SAR→IR, SAR→RGB (crop 1024→256)"
echo "  Flagship for: SAR→EO competition task"
echo ""
echo "⚠️  Unified multi-task training loop is not yet implemented."
echo "    Planned invocation (placeholder):"
echo "    python -m examples.ddbm.train_unified \\"
echo "      --num_channels ${NUM_CHANNELS} \\"
echo "      --num_res_blocks ${NUM_RES_BLOCKS} \\"
echo "      --attention_resolutions ${ATTENTION_RESOLUTIONS} \\"
echo "      --channel_mult ${CHANNEL_MULT} \\"
echo "      --resolution ${RESOLUTION} \\"
echo "      --use_latent_target ${USE_LATENT_TARGET} \\"
echo "      --latent_vae_path ${LATENT_VAE_PATH} \\"
echo "      --use_rep_alignment true \\"
echo "      --output_dir ${OUTPUT_DIR}"
