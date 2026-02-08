#!/usr/bin/env bash
# Stage 4c — Fine-tune unified DDBM at 512px
#
# Fine-tune the 256px base model (from Stage 4b) at 512px resolution using
# random crops from the three 1024px tasks:
#   - RGB→IR / SAR→IR / SAR→RGB:  random-crop from 1024px to 512px
#
# This checkpoint is the flagship model for the three 1024px competition tasks.
#
# DDBM large config (~404 M params).
#
# ⚠️ NOTE: in_channels / out_channels of the UNet should be 32 (latent dim)
#   rather than the default pixel channel count. This will be updated later.
#
# Usage:
#   bash scripts/train_stage4c_unified_512.sh
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage4c_unified_512.sh
#
# TODO: The unified multi-task training loop is not yet implemented.

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
RESOLUTION=512   # 1024px tasks are random-cropped to 512px

# --- Resume from 256px base model ---
RESUME_FROM="./ckpt/stage4b_unified_256"

OUTPUT_DIR="./ckpt/stage4c_unified_512"

echo "=== Stage 4c: Unified DDBM — 512px fine-tune ==="
echo "  UNet config : large (num_channels=${NUM_CHANNELS})"
echo "  Resolution  : ${RESOLUTION}px (random crop from 1024px)"
echo "  Resume from : ${RESUME_FROM} (256px base model)"
echo "  RS-VAE      : ${LATENT_VAE_PATH}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo ""
echo "  Tasks: RGB→IR, SAR→IR, SAR→RGB (crop 1024→512)"
echo "  Flagship for: RGB→IR, SAR→IR, SAR→RGB competition tasks"
echo ""
echo "⚠️  Unified multi-task training loop is not yet implemented."
echo "    Planned invocation (placeholder):"
echo "    python -m src.ddbm_baseline.train_unified \\"
echo "      --num_channels ${NUM_CHANNELS} \\"
echo "      --num_res_blocks ${NUM_RES_BLOCKS} \\"
echo "      --attention_resolutions ${ATTENTION_RESOLUTIONS} \\"
echo "      --channel_mult ${CHANNEL_MULT} \\"
echo "      --resolution ${RESOLUTION} \\"
echo "      --use_latent_target ${USE_LATENT_TARGET} \\"
echo "      --latent_vae_path ${LATENT_VAE_PATH} \\"
echo "      --use_rep_alignment true \\"
echo "      --resume_from_checkpoint ${RESUME_FROM} \\"
echo "      --output_dir ${OUTPUT_DIR}"
