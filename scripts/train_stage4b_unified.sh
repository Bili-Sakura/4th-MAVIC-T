#!/usr/bin/env bash
# Stage 4b — Unified DDBM training with RS-VAE
#
# Train a single DDBM model on all 4 tasks (RGB→IR, SAR→IR, SAR→RGB, SAR→EO)
# simultaneously, using the RS-VAE (from Stage 4a) to encode all modalities
# into a shared latent space.
#
# DDBM large config (~404 M params).
#
# ⚠️ NOTE: in_channels / out_channels of the UNet should be 32 (latent dim)
#   rather than the default pixel channel count. This will be updated later.
#
# Usage:
#   bash scripts/train_stage4b_unified.sh
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage4b_unified.sh
#
# TODO: The unified multi-task training loop is not yet implemented.
#   This script will be updated once the data loader supports mixed-task
#   batches and the RS-VAE checkpoint is available.

set -euo pipefail

NGPU="${NGPU:-1}"

# --- Large config from configs/model_scaling_variants.yaml ---
NUM_CHANNELS=256
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,2,4,4"

# --- Latent-space settings (RS-VAE from Stage 4a) ---
USE_LATENT_TARGET=true
LATENT_VAE_PATH="./ckpt/stage4a_rs_vae"  # RS-VAE fine-tuned on RS data

OUTPUT_DIR="./ckpt/stage4b_unified"

echo "=== Stage 4b: Unified DDBM (all 4 tasks) ==="
echo "  UNet config : large (num_channels=${NUM_CHANNELS})"
echo "  RS-VAE      : ${LATENT_VAE_PATH}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo ""
echo "⚠️  Unified multi-task training loop is not yet implemented."
echo "    This script will be updated once the mixed-task data loader"
echo "    and RS-VAE checkpoint are available."
echo ""
echo "    Planned invocation (placeholder):"
echo "    python -m src.ddbm_baseline.train_unified \\"
echo "      --num_channels ${NUM_CHANNELS} \\"
echo "      --num_res_blocks ${NUM_RES_BLOCKS} \\"
echo "      --attention_resolutions ${ATTENTION_RESOLUTIONS} \\"
echo "      --channel_mult ${CHANNEL_MULT} \\"
echo "      --use_latent_target ${USE_LATENT_TARGET} \\"
echo "      --latent_vae_path ${LATENT_VAE_PATH} \\"
echo "      --output_dir ${OUTPUT_DIR}"
