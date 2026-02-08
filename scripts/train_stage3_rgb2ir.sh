#!/usr/bin/env bash
# Stage 3 — RGB→IR — Pixel-space modelling (no VAE)
# DDBM large config — same architecture tier as Stage 2
#
# Usage:
#   bash scripts/train_stage3_rgb2ir.sh
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage3_rgb2ir.sh

set -euo pipefail

NGPU="${NGPU:-1}"

# --- Large config from configs/model_scaling_variants.yaml ---
NUM_CHANNELS=256
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,2,4,4"

# --- Pixel-space (no latent VAE) ---
USE_LATENT_TARGET=false

# --- Representation alignment (REPA) ---
USE_REP_ALIGNMENT=true

OUTPUT_DIR="./ckpt/stage3_rgb2ir"

COMMON_ARGS=(
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --attention_resolutions "${ATTENTION_RESOLUTIONS}"
  --channel_mult "${CHANNEL_MULT}"
  --use_latent_target "${USE_LATENT_TARGET}"
  --use_rep_alignment "${USE_REP_ALIGNMENT}"
  --output_dir "${OUTPUT_DIR}"
)

if [ "${NGPU}" -gt 1 ]; then
  accelerate launch --num_processes "${NGPU}" -m src.ddbm_baseline.train_rgb2ir \
    "${COMMON_ARGS[@]}"
else
  python -m src.ddbm_baseline.train_rgb2ir \
    "${COMMON_ARGS[@]}"
fi
