#!/usr/bin/env bash
# Stage 2 — SAR→RGB — Latent-space modelling, scaled-up model
# DDBM large config for latent shape (128, 128, 32)
#
# Usage:
#   bash scripts/train_stage2_sar2rgb.sh
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage2_sar2rgb.sh

set -euo pipefail

NGPU="${NGPU:-1}"

# --- Large config from configs/model_scaling_variants.yaml ---
NUM_CHANNELS=256
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,2,4,4"

# --- Latent-space settings ---
USE_LATENT_TARGET=true
LATENT_VAE_PATH="./models/BiliSakura/VAEs"  # FLUX2-VAE or SD21-VAE

# --- Representation alignment (REPA) ---
USE_REP_ALIGNMENT=true

OUTPUT_DIR="./ckpt/stage2_sar2rgb"

COMMON_ARGS=(
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --attention_resolutions "${ATTENTION_RESOLUTIONS}"
  --channel_mult "${CHANNEL_MULT}"
  --use_latent_target "${USE_LATENT_TARGET}"
  --latent_vae_path "${LATENT_VAE_PATH}"
  --use_rep_alignment "${USE_REP_ALIGNMENT}"
  --output_dir "${OUTPUT_DIR}"
)

if [ "${NGPU}" -gt 1 ]; then
  accelerate launch --num_processes "${NGPU}" -m examples.ddbm.train_sar2rgb \
    "${COMMON_ARGS[@]}"
else
  python -m examples.ddbm.train_sar2rgb \
    "${COMMON_ARGS[@]}"
fi
