#!/usr/bin/env bash
# Stage 1 — SAR→EO — Latent-space modelling with frozen pre-trained VAE
# DDBM small config for latent shape (32, 32, 32)
#
# ⚠️ NOTE: in_channels / out_channels of the UNet should be 32 (latent dim)
#   rather than the default pixel channel count. This will be updated later.
#
# Usage:
#   bash scripts/train_stage1_sar2eo.sh
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage1_sar2eo.sh

set -euo pipefail

NGPU="${NGPU:-1}"

# --- Small config from configs/model_scaling_variants.yaml ---
NUM_CHANNELS=64
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS=""
CHANNEL_MULT="1,2,3,4"

# --- Latent-space settings ---
USE_LATENT_TARGET=true
LATENT_VAE_PATH="./models/BiliSakura/VAEs"  # FLUX2-VAE or SD21-VAE

OUTPUT_DIR="./ckpt/stage1_sar2eo"

CMD="python -m src.ddbm_baseline.train_sar2eo \
  --num_channels ${NUM_CHANNELS} \
  --num_res_blocks ${NUM_RES_BLOCKS} \
  --attention_resolutions ${ATTENTION_RESOLUTIONS} \
  --channel_mult ${CHANNEL_MULT} \
  --use_latent_target ${USE_LATENT_TARGET} \
  --latent_vae_path ${LATENT_VAE_PATH} \
  --output_dir ${OUTPUT_DIR}"

if [ "${NGPU}" -gt 1 ]; then
  accelerate launch --num_processes "${NGPU}" -m src.ddbm_baseline.train_sar2eo \
    --num_channels ${NUM_CHANNELS} \
    --num_res_blocks ${NUM_RES_BLOCKS} \
    --attention_resolutions "${ATTENTION_RESOLUTIONS}" \
    --channel_mult "${CHANNEL_MULT}" \
    --use_latent_target ${USE_LATENT_TARGET} \
    --latent_vae_path "${LATENT_VAE_PATH}" \
    --output_dir "${OUTPUT_DIR}"
else
  ${CMD}
fi
