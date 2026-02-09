#!/usr/bin/env bash
# Stage 1 — SAR→EO — Latent-space modelling with frozen pre-trained VAE
# DDBM small config for latent shape (32, 32, 32)
#
# Usage:
#   bash scripts/train_stage1_sar2eo.sh
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage1_sar2eo.sh

set -euo pipefail

NGPU="${NGPU:-1}"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/train_stage1_sar2eo.log"

mkdir -p "${LOG_DIR}"

# --- Small config from configs/model_scaling_variants.yaml ---
NUM_CHANNELS=64
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS=""
CHANNEL_MULT="1,2,3,4"

# --- Latent-space settings ---
USE_LATENT_TARGET=true
LATENT_VAE_PATH="./models/BiliSakura/VAEs"  # FLUX2-VAE or SD21-VAE

# --- Representation alignment (REPA) ---
USE_REP_ALIGNMENT=true

# --- Data augmentation and filtering ---
USE_AUGMENTED=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=true
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"

# --- Training settings ---
OPTIMIZER_TYPE="prodigy"
USE_MAVIC_LOSS=false
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=16
NUM_EPOCHS=5
GRADIENT_ACCUMULATION_STEPS=1
USE_EMA=true
SAVE_MODEL_EPOCHS=1
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS="1000"
PUSH_TO_HUB=true
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

OUTPUT_DIR="./ckpt/stage1_sar2eo"

COMMON_ARGS=(
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --attention_resolutions "${ATTENTION_RESOLUTIONS}"
  --channel_mult "${CHANNEL_MULT}"
  --use_latent_target "${USE_LATENT_TARGET}"
  --latent_vae_path "${LATENT_VAE_PATH}"
  --use_rep_alignment "${USE_REP_ALIGNMENT}"
  --use_augmented "${USE_AUGMENTED}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --exclude_file "${EXCLUDE_FILE}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --use_mavic_loss "${USE_MAVIC_LOSS}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --use_ema "${USE_EMA}"
  --save_model_epochs "${SAVE_MODEL_EPOCHS}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --push_to_hub "${PUSH_TO_HUB}"
  --mixed_precision "${MIXED_PRECISION}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --seed "${SEED}"
  --output_dir "${OUTPUT_DIR}"
)

if [ -n "${VALIDATION_STEPS}" ]; then
  COMMON_ARGS+=(--validation_steps "${VALIDATION_STEPS}")
fi

if [ "${NGPU}" -gt 1 ]; then
  nohup accelerate launch --num_processes "${NGPU}" -m examples.ddbm.train_sar2eo \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup python -m examples.ddbm.train_sar2eo \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
