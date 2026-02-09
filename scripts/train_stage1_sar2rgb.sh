#!/usr/bin/env bash
# Stage 1 — SAR→RGB — Latent-space modelling with frozen pre-trained VAE
# DDBM medium config for latent shape (128, 128, 32)
#
# Usage:
#   bash scripts/train_stage1_sar2rgb.sh
#   # run on a specific GPU (e.g., cuda:0):
#   CUDA_VISIBLE_DEVICES=3 bash scripts/train_stage1_sar2rgb.sh
#   # (pick 0-3 to spread stages across 4 GPUs)
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage1_sar2rgb.sh

set -euo pipefail

export HF_TOKEN="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"
export HF_ENDPOINT="https://hf-mirror.com"

NGPU="${NGPU:-1}"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/train_stage1_sar2rgb.log"

mkdir -p "${LOG_DIR}"

# --- Medium config from configs/model_scaling_variants.yaml ---
NUM_CHANNELS=128
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,1,2,2,4,4"

# --- Latent-space settings ---
USE_LATENT_TARGET=true
LATENT_VAE_PATH="./models/BiliSakura/VAEs/FLUX2-VAE"  # FLUX2-VAE

# --- Representation alignment (REPA) ---
USE_REP_ALIGNMENT=true
LAMBDA_REP_ALIGNMENT=0.1

# --- Data augmentation and filtering ---
USE_AUGMENTED=false
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=true
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"

# --- Training settings ---
OPTIMIZER_TYPE="prodigy"
USE_MAVIC_LOSS=false
TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=2
NUM_EPOCHS=2
MAX_TRAIN_STEPS=10000
GRADIENT_ACCUMULATION_STEPS=1
USE_EMA=true
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=1000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS="1000"
VALIDATION_EPOCHS=""
PUSH_TO_HUB=true
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

OUTPUT_DIR="./ckpt/stage1_sar2rgb"

# --- Resume from checkpoint ---
RESUME_FROM_CHECKPOINT="checkpoint-3000"

COMMON_ARGS=(
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --attention_resolutions "${ATTENTION_RESOLUTIONS}"
  --channel_mult "${CHANNEL_MULT}"
  --use_latent_target "${USE_LATENT_TARGET}"
  --latent_vae_path "${LATENT_VAE_PATH}"
  --use_rep_alignment "${USE_REP_ALIGNMENT}"
  --lambda_rep_alignment "${LAMBDA_REP_ALIGNMENT}"
  --use_augmented "${USE_AUGMENTED}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --exclude_file "${EXCLUDE_FILE}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --use_mavic_loss "${USE_MAVIC_LOSS}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --use_ema "${USE_EMA}"
  --save_model_epochs "${SAVE_MODEL_EPOCHS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --push_to_hub "${PUSH_TO_HUB}"
  --mixed_precision "${MIXED_PRECISION}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --seed "${SEED}"
  --output_dir "${OUTPUT_DIR}"
  --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"
)

if [ -n "${VALIDATION_STEPS}" ]; then
  COMMON_ARGS+=(--validation_steps "${VALIDATION_STEPS}")
fi
if [ -n "${VALIDATION_EPOCHS}" ]; then
  COMMON_ARGS+=(--validation_epochs "${VALIDATION_EPOCHS}")
fi

if [ "${NGPU}" -gt 1 ]; then
  nohup accelerate launch --num_processes "${NGPU}" -m examples.ddbm.train_sar2rgb \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup python -m examples.ddbm.train_sar2rgb \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
