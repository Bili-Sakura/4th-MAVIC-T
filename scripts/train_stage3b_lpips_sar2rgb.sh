#!/usr/bin/env bash
# Stage 3b — SAR→RGB — Pixel-space LPIPS/VGG finetuning (no VAE)
# Resume a strong Stage-3 pixel checkpoint and continue training with
# MAVIC loss (LPIPS[VGG] + L1) to reduce structural blurriness.
#
# Usage:
#   bash scripts/train_stage3b_lpips_sar2rgb.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_stage3b_lpips_sar2rgb.sh
#   NGPU=4 bash scripts/train_stage3b_lpips_sar2rgb.sh

set -euo pipefail

export HF_TOKEN="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"
export HF_ENDPOINT="https://hf-mirror.com"

NGPU="${NGPU:-1}"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/train_stage3b_lpips_sar2rgb.log"
mkdir -p "${LOG_DIR}"

# --- Match Stage-3 SAR→RGB pixel model architecture ---
NUM_CHANNELS=256
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,2,4,4"
USE_LATENT_TARGET=false

# --- Finetuning objective: LPIPS(VGG) + L1 ---
USE_MAVIC_LOSS=true
MAVIC_LPIPS_WEIGHT="${MAVIC_LPIPS_WEIGHT:-1.0}"
MAVIC_L1_WEIGHT="${MAVIC_L1_WEIGHT:-0.5}"
MAVIC_LOSS_WEIGHT="${MAVIC_LOSS_WEIGHT:-0.2}"

# --- Data ---
USE_AUGMENTED=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=true
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"

# --- Training schedule (short finetune) ---
OPTIMIZER_TYPE="prodigy"
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=4
NUM_EPOCHS=0
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
GRADIENT_ACCUMULATION_STEPS=1
USE_EMA=true
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=500
CHECKPOINTS_TOTAL_LIMIT=2
VALIDATION_STEPS=500
VALIDATION_EPOCHS=
PUSH_TO_HUB=true
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

# Keep output_dir on the Stage-3 run so "latest" resumes from your best pixel model.
OUTPUT_DIR="${OUTPUT_DIR:-./ckpt/stage3_sar2rgb}"
RESUME_FROM_CHECKPOINT="latest"

COMMON_ARGS=(
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --attention_resolutions "${ATTENTION_RESOLUTIONS}"
  --channel_mult "${CHANNEL_MULT}"
  --use_latent_target "${USE_LATENT_TARGET}"
  --use_augmented "${USE_AUGMENTED}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --exclude_file "${EXCLUDE_FILE}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --use_mavic_loss "${USE_MAVIC_LOSS}"
  --mavic_lpips_weight "${MAVIC_LPIPS_WEIGHT}"
  --mavic_l1_weight "${MAVIC_L1_WEIGHT}"
  --mavic_loss_weight "${MAVIC_LOSS_WEIGHT}"
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
  --validation_steps "${VALIDATION_STEPS}"
  --output_dir "${OUTPUT_DIR}"
  --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"
)

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
