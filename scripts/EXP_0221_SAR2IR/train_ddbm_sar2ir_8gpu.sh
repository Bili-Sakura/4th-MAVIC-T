#!/usr/bin/env bash
# EXP-0221 — DDBM — SAR→IR — runtime random-crop 1024->512
#
# Usage:
#   bash scripts/EXP_0221_SAR2IR/train_ddbm_sar2ir_8gpu.sh
#   NGPU=8 bash scripts/EXP_0221_SAR2IR/train_ddbm_sar2ir_8gpu.sh

set -euo pipefail

export HF_TOKEN="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"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="MR3DpLBq2VJ01nXRIMh8f"

NGPU="${NGPU:-8}"
LOG_DIR="./logs/EXP_0221_SAR2IR"
LOG_FILE="${LOG_DIR}/train_ddbm_sar2ir_8gpu.log"
mkdir -p "${LOG_DIR}"

# --- Model config (SAR-lite medium tier for 512 crop training) ---
NUM_CHANNELS=96
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,1,2,2,4,4"

# --- Data / resolution ---
RESOLUTION=512
OUTPUT_RESOLUTION=1024
USE_AUGMENTED=true
USE_RANDOM_CROP=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=true
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
PAIRED_VAL_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2ir.txt"

# --- Training ---
OPTIMIZER_TYPE="prodigy"
TRAIN_BATCH_SIZE=8
MAX_TRAIN_STEPS=120000
NUM_EPOCHS=0
GRADIENT_ACCUMULATION_STEPS=1
USE_EMA=true
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=10000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS=5000
NUM_INFERENCE_STEPS=100
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0221"
OUTPUT_DIR="./ckpt/EXP_0221_SAR2IR/ddbm/sar2ir_512"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# --- Optional extras ---
USE_LATENT_TARGET=false
USE_REP_ALIGNMENT=false
LAMBDA_REP_ALIGNMENT=0.1
USE_MAVIC_LOSS=false

COMMON_ARGS=(
  --log_with tensorboard
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --attention_resolutions "${ATTENTION_RESOLUTIONS}"
  --channel_mult "${CHANNEL_MULT}"
  --resolution "${RESOLUTION}"
  --output_resolution "${OUTPUT_RESOLUTION}"
  --use_augmented "${USE_AUGMENTED}"
  --use_random_crop "${USE_RANDOM_CROP}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --exclude_file "${EXCLUDE_FILE}"
  --paired_val_manifest "${PAIRED_VAL_MANIFEST}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --num_epochs "${NUM_EPOCHS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --use_ema "${USE_EMA}"
  --save_model_epochs "${SAVE_MODEL_EPOCHS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --validation_steps "${VALIDATION_STEPS}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --mixed_precision "${MIXED_PRECISION}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --seed "${SEED}"
  --push_to_hub "${PUSH_TO_HUB}"
  --hub_model_id "${HUB_MODEL_ID}"
  --output_dir "${OUTPUT_DIR}"
  --use_latent_target "${USE_LATENT_TARGET}"
  --use_rep_alignment "${USE_REP_ALIGNMENT}"
  --lambda_rep_alignment "${LAMBDA_REP_ALIGNMENT}"
  --use_mavic_loss "${USE_MAVIC_LOSS}"
)

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  COMMON_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [ "${NGPU}" -gt 1 ]; then
  nohup accelerate launch --num_processes "${NGPU}" -m examples.ddbm.train_sar2ir \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup python -m examples.ddbm.train_sar2ir \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
