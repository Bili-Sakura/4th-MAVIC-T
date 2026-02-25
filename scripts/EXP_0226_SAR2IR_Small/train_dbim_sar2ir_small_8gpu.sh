#!/usr/bin/env bash
# EXP-0226 — DBIM — SAR→IR — Small tier (match SAR2EO success config)
#
# Recovery from failed EXP-0221 (SAR-lite medium, 512px). Adopts small tier from
# models/BiliSakura/4th-MAVIC-T-ckpt-0216/dbim/sar2eo/checkpoint-100000:
#   num_channels=64, channel_mult="1,2,3,4", attention_resolutions="", ~20M params.
#
# Usage:
#   bash scripts/EXP_0226_SAR2IR_Small/train_dbim_sar2ir_small_8gpu.sh
#   NGPU=8 bash scripts/EXP_0226_SAR2IR_Small/train_dbim_sar2ir_small_8gpu.sh

set -euo pipefail

export HF_TOKEN="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"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="MR3DpLBq2VJ01nXRIMh8f"

NGPU="${NGPU:-8}"
LOG_DIR="./logs/EXP_0226_SAR2IR_Small"
LOG_FILE="${LOG_DIR}/train_dbim_sar2ir_small_8gpu.log"
mkdir -p "${LOG_DIR}"

# --- Small config for 512px (5 stages: 512→16; extend SAR2EO "1,2,3,4" with 5th) ---
NUM_CHANNELS=64
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS=""
CHANNEL_MULT="1,2,3,4,4"

# --- Data / resolution (512 for 1024px task; crop from 1024) ---
RESOLUTION=512
OUTPUT_RESOLUTION=1024
USE_RANDOM_CROP=true
USE_AUGMENTED=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=true
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
PAIRED_VAL_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2ir.txt"

# --- Training (match SAR2EO: prodigy, constant LR, 100k steps) ---
OPTIMIZER_TYPE="prodigy"
LEARNING_RATE=1.0
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=500
PRODIGY_D0=1e-5
TRAIN_BATCH_SIZE=32
MAX_TRAIN_STEPS=100000
NUM_EPOCHS=0
GRADIENT_ACCUMULATION_STEPS=1
USE_EMA=true
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=10000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS=10000
NUM_INFERENCE_STEPS=100
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0226"
OUTPUT_DIR="./ckpt/EXP_0226_SAR2IR_Small/dbim/sar2ir_small_512"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# --- SwanLab ---
SWANLOG_DIR="./ckpt/swanlog"
SWANLAB_EXPERIMENT_NAME="exp-0226-sar2ir-small-512"
SWANLAB_DESCRIPTION="EXP-0226 DBIM SAR→IR Small (512, ~29.6M, 5-stage for 512px)"
SWANLAB_TAGS="dbim,exp-0226,sar2ir,small"

# --- Pixel-space DBIM (no VAE) ---
USE_LATENT_TARGET=false
USE_REP_ALIGNMENT=false
LAMBDA_REP_ALIGNMENT=0.1
USE_MAVIC_LOSS=false
SAMPLER="dbim"

COMMON_ARGS=(
  --log_with swanlab
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}"
  --swanlab_description "${SWANLAB_DESCRIPTION}"
  --swanlab_tags "${SWANLAB_TAGS}"
  --swanlab_init_kwargs_json '{"logdir":"'"${SWANLOG_DIR}"'","workspace":"EarthBridge"}'
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --attention_resolutions "${ATTENTION_RESOLUTIONS}"
  --channel_mult "${CHANNEL_MULT}"
  --resolution "${RESOLUTION}"
  --output_resolution "${OUTPUT_RESOLUTION}"
  --use_random_crop "${USE_RANDOM_CROP}"
  --use_augmented "${USE_AUGMENTED}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --exclude_file "${EXCLUDE_FILE}"
  --paired_val_manifest "${PAIRED_VAL_MANIFEST}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --prodigy_d0 "${PRODIGY_D0}"
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
  --sampler "${SAMPLER}"
)

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  COMMON_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [ "${NGPU}" -gt 1 ]; then
  nohup accelerate launch --num_processes "${NGPU}" -m examples.dbim.train_sar2ir \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup python -m examples.dbim.train_sar2ir \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
