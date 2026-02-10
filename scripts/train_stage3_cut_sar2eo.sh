#!/usr/bin/env bash
# Stage 3 — CUT Ablation — SAR→EO — Pixel-space (huge config, aggressive)
# CUT huge: ngf=256, ndf=256, n_layers_D=4 (~293 M, 512–1024)
#
# Usage:
#   bash scripts/train_stage3_cut_sar2eo.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_stage3_cut_sar2eo.sh
#   NGPU=4 bash scripts/train_stage3_cut_sar2eo.sh

set -euo pipefail

export HF_TOKEN="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"
export HF_ENDPOINT="https://hf-mirror.com"

NGPU="${NGPU:-1}"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/train_stage3_cut_sar2eo.log"

mkdir -p "${LOG_DIR}"

# --- Huge config (aggressive for 256 px) from configs/model_scaling_variants.yaml ---
NGF=256
NDF=256
N_LAYERS_D=4
NET_G="resnet_9blocks"

# --- CUT is always pixel-space (no VAE) ---
USE_LATENT_TARGET=false

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
N_EPOCHS=5
N_EPOCHS_DECAY=0
GRADIENT_ACCUMULATION_STEPS=1
SAVE_MODEL_EPOCHS=1
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS="1000"
VALIDATION_EPOCHS=""
PUSH_TO_HUB=true
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

OUTPUT_DIR="./ckpt/stage3_cut_sar2eo"

COMMON_ARGS=(
  --ngf "${NGF}"
  --ndf "${NDF}"
  --n_layers_D "${N_LAYERS_D}"
  --netG "${NET_G}"
  --use_latent_target "${USE_LATENT_TARGET}"
  --use_augmented "${USE_AUGMENTED}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --exclude_file "${EXCLUDE_FILE}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --use_mavic_loss "${USE_MAVIC_LOSS}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --n_epochs "${N_EPOCHS}"
  --n_epochs_decay "${N_EPOCHS_DECAY}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
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
if [ -n "${VALIDATION_EPOCHS}" ]; then
  COMMON_ARGS+=(--validation_epochs "${VALIDATION_EPOCHS}")
fi

if [ "${NGPU}" -gt 1 ]; then
  nohup accelerate launch --num_processes "${NGPU}" -m examples.cut.train_sar2eo \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup python -m examples.cut.train_sar2eo \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
