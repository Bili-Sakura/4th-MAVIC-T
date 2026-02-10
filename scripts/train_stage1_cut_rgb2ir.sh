#!/usr/bin/env bash
# Stage 1 — CUT Ablation — RGB→IR — Pixel-space (large config, 512 crop)
# CUT large: ngf=128, ndf=128, n_layers_D=3 (~56.5 M, 256–512)
#
# Usage:
#   bash scripts/train_stage1_cut_rgb2ir.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_stage1_cut_rgb2ir.sh
#   NGPU=4 bash scripts/train_stage1_cut_rgb2ir.sh

set -euo pipefail

export HF_TOKEN="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"
export HF_ENDPOINT="https://hf-mirror.com"

NGPU="${NGPU:-1}"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/train_stage1_cut_rgb2ir.log"

mkdir -p "${LOG_DIR}"

# --- Large config (256–512) + 512 crop for faster iteration ---
NGF=128
NDF=128
N_LAYERS_D=3
NET_G="resnet_9blocks"
RESOLUTION=512

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
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=4
N_EPOCHS=0
N_EPOCHS_DECAY=0
MAX_TRAIN_STEPS=10000
GRADIENT_ACCUMULATION_STEPS=1
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=2000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS="1000"
VALIDATION_EPOCHS=""
PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt"
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

OUTPUT_DIR="./ckpt/stage1_cut_rgb2ir"
RESUME_FROM_CHECKPOINT="latest"

COMMON_ARGS=(
  --ngf "${NGF}"
  --ndf "${NDF}"
  --n_layers_D "${N_LAYERS_D}"
  --netG "${NET_G}"
  --resolution "${RESOLUTION}"
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
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --save_model_epochs "${SAVE_MODEL_EPOCHS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --push_to_hub "${PUSH_TO_HUB}"
  --hub_model_id "${HUB_MODEL_ID}"
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
  nohup accelerate launch --num_processes "${NGPU}" -m examples.cut.train_rgb2ir \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup python -m examples.cut.train_rgb2ir \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
