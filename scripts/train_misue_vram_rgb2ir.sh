#!/usr/bin/env bash
# misue-vram — RGB→IR — Latent-space modelling (frozen VAE), small config
# DDBM medium config for reduced VRAM usage
# Resume from ckpt/misue-vram with RESUME_FROM_CHECKPOINT
#
# Usage:
#   bash scripts/train_misue_vram_rgb2ir.sh
#   RESUME_FROM_CHECKPOINT="$(pwd)/ckpt/misue-vram/misue-vram_rgb2ir/ddbm/rgb2ir/checkpoint-7000" bash scripts/train_misue_vram_rgb2ir.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_misue_vram_rgb2ir.sh
#   NGPU=4 bash scripts/train_misue_vram_rgb2ir.sh

set -euo pipefail

export HF_TOKEN="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"
export HF_ENDPOINT="https://hf-mirror.com"

NGPU="${NGPU:-1}"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/train_misue_vram_rgb2ir.log"

mkdir -p "${LOG_DIR}"

NUM_CHANNELS=128
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,1,2,2,4,4"

USE_LATENT_TARGET=true
LATENT_VAE_PATH="./models/BiliSakura/VAEs/FLUX2-VAE"

USE_REP_ALIGNMENT=true
LAMBDA_REP_ALIGNMENT=0.1

USE_AUGMENTED=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=true
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"

OPTIMIZER_TYPE="prodigy"
USE_MAVIC_LOSS=false
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=4
NUM_EPOCHS=5
GRADIENT_ACCUMULATION_STEPS=1
USE_EMA=true
SAVE_MODEL_EPOCHS=1
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS="1000"
VALIDATION_EPOCHS=""
PUSH_TO_HUB=true
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

OUTPUT_DIR="./ckpt/misue-vram/misue-vram_rgb2ir"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"

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
if [ -n "${VALIDATION_EPOCHS}" ]; then
  COMMON_ARGS+=(--validation_epochs "${VALIDATION_EPOCHS}")
fi
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  COMMON_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [ "${NGPU}" -gt 1 ]; then
  nohup accelerate launch --num_processes "${NGPU}" -m examples.ddbm.train_rgb2ir \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup python -m examples.ddbm.train_rgb2ir \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
