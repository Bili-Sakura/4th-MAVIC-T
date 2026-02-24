#!/usr/bin/env bash
# EXP-0225 — Text2Earth SAR→RGB via InstructPix2Pix-style conditioning (1 GPU)
#
# Usage:
#   bash scripts/EXP_0225_Text2Earth_SAR2RGB/train_sar2rgb_instructpix2pix_1gpu.sh

set -euo pipefail

export HF_TOKEN="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"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="MR3DpLBq2VJ01nXRIMh8f"

NGPU=1
LOG_DIR="./logs/EXP_0225_Text2Earth_SAR2RGB"
LOG_FILE="$LOG_DIR/train_sar2rgb_instructpix2pix_1gpu.log"
mkdir -p "$LOG_DIR"

PRETRAINED_MODEL="models/lcybuaa/Text2Earth"
OUTPUT_DIR="./ckpt/EXP_0225_text2earth_sar2rgb_instructpix2pix"
RESOLUTION=512  # random crop size (not resize); images must be >= 512x512

# --- Data / dataset ---
REFINED_ROOT=
SAR2RGB_SUP_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_sar2rgb_sup.txt"
PAIRED_VAL_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2rgb.txt"
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
USE_AUGMENTED=true
# Text2Earth: N_GOOGLE_LEVEL_ sets resolution 2^(17-N)m; 17→1m, 18→0.5m
CAPTION="18_GOOGLE_LEVEL_ a satellite optical image"

# --- Training ---
OPTIMIZER_TYPE=prodigy
LEARNING_RATE=1.0
PRODIGY_D0=1e-5
TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
MAX_TRAIN_STEPS=10000
CHECKPOINTING_STEPS=1000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS=1000
CONDITIONING_DROPOUT_PROB=0.05
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42
RESUME_FROM_CHECKPOINT=latest

# --- Representation alignment (REPA) ---
USE_REP_ALIGNMENT=true
LAMBDA_REP_ALIGNMENT=1.0
LAMBDA_REP_ALIGNMENT_DECAY_STEPS=1500
LAMBDA_REP_ALIGNMENT_END=0.0
REP_ALIGNMENT_MODEL_PATH="./models/BiliSakura/MaRS-Base-RGB"

# --- Hub & SwanLab
PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0225-text2earth-sar2rgb"
HUB_TOKEN="${HF_TOKEN:-}"
LOG_WITH="swanlab"
SWANLOG_DIR="./ckpt/swanlog"
SWANLAB_EXPERIMENT_NAME="exp-0225-text2earth-sar2rgb-instructpix2pix-1gpu"
SWANLAB_DESCRIPTION="EXP-0225 Text2Earth SAR→RGB InstructPix2Pix (1 GPU)"
SWANLAB_TAGS="text2earth,instructpix2pix,exp-0225,sar2rgb,1gpu"

COMMON_ARGS=(
  --pretrained_model_name_or_path "$PRETRAINED_MODEL"
  --output_dir "$OUTPUT_DIR"
  --resolution "$RESOLUTION"
  --optimizer_type "$OPTIMIZER_TYPE"
  --learning_rate "$LEARNING_RATE"
  --prodigy_d0 "$PRODIGY_D0"
  --sar2rgb_sup_manifest "$SAR2RGB_SUP_MANIFEST"
  --paired_val_manifest "$PAIRED_VAL_MANIFEST"
  --exclude_file "$EXCLUDE_FILE"
  --caption "$CAPTION"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --max_train_steps "$MAX_TRAIN_STEPS"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
  --validation_steps "$VALIDATION_STEPS"
  --conditioning_dropout_prob "$CONDITIONING_DROPOUT_PROB"
  --use_ema
  --mixed_precision "$MIXED_PRECISION"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --seed "$SEED"
  --enable_xformers_memory_efficient_attention
  --report_to "$LOG_WITH"
  --swanlab_experiment_name "$SWANLAB_EXPERIMENT_NAME"
  --swanlab_description "$SWANLAB_DESCRIPTION"
  --swanlab_tags "$SWANLAB_TAGS"
  --swanlab_init_kwargs_json '{"logdir":"'"$SWANLOG_DIR"'","workspace":"EarthBridge"}'
)

if [ -n "$REFINED_ROOT" ]; then
  COMMON_ARGS+=(--refined_root "$REFINED_ROOT")
fi
if [ "$USE_AUGMENTED" = "true" ]; then
  COMMON_ARGS+=(--use_augmented)
fi
if [ "$USE_REP_ALIGNMENT" = "true" ]; then
  COMMON_ARGS+=(
    --use_rep_alignment
    --rep_alignment_model_path "$REP_ALIGNMENT_MODEL_PATH"
    --lambda_rep_alignment "$LAMBDA_REP_ALIGNMENT"
    --lambda_rep_alignment_decay_steps "$LAMBDA_REP_ALIGNMENT_DECAY_STEPS"
    --lambda_rep_alignment_end "$LAMBDA_REP_ALIGNMENT_END"
  )
fi

if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  COMMON_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

if [ "$PUSH_TO_HUB" = "true" ]; then
  # Force official Hub endpoint for write operations.
  export HF_ENDPOINT="https://huggingface.co"
  export HUGGINGFACE_HUB_TOKEN="${HUB_TOKEN:-}"
  COMMON_ARGS+=(--push_to_hub --hub_model_id "$HUB_MODEL_ID")
  if [ -n "${HUB_TOKEN:-}" ]; then
    COMMON_ARGS+=(--hub_token "$HUB_TOKEN")
  fi
else
  # Mirror endpoint can still be used for model downloading.
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

if [ "$NGPU" -gt 1 ]; then
  nohup accelerate launch --num_processes "$NGPU" --mixed_precision "$MIXED_PRECISION" \
    -m examples.text2earth_sar2rgb.train_sar2rgb_instructpix2pix \
    "${COMMON_ARGS[@]}" > "$LOG_FILE" 2>&1 &
else
  nohup accelerate launch --mixed_precision "$MIXED_PRECISION" \
    -m examples.text2earth_sar2rgb.train_sar2rgb_instructpix2pix \
    "${COMMON_ARGS[@]}" > "$LOG_FILE" 2>&1 &
fi

echo "Training started. Log: $LOG_FILE"
