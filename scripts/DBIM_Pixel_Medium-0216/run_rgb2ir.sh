#!/usr/bin/env bash
# DBIM-Pixel-Medium-0216 — RGB->IR test-set inference for submission
#
# Default checkpoint:
#   /data/projects/models/hf_models/BiliSakura/4th-MAVIC-T-ckpt-0216/dbim/rgb2ir/checkpoint-100000
#
# Usage:
#   bash scripts/DBIM_Pixel_Medium-0216/run_rgb2ir.sh
#   DEVICES="cuda:0 cuda:1" bash scripts/DBIM_Pixel_Medium-0216/run_rgb2ir.sh
#   CKPT_PATH=/path/to/checkpoint bash scripts/DBIM_Pixel_Medium-0216/run_rgb2ir.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_PATH="${CKPT_PATH:-/data/projects/models/hf_models/BiliSakura/4th-MAVIC-T-ckpt-0216/dbim/rgb2ir/checkpoint-100000}"
MODEL_NAME="${MODEL_NAME:-dbim_pixel_medium_0216_rgb2ir}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PROJECT_ROOT}/datasets/BiliSakura/MACIV-T-2025-Submissions/ckpt-0216}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-100}"
SAMPLER="${SAMPLER:-dbim}"
RESOLUTION="${RESOLUTION:-1024}"
DEVICES="${DEVICES:-cuda:0}"

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_PATH}"
  exit 1
fi

read -r -a DEVICE_ARR <<< "${DEVICES}"

echo "=== Running DBIM rgb2ir test inference ==="
echo "checkpoint: ${CKPT_PATH}"
echo "output:     ${SUBMISSION_ROOT}/rgb2ir/${MODEL_NAME}"
echo "devices:    ${DEVICES}"

python -m examples.dbim.sample \
  --task rgb2ir \
  --pretrained_model_name_or_path "${CKPT_PATH}" \
  --split test \
  --model_name "${MODEL_NAME}" \
  --submission_root "${SUBMISSION_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_inference_steps "${NUM_STEPS}" \
  --sampler "${SAMPLER}" \
  --resolution "${RESOLUTION}" \
  --device "${DEVICE_ARR[@]}" \
  "$@"

echo "=== Done ==="
echo "Generated files: ${SUBMISSION_ROOT}/rgb2ir/${MODEL_NAME}"
echo "To package zip later: python scripts/create_submission_zip.py --submission_root \"${SUBMISSION_ROOT}\" --model_name \"${MODEL_NAME}\" --output \"${SUBMISSION_ROOT}/submission.zip\""
