#!/usr/bin/env bash
# DDIB — RGB->IR test-set inference for submission
#
# Usage:
#   CKPT_PATH=/path/to/ddib_pipeline bash scripts/run_rgb2ir_ddib.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_PATH="${CKPT_PATH:-}"
MODEL_NAME="${MODEL_NAME:-ddib_rgb2ir}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PROJECT_ROOT}/datasets/BiliSakura/MACIV-T-2025-Submissions}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_STEPS="${NUM_STEPS:-250}"
DEVICES="${DEVICES:-cuda:0}"
ETA="${ETA:-0.0}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Please set CKPT_PATH to a DDIB combined pipeline checkpoint directory."
  exit 1
fi
if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_PATH}"
  exit 1
fi

read -r -a DEVICE_ARR <<< "${DEVICES}"
OUT_DIR="${SUBMISSION_ROOT}/rgb2ir/${MODEL_NAME}"

python -m examples.ddib.sample \
  --task rgb2ir \
  --pretrained_model_name_or_path "${CKPT_PATH}" \
  --split test \
  --output_dir "${OUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_inference_steps "${NUM_STEPS}" \
  --eta "${ETA}" \
  --device "${DEVICE_ARR[@]}" \
  "$@"

echo "Generated files: ${OUT_DIR}"
