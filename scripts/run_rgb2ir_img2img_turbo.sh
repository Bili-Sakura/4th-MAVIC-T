#!/usr/bin/env bash
# Pix2Pix-Turbo — RGB->IR test-set inference for submission
#
# Usage:
#   CKPT_PATH=/path/to/model_final.pkl bash scripts/run_rgb2ir_img2img_turbo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_PATH="${CKPT_PATH:-}"
MODEL_NAME="${MODEL_NAME:-img2img_turbo_rgb2ir}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PROJECT_ROOT}/datasets/BiliSakura/MACIV-T-2025-Submissions}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICES="${DEVICES:-cuda:0}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Please set CKPT_PATH to the Pix2Pix-Turbo .pkl checkpoint."
  exit 1
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint file not found: ${CKPT_PATH}"
  exit 1
fi

read -r -a DEVICE_ARR <<< "${DEVICES}"
OUT_DIR="${SUBMISSION_ROOT}/rgb2ir/${MODEL_NAME}"

python -m examples.img2img_turbo.sample \
  --task rgb2ir \
  --model_path "${CKPT_PATH}" \
  --split test \
  --output_dir "${OUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE_ARR[@]}" \
  "$@"

echo "Generated files: ${OUT_DIR}"
