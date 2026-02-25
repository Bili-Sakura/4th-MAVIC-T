#!/usr/bin/env bash
# I2SB — RGB->IR test-set inference for submission
#
# Usage:
#   CKPT_PATH=/path/to/checkpoint bash scripts/run_rgb2ir_i2sb.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_PATH="${CKPT_PATH:-}"
MODEL_NAME="${MODEL_NAME:-i2sb_rgb2ir}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PROJECT_ROOT}/datasets/BiliSakura/MACIV-T-2025-Submissions}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NFE="${NFE:-100}"
DEVICES="${DEVICES:-cuda:0}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Please set CKPT_PATH to the I2SB checkpoint directory."
  exit 1
fi
if [[ ! -d "${CKPT_PATH}" && ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint path not found: ${CKPT_PATH}"
  exit 1
fi

read -r -a DEVICE_ARR <<< "${DEVICES}"
OUT_DIR="${SUBMISSION_ROOT}/rgb2ir/${MODEL_NAME}"

python -m examples.i2sb.sample \
  --task rgb2ir \
  --pretrained_model_name_or_path "${CKPT_PATH}" \
  --split test \
  --output_dir "${OUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --nfe "${NFE}" \
  --device "${DEVICE_ARR[@]}" \
  "$@"

echo "Generated files: ${OUT_DIR}"
