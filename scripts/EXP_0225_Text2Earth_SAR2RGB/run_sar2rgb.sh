#!/usr/bin/env bash
# EXP-0225 — Text2Earth InstructPix2Pix SAR→RGB test-set inference
#
# Default checkpoint: checkpoint-20000
#
# Usage:
#   bash scripts/EXP_0225_Text2Earth_SAR2RGB/run_sar2rgb.sh
#   bash scripts/EXP_0225_Text2Earth_SAR2RGB/run_sar2rgb.sh --CHECKPOINT_PATH /path/to/checkpoint-30000
#   bash scripts/EXP_0225_Text2Earth_SAR2RGB/run_sar2rgb.sh --BATCH_SIZE 4 --NUM_STEPS 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_PATH="${CKPT_PATH:-/data/projects/4th-MAVIC-T/ckpt/EXP_0225_text2earth_sar2rgb_instructpix2pix/checkpoint-20000}"
MODEL_NAME="${MODEL_NAME:-text2earth_instructpix2pix_ckpt20000}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PROJECT_ROOT}/datasets/BiliSakura/MACIV-T-2025-Submissions}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_STEPS="${NUM_STEPS:-50}"
RESOLUTION="${RESOLUTION:-512}"
DEVICE="${DEVICE:-cuda:0}"

# Parse command-line overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --CHECKPOINT_PATH|--CKPT_PATH)
      CKPT_PATH="$2"
      shift 2
      ;;
    --MODEL_NAME)
      MODEL_NAME="$2"
      shift 2
      ;;
    --SUBMISSION_ROOT)
      SUBMISSION_ROOT="$2"
      shift 2
      ;;
    --SPLIT)
      SPLIT="$2"
      shift 2
      ;;
    --BATCH_SIZE)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --NUM_STEPS)
      NUM_STEPS="$2"
      shift 2
      ;;
    --RESOLUTION)
      RESOLUTION="$2"
      shift 2
      ;;
    --DEVICE)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_PATH}" >&2
  exit 1
fi

echo "=== Running Text2Earth InstructPix2Pix SAR2RGB test inference ==="
echo "checkpoint: ${CKPT_PATH}"
echo "output:     ${SUBMISSION_ROOT}/sar2rgb/${MODEL_NAME}"
echo "split:      ${SPLIT}"
echo "device:     ${DEVICE}"

python -m examples.text2earth_sar2rgb.sample_sar2rgb \
  --checkpoint_path "${CKPT_PATH}" \
  --split "${SPLIT}" \
  --model_name "${MODEL_NAME}" \
  --submission_root "${SUBMISSION_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_inference_steps "${NUM_STEPS}" \
  --resolution "${RESOLUTION}" \
  --device "${DEVICE}"

echo "=== Done ==="
echo "Generated files: ${SUBMISSION_ROOT}/sar2rgb/${MODEL_NAME}"
