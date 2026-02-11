#!/usr/bin/env bash
# Run DDBM inference on all 4 MAVIC-T test tasks and create submission.zip.
#
# Usage:
#   bash scripts/sample_all_tasks_ddbm.sh
#   # Custom checkpoint base and model name:
#   CKPT_BASE=./ckpt/exp3 MODEL_NAME=ddbm bash scripts/sample_all_tasks_ddbm.sh
#
# Output:
#   datasets/BiliSakura/MACIV-T-2025-Submissions/<task>/<model_name>/*.png
#   submission.zip (after all tasks complete)

set -euo pipefail

# Load paths from paths.env (PROJECT_ROOT + derived paths)
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR}/../paths.env" ]]; then
  set -a && source "${_SCRIPT_DIR}/../paths.env" && set +a
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
CKPT_BASE="${CKPT_BASE:-${PROJECT_ROOT}/ckpt/exp3}"
MODEL_NAME="${MODEL_NAME:-ddbm}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_STEPS="${NUM_STEPS:-1000}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PROJECT_ROOT}/datasets/BiliSakura/MACIV-T-2025-Submissions}"

for task in sar2eo sar2rgb rgb2ir sar2ir; do
  stage="stage1_${task}"
  ckpt="${CKPT_BASE}/${stage}/ddbm/${task}/checkpoint-10000"
  if [[ ! -d "${ckpt}" ]]; then
    # Try latest checkpoint
    ckpt_dir="${CKPT_BASE}/${stage}/ddbm/${task}"
    if [[ -d "${ckpt_dir}" ]]; then
      latest=$(ls -d "${ckpt_dir}"/checkpoint-* 2>/dev/null | tail -1)
      ckpt="${latest:-${ckpt}}"
    fi
  fi
  if [[ ! -d "${ckpt}" ]]; then
    echo "Skipping ${task}: checkpoint not found at ${ckpt}"
    continue
  fi
  echo "=== Running ${task} with ${ckpt} ==="
  python -m examples.ddbm.sample \
    --task "${task}" \
    --pretrained_model_name_or_path "${ckpt}" \
    --split test \
    --model_name "${MODEL_NAME}" \
    --submission_root "${SUBMISSION_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_inference_steps "${NUM_STEPS}"
done

echo "=== Creating submission.zip ==="
python scripts/create_submission_zip.py \
  --submission_root "${SUBMISSION_ROOT}" \
  --model_name "${MODEL_NAME}" \
  --output "${SUBMISSION_ROOT}/submission.zip"
