#!/usr/bin/env bash
# Run inference-step ablation: 10-100 (step 10) + 100-1000 (step 100) on sample id 2.
# Output: /data/projects/4th-MAVIC-T/temp/ablation_steps_sample2_10-1000_full.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLE_ID="${SAMPLE_ID:-2}"
CKPT_PATH="${CKPT_PATH:-/data/projects/models/hf_models/BiliSakura/4th-MAVIC-T-ckpt-0216/dbim/rgb2ir/checkpoint-100000}"

echo "=== Ablation: steps 10-100 (step 10) + 100-1000 (step 100) on sample ${SAMPLE_ID} ==="
echo "Output: /data/projects/4th-MAVIC-T/temp/"

# Use rsgen python directly (avoids conda activate/deactivate script errors)
PYTHON="${RSGEN_PYTHON:-/data/miniconda3/envs/rsgen/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=python
fi

SAMPLE_ID="${SAMPLE_ID}" CKPT_PATH="${CKPT_PATH}" "${PYTHON}" scripts/DBIM_Pixel_Medium-0216/ablation_inference_steps.py
