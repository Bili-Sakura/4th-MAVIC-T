#!/usr/bin/env bash
# DBIM-Pixel-Medium-0216 — Evaluate all four tasks on paired_val manifests (LPIPS, L1, FID).
# Reports score for each task: sar2ir, sar2eo, sar2rgb, rgb2ir.
#
# Default checkpoints (override with CKPT_ROOT or per-task CKPT_<TASK>):
#   CKPT_ROOT/dbim/{sar2ir,sar2eo,sar2rgb,rgb2ir}/checkpoint-100000
#
# Usage:
#   bash scripts/DBIM_Pixel_Medium-0216/eval_all.sh
#   CKPT_ROOT=/path/to/ckpt bash scripts/DBIM_Pixel_Medium-0216/eval_all.sh
#   MANIFEST_DIR=path/to/manifests bash scripts/DBIM_Pixel_Medium-0216/eval_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_ROOT="${CKPT_ROOT:-/data/projects/models/hf_models/BiliSakura/4th-MAVIC-T-ckpt-0216/dbim}"
MANIFEST_DIR="${MANIFEST_DIR:-datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_STEPS="${NUM_STEPS:-100}"
DEVICE="${DEVICE:-cuda:0}"

# Task config: task_name | resolution
# sar2eo uses 256; others use 1024
eval_sar2ir() {
  local res="${RESOLUTION_SAR2IR:-1024}"
  local ckpt="${CKPT_SAR2IR:-${CKPT_ROOT}/sar2ir/checkpoint-100000}"
  local manifest="${MANIFEST_DIR}/paired_val_sar2ir.txt"
  if [[ ! -d "${ckpt}" ]]; then echo "  [skip] checkpoint not found: ${ckpt}"; return 0; fi
  echo "--- sar2ir (resolution=${res}) ---"
  python -m examples.dbim.evaluate_metrics \
    --checkpoint_dir "${ckpt}" \
    --manifest "${manifest}" \
    --task sar2ir \
    --batch_size "${BATCH_SIZE}" \
    --num_inference_steps "${NUM_STEPS}" \
    --resolution "${res}" \
    --device "${DEVICE}" \
    "$@"
}

eval_sar2eo() {
  local res="${RESOLUTION_SAR2EO:-256}"
  local ckpt="${CKPT_SAR2EO:-${CKPT_ROOT}/sar2eo/checkpoint-100000}"
  local manifest="${MANIFEST_DIR}/paired_val_sar2eo.txt"
  if [[ ! -d "${ckpt}" ]]; then echo "  [skip] checkpoint not found: ${ckpt}"; return 0; fi
  echo "--- sar2eo (resolution=${res}) ---"
  python -m examples.dbim.evaluate_metrics \
    --checkpoint_dir "${ckpt}" \
    --manifest "${manifest}" \
    --task sar2eo \
    --batch_size "${BATCH_SIZE}" \
    --num_inference_steps "${NUM_STEPS}" \
    --resolution "${res}" \
    --device "${DEVICE}" \
    "$@"
}

eval_sar2rgb() {
  local res="${RESOLUTION_SAR2RGB:-1024}"
  local ckpt="${CKPT_SAR2RGB:-${CKPT_ROOT}/sar2rgb/checkpoint-100000}"
  local manifest="${MANIFEST_DIR}/paired_val_sar2rgb.txt"
  if [[ ! -d "${ckpt}" ]]; then echo "  [skip] checkpoint not found: ${ckpt}"; return 0; fi
  echo "--- sar2rgb (resolution=${res}) ---"
  python -m examples.dbim.evaluate_metrics \
    --checkpoint_dir "${ckpt}" \
    --manifest "${manifest}" \
    --task sar2rgb \
    --batch_size "${BATCH_SIZE}" \
    --num_inference_steps "${NUM_STEPS}" \
    --resolution "${res}" \
    --device "${DEVICE}" \
    "$@"
}

eval_rgb2ir() {
  local res="${RESOLUTION_RGB2IR:-1024}"
  local ckpt="${CKPT_RGB2IR:-${CKPT_ROOT}/rgb2ir/checkpoint-100000}"
  local manifest="${MANIFEST_DIR}/paired_val_rgb2ir.txt"
  if [[ ! -d "${ckpt}" ]]; then echo "  [skip] checkpoint not found: ${ckpt}"; return 0; fi
  echo "--- rgb2ir (resolution=${res}) ---"
  python -m examples.dbim.evaluate_metrics \
    --checkpoint_dir "${ckpt}" \
    --manifest "${manifest}" \
    --task rgb2ir \
    --batch_size "${BATCH_SIZE}" \
    --num_inference_steps "${NUM_STEPS}" \
    --resolution "${res}" \
    --device "${DEVICE}" \
    "$@"
}

echo "=== Evaluating DBIM on all four tasks ==="
echo "CKPT_ROOT:   ${CKPT_ROOT}"
echo "MANIFEST_DIR: ${MANIFEST_DIR}"
echo "BATCH_SIZE:  ${BATCH_SIZE}"
echo "NUM_STEPS:   ${NUM_STEPS}"
echo "DEVICE:      ${DEVICE}"
echo ""

eval_sar2ir
echo ""

eval_sar2eo
echo ""

eval_sar2rgb
echo ""

eval_rgb2ir

echo "=== Done (all tasks) ==="
