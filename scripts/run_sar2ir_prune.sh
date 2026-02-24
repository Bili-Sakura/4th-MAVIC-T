#!/usr/bin/env bash
# Run full sar2ir dataset pruning pipeline: compute IR quality metrics, then prune.
#
# Output saved under datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/:
#   - ir_quality_metrics.csv
#   - pruned_sar2ir_exclude.txt
#
# Usage:
#   bash scripts/run_sar2ir_prune.sh
#   NUM_WORKERS=32 bash scripts/run_sar2ir_prune.sh   # default 32 data workers
#   MIN_VARIANCE=50 MIN_LAPLACIAN=10 MAX_CLIPPING=0.1 bash scripts/run_sar2ir_prune.sh
#   MERGE_FILE="datasets/.../manifests/bad_samples.txt" bash scripts/run_sar2ir_prune.sh
#
# Stricter: ~2–3x more exclusions
# MIN_VARIANCE=50 MIN_LAPLACIAN=10 MAX_CLIPPING=0.10 bash scripts/run_sar2ir_prune.sh

# Much stricter: ~5–10x more exclusions
# MIN_VARIANCE=100 MIN_LAPLACIAN=50 MAX_CLIPPING=0.05 bash scripts/run_sar2ir_prune.sh

# Very strict: keep only high-quality IR
# MIN_VARIANCE=200 MIN_LAPLACIAN=100 MAX_CLIPPING=0.03 bash scripts/run_sar2ir_prune.sh
# Use the exclude file in training:
#   EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/pruned_sar2ir_exclude.txt"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

REFINED_ROOT="${REFINED_ROOT:-datasets/BiliSakura/MACIV-T-2025-Structure-Refined}"
MANIFESTS_DIR="${PROJECT_ROOT}/${REFINED_ROOT}/manifests"
NUM_WORKERS="${NUM_WORKERS:-32}"
METRICS_OUT="${METRICS_OUT:-${MANIFESTS_DIR}/ir_quality_metrics.csv}"
EXCLUDE_OUT="${EXCLUDE_OUT:-${MANIFESTS_DIR}/pruned_sar2ir_exclude.txt}"
MERGE_FILE="${MERGE_FILE:-}"

echo "=== Step 1: Compute IR quality metrics ==="
python scripts/ir_quality_metrics.py \
  --refined_root "${REFINED_ROOT}" \
  --workers "${NUM_WORKERS}" \
  -o "${METRICS_OUT}"

echo ""
echo "=== Step 2: Prune low-quality samples ==="
PRUNE_ARGS=(
  --metrics "${METRICS_OUT}"
  --output "${EXCLUDE_OUT}"
)
if [[ -n "${MIN_VARIANCE:-}" ]]; then PRUNE_ARGS+=(--min_variance "${MIN_VARIANCE}"); fi
if [[ -n "${MIN_LAPLACIAN:-}" ]]; then PRUNE_ARGS+=(--min_laplacian "${MIN_LAPLACIAN}"); fi
if [[ -n "${MAX_CLIPPING:-}" ]]; then PRUNE_ARGS+=(--max_clipping "${MAX_CLIPPING}"); fi
if [[ -n "${MERGE_FILE}" ]]; then PRUNE_ARGS+=(--merge "${MERGE_FILE}"); fi

python scripts/prune_sar2ir_dataset.py "${PRUNE_ARGS[@]}"

echo ""
echo "Done. Exclude file: ${EXCLUDE_OUT}"
echo "Use with training: EXCLUDE_FILE=\"${EXCLUDE_OUT}\""
