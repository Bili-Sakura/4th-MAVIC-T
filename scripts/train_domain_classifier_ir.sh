#!/usr/bin/env bash
# Train IR-domain real/fake classifier with domain-specific normalization stats.
#
# Pre-computes mean/std from the IR target domain dataset and uses them instead
# of ImageNet statistics. Run this script from the project root.
#
# Usage:
#   bash scripts/train_domain_classifier_ir.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_domain_classifier_ir.sh
#
# Options (set before running):
#   COMPUTE_STATS=true   - Recompute dataset stats (default: true if stats file missing)
#   SKIP_STATS=false     - Skip stats computation, use existing or ImageNet (set SKIP_STATS=true)

set -euo pipefail

LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/train_domain_classifier_ir.log"
STATS_PATH="./stats/ir_domain_stats.json"
OUTPUT_DIR="./ckpt/domain_classifier"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${STATS_PATH}")"

TARGET_DOMAIN="ir"
COMPUTE_STATS="${COMPUTE_STATS:-auto}"
SKIP_STATS="${SKIP_STATS:-false}"

# Compute dataset stats if not present (or if COMPUTE_STATS=true)
if [ "${SKIP_STATS}" = "true" ]; then
  echo "Skipping stats computation (SKIP_STATS=true)"
elif [ ! -f "${STATS_PATH}" ] || [ "${COMPUTE_STATS}" = "true" ]; then
  echo "Computing dataset stats for ${TARGET_DOMAIN} domain..."
  python -m examples.domain_classifier.compute_dataset_stats \
    --target_domain "${TARGET_DOMAIN}" \
    --output_path "${STATS_PATH}" 2>&1 | tee -a "${LOG_FILE}" || true
  echo "Stats saved to ${STATS_PATH}"
else
  echo "Using existing stats: ${STATS_PATH}"
fi

# Train (with domain stats if available)
TRAIN_ARGS=(
  --target_domain "${TARGET_DOMAIN}"
  --output_dir "${OUTPUT_DIR}"
  --train_batch_size 8
  --num_epochs 5
  --dataloader_num_workers 4
)
if [ -f "${STATS_PATH}" ]; then
  TRAIN_ARGS+=(--dataset_stats_path "${STATS_PATH}")
  echo "Training IR-domain classifier with dataset stats: ${STATS_PATH}"
else
  echo "Training IR-domain classifier (ImageNet normalization fallback)"
fi

python -m examples.domain_classifier.train_domain_classifier \
  "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

echo "Done. Checkpoint: ${OUTPUT_DIR}/${TARGET_DOMAIN}-resnet18-real-fake/"
