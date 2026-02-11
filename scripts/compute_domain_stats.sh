#!/usr/bin/env bash
# Pre-compute mean and std for domain datasets (IR, EO, RGB).
#
# Run this before training to generate stats files. Training scripts will
# auto-compute if stats are missing, but this script lets you compute all
# domains at once or with custom options.
#
# Usage:
#   bash scripts/compute_domain_stats.sh              # IR only (default)
#   bash scripts/compute_domain_stats.sh ir eo rgb    # All domains
#   DOMAIN=eo bash scripts/compute_domain_stats.sh     # Single domain via env

set -euo pipefail

STATS_DIR="./stats"
mkdir -p "${STATS_DIR}"

# Domains to compute: from args or env DOMAIN or default "ir"
if [ $# -gt 0 ]; then
  DOMAINS=("$@")
elif [ -n "${DOMAIN:-}" ]; then
  DOMAINS=("${DOMAIN}")
else
  DOMAINS=("ir")
fi

for d in "${DOMAINS[@]}"; do
  echo "Computing stats for ${d} domain..."
  python -m examples.domain_classifier.compute_dataset_stats \
    --target_domain "${d}" \
    --output_path "${STATS_DIR}/${d}_domain_stats.json"
  echo ""
done

echo "All stats saved to ${STATS_DIR}/"
