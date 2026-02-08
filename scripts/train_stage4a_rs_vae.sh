#!/usr/bin/env bash
# Stage 4a — Fine-tune RS-VAE on remote-sensing data
#
# Initialise from SD21-VAE and fine-tune on the combined SAR+EO+RGB+IR dataset
# so the VAE can compress/reconstruct all remote-sensing modalities.
#
# The RS-VAE accepts 3-channel input. For single-channel modalities (SAR, EO, IR):
#   Encode: repeat 1-ch pixel → 3-ch input
#   Decode: average 3-ch output → 1-ch result
#
# Usage:
#   bash scripts/train_stage4a_rs_vae.sh
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage4a_rs_vae.sh
#
# TODO: This script is a placeholder. The RS-VAE fine-tuning pipeline
#   needs to be implemented (training loop, data loader for multi-modal data).

set -euo pipefail

NGPU="${NGPU:-1}"

# --- RS-VAE config ---
INIT_VAE_PATH="stabilityai/stable-diffusion-2-1"  # SD21-VAE init checkpoint
OUTPUT_DIR="./ckpt/stage4a_rs_vae"

echo "=== Stage 4a: Fine-tune RS-VAE ==="
echo "  Init VAE   : ${INIT_VAE_PATH}"
echo "  Output dir : ${OUTPUT_DIR}"
echo ""
echo "⚠️  RS-VAE fine-tuning pipeline is not yet implemented."
echo "    This script will be updated once the training loop is ready."
echo "    The RS-VAE should be trained on combined SAR+EO+RGB+IR data"
echo "    with 3-channel input (1-ch modalities use channel-repeat trick)."
