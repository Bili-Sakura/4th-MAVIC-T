#!/usr/bin/env bash
# Stage 4d — Fine-tune unified DDBM at 1024px (optional)
#
# Further fine-tune the 512px model (from Stage 4c) at full 1024px resolution
# on the three 1024px tasks:
#   - RGB→IR / SAR→IR / SAR→RGB:  native 1024×1024 images (no crop)
#
# Run this step only if sufficient time and compute are available.
#
# DDBM large config (~404 M params).
#
# Usage:
#   bash scripts/train_stage4d_unified_1024.sh
#   # run on a specific GPU (e.g., cuda:0):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_stage4d_unified_1024.sh
#   # (pick 0-3 to spread stages across 4 GPUs)
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage4d_unified_1024.sh
#
# TODO: The unified multi-task training loop is not yet implemented.

set -euo pipefail

export HF_TOKEN="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"
export HF_ENDPOINT="https://hf-mirror.com"

NGPU="${NGPU:-1}"

# --- Large config from configs/model_scaling_variants.yaml ---
NUM_CHANNELS=256
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,2,4,4"

# --- Latent-space settings (RS-VAE from external repo) ---
USE_LATENT_TARGET=true
LATENT_VAE_PATH="./models/rs_vae"

# --- Representation alignment (REPA) ---
LAMBDA_REP_ALIGNMENT=0.1

# --- Resolution ---
RESOLUTION=1024  # native 1024px (no crop)

# --- Resume from 512px fine-tuned model ---
RESUME_FROM="./ckpt/stage4c_unified_512"

OUTPUT_DIR="./ckpt/stage4d_unified_1024"

echo "=== Stage 4d: Unified DDBM — 1024px fine-tune (optional) ==="
echo "  UNet config : large (num_channels=${NUM_CHANNELS})"
echo "  Resolution  : ${RESOLUTION}px (native, no crop)"
echo "  Resume from : ${RESUME_FROM} (512px fine-tuned model)"
echo "  RS-VAE      : ${LATENT_VAE_PATH}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo ""
echo "  Tasks: RGB→IR, SAR→IR, SAR→RGB (native 1024px)"
echo ""
echo "⚠️  Unified multi-task training loop is not yet implemented."
echo "    Planned invocation (placeholder):"
echo "    python -m examples.ddbm.train_unified \\"
echo "      --num_channels ${NUM_CHANNELS} \\"
echo "      --num_res_blocks ${NUM_RES_BLOCKS} \\"
echo "      --attention_resolutions ${ATTENTION_RESOLUTIONS} \\"
echo "      --channel_mult ${CHANNEL_MULT} \\"
echo "      --resolution ${RESOLUTION} \\"
echo "      --use_latent_target ${USE_LATENT_TARGET} \\"
echo "      --latent_vae_path ${LATENT_VAE_PATH} \\"
echo "      --use_rep_alignment true \\"
echo "      --lambda_rep_alignment ${LAMBDA_REP_ALIGNMENT} \\"
echo "      --resume_from_checkpoint ${RESUME_FROM} \\"
echo "      --output_dir ${OUTPUT_DIR}"
