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
# ⚠️ NOTE: in_channels / out_channels of the UNet should be 32 (latent dim)
#   rather than the default pixel channel count. This will be updated later.
#
# Usage:
#   bash scripts/train_stage4d_unified_1024.sh
#   # or multi-GPU:
#   NGPU=4 bash scripts/train_stage4d_unified_1024.sh
#
# TODO: The unified multi-task training loop is not yet implemented.

set -euo pipefail

NGPU="${NGPU:-1}"

# --- Large config from configs/model_scaling_variants.yaml ---
NUM_CHANNELS=256
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,2,4,4"

# --- Latent-space settings (RS-VAE from external repo) ---
USE_LATENT_TARGET=true
LATENT_VAE_PATH="./models/rs_vae"

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
echo "    python -m src.ddbm_baseline.train_unified \\"
echo "      --num_channels ${NUM_CHANNELS} \\"
echo "      --num_res_blocks ${NUM_RES_BLOCKS} \\"
echo "      --attention_resolutions ${ATTENTION_RESOLUTIONS} \\"
echo "      --channel_mult ${CHANNEL_MULT} \\"
echo "      --resolution ${RESOLUTION} \\"
echo "      --use_latent_target ${USE_LATENT_TARGET} \\"
echo "      --latent_vae_path ${LATENT_VAE_PATH} \\"
echo "      --resume_from_checkpoint ${RESUME_FROM} \\"
echo "      --output_dir ${OUTPUT_DIR}"
