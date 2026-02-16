#!/usr/bin/env bash
# Stage 4a — RS-VAE (external)
#
# The RS-VAE is trained and borrowed from an external repository.
# It is initialised from SD21-VAE and fine-tuned on combined remote-sensing
# data (SAR, EO, RGB, IR).
#
# The RS-VAE accepts 3-channel input. For single-channel modalities (SAR, EO, IR):
#   Encode: repeat 1-ch pixel → 3-ch input
#   Decode: average 3-ch output → 1-ch result
#
# This script is a reference placeholder only — no local training is needed.

set -euo pipefail

export HF_TOKEN="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"
export HF_ENDPOINT="https://hf-mirror.com"

echo "=== Stage 4a: RS-VAE ==="
echo ""
echo "  The RS-VAE is trained and borrowed from an external repository."
echo "  Place the RS-VAE checkpoint at: ./models/rs_vae"
echo "  (or update LATENT_VAE_PATH in stage4b/4c/4d scripts)"
