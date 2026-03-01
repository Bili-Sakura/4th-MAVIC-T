#!/usr/bin/env python
# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Quick test of Text2Earth inference pipeline (base model, no SAR2RGB)."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, EulerDiscreteScheduler

MODEL_PATH = _PROJECT_ROOT / "models/lcybuaa/Text2Earth"
OUTPUT_PATH = _PROJECT_ROOT / "examples/text2earth_sar2rgb/test_output.png"
VAE_MODEL = _PROJECT_ROOT / "models/lcybuaa/Text2Earth/vae"  # sdvae default for text2earth

def main():
    print("Loading Text2Earth pipeline...")
    vae = AutoencoderKL.from_pretrained(str(VAE_MODEL))
    scheduler = EulerDiscreteScheduler.from_pretrained(str(MODEL_PATH), subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        scheduler=scheduler,
        vae=vae,
        custom_pipeline="pipeline_text2earth_diffusion",
        safety_checker=None,
    )
    pipe = pipe.to("cuda")

    # Use resolution prompt: 17_GOOGLE_LEVEL_ = 1m, 18_GOOGLE_LEVEL_ = 0.5m
    prompt = "18_GOOGLE_LEVEL_ Seven green circular farmlands are neatly arranged on the ground"
    print(f"Generating with prompt: {prompt}")

    image = pipe(
        prompt,
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=3.5,
    ).images[0]

    image.save(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
