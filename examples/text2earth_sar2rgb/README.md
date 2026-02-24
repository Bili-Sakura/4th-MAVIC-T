# Text2Earth SAR2RGB Training Examples

Fine-tune the pre-trained **Text2Earth** text2image diffusion model for SAR-to-RGB translation. Text2Earth is a Stable Diffusion–based model with minor modifications (resolution class embedding). Two conditioning approaches are supported:

1. **ControlNet**: Add a ControlNet branch conditioned on SAR imagery. The base UNet, VAE, and text encoder are frozen; only the ControlNet is trained.
2. **InstructPix2Pix-style**: Direct image conditioning by concatenating the SAR latent with the noisy RGB latent (8-channel UNet input). Only the UNet is fine-tuned.

## Prerequisites

- Text2Earth model at `/data/projects/4th-MAVIC-T/models/lcybuaa/Text2Earth` (or set `--pretrained_model_name_or_path`)
- MAVIC-T dataset at `datasets/BiliSakura/MACIV-T-2025-Structure-Refined`
- Dependencies: `diffusers`, `accelerate`, `transformers`, `torch`, `torchvision`

## ControlNet Approach

```bash
accelerate launch --mixed_precision="bf16" examples/text2earth_sar2rgb/train_sar2rgb_controlnet.py \
  --pretrained_model_name_or_path=/data/projects/4th-MAVIC-T/models/lcybuaa/Text2Earth \
  --output_dir=./ckpt/text2earth_sar2rgb_controlnet \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --learning_rate=5e-6 \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --enable_xformers_memory_efficient_attention
```

## InstructPix2Pix Approach

```bash
accelerate launch --mixed_precision="bf16" examples/text2earth_sar2rgb/train_sar2rgb_instructpix2pix.py \
  --pretrained_model_name_or_path=/data/projects/4th-MAVIC-T/models/lcybuaa/Text2Earth \
  --output_dir=./ckpt/text2earth_sar2rgb_instructpix2pix \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 \
  --conditioning_dropout_prob=0.05 \
  --max_train_steps=10000 \
  --checkpointing_steps=500 \
  --use_ema \
  --enable_xformers_memory_efficient_attention
```

## Dataset

Training uses the MAVIC-T SAR2RGB pairs from:

- `refined_manifest.csv` (train split, task `sar2rgb`)
- `sar2rgb_crop_aug` (optional, enabled by default)
- `paired_sar2rgb_sup.txt` (extra supervised pairs: OpenEarthMap-SAR, SpaceNet6, FUSAR-Map)

Validation pairs from `paired_val_sar2rgb.txt` are excluded from training.

## Inference

### ControlNet

Load the trained ControlNet with Text2Earth and use `StableDiffusionControlNetPipeline`:

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

controlnet = ControlNetModel.from_pretrained("./ckpt/text2earth_sar2rgb_controlnet")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "/data/projects/4th-MAVIC-T/models/lcybuaa/Text2Earth",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

# SAR image as 3ch grayscale conditioning
sar_image = Image.open("sar.png").convert("RGB")  # or replicate 1ch to 3ch
image = pipe(
    "a satellite optical image",
    sar_image,
    num_inference_steps=20,
).images[0]
```

### InstructPix2Pix

Use a custom pipeline that concatenates SAR latent with the initial noise and runs the 8-channel UNet. The trained UNet expects `(noisy_latent, sar_latent)` concatenated along the channel dimension.

## Reference

- ControlNet example: `libs/diffusers/examples/controlnet/train_controlnet.py`
- InstructPix2Pix example: `libs/diffusers/examples/instruct_pix2pix/train_instruct_pix2pix.py`
- Text2Earth model: `/data/projects/4th-MAVIC-T/models/lcybuaa/Text2Earth`
