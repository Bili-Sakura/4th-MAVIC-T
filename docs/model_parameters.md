# Model Parameters Documentation

This document provides detailed information about model architectures and parameter counts
for all baselines in the 4th-MAVIC-T project. All models are initialized from their default
configurations for each task.

**Note:** Parameter counts are estimates based on model configurations. Actual counts may vary
slightly depending on implementation details.

**Tasks:**
- `sar2eo`: SAR to EO (optical) translation
- `rgb2ir`: RGB to Infrared translation
- `sar2ir`: SAR to Infrared translation
- `sar2rgb`: SAR to RGB translation

## DDBM Baseline

**Description:** Denoising Diffusion Bridge Models for image-to-image translation.
Uses a UNet architecture with conditioning via concatenation.

| Task | Estimated Parameters | Resolution | Channels | Model Channels |
|------|---------------------|------------|----------|----------------|
| sar2eo | ~61,803,904 | 256×256 | 1→1 | 1 |
| rgb2ir | ~61,810,816 | 1024×1024 | 3→1 | 3 |
| sar2ir | ~61,803,904 | 1024×1024 | 1→1 | 1 |
| sar2rgb | ~61,810,816 | 1024×1024 | 1→3 | 3 |

### DDBM - Detailed Configuration

#### sar2eo

```yaml
task_name: sar2eo
resolution: 256x256
channels: 1 → 1
model_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### rgb2ir

```yaml
task_name: rgb2ir
resolution: 1024x1024
channels: 3 → 1
model_channels: 3
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### sar2ir

```yaml
task_name: sar2ir
resolution: 1024x1024
channels: 1 → 1
model_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### sar2rgb

```yaml
task_name: sar2rgb
resolution: 1024x1024
channels: 1 → 3
model_channels: 3
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

## BiBBDM Baseline

**Description:** Bidirectional Brownian Bridge Diffusion Models with reversible translation.
Supports bidirectional sampling (source→target and target→source).

| Task | Estimated Parameters | Resolution | Channels | Model Channels |
|------|---------------------|------------|----------|----------------|
| sar2eo | ~61,803,904 | 256×256 | 1→1 | 1 |
| rgb2ir | ~61,810,816 | 1024×1024 | 3→1 | 3 |
| sar2ir | ~61,803,904 | 1024×1024 | 1→1 | 1 |
| sar2rgb | ~61,810,816 | 1024×1024 | 1→3 | 3 |

### BiBBDM - Detailed Configuration

#### sar2eo

```yaml
task_name: sar2eo
resolution: 256x256
channels: 1 → 1
model_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### rgb2ir

```yaml
task_name: rgb2ir
resolution: 1024x1024
channels: 3 → 1
model_channels: 3
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### sar2ir

```yaml
task_name: sar2ir
resolution: 1024x1024
channels: 1 → 1
model_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### sar2rgb

```yaml
task_name: sar2rgb
resolution: 1024x1024
channels: 1 → 3
model_channels: 3
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

## I2SB Baseline

**Description:** Image-to-Image Schrödinger Bridge for paired image translation.
Uses Schrödinger Bridge formulation with ODE/SDE samplers.

| Task | Estimated Parameters | Resolution | Channels | Model Channels |
|------|---------------------|------------|----------|----------------|
| sar2eo | ~61,803,904 | 256×256 | 1→1 | 1 |
| rgb2ir | ~61,810,816 | 1024×1024 | 3→1 | 3 |
| sar2ir | ~61,803,904 | 1024×1024 | 1→1 | 1 |
| sar2rgb | ~61,810,816 | 1024×1024 | 1→3 | 3 |

### I2SB - Detailed Configuration

#### sar2eo

```yaml
task_name: sar2eo
resolution: 256x256
channels: 1 → 1
model_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### rgb2ir

```yaml
task_name: rgb2ir
resolution: 1024x1024
channels: 3 → 1
model_channels: 3
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### sar2ir

```yaml
task_name: sar2ir
resolution: 1024x1024
channels: 1 → 1
model_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### sar2rgb

```yaml
task_name: sar2rgb
resolution: 1024x1024
channels: 1 → 3
model_channels: 3
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

## DDIB Baseline

**Description:** Dual Diffusion Implicit Bridges - trains two independent unconditional diffusion models.
Translation works via DDIM inversion and forward sampling through shared latent space.

| Task | Source Model | Target Model | Total Parameters | Resolution | Channels |
|------|--------------|--------------|------------------|------------|----------|
| sar2eo | ~61,802,752 | ~61,802,752 | ~123,605,504 | 256×256 | 1→1 |
| rgb2ir | ~61,807,360 | ~61,807,360 | ~123,614,720 | 1024×1024 | 3→1 |
| sar2ir | ~61,802,752 | ~61,802,752 | ~123,605,504 | 1024×1024 | 1→1 |
| sar2rgb | ~61,807,360 | ~61,807,360 | ~123,614,720 | 1024×1024 | 1→3 |

### DDIB - Detailed Configuration

#### sar2eo

```yaml
task_name: sar2eo
resolution: 256x256
source_channels: 1
target_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
```

#### rgb2ir

```yaml
task_name: rgb2ir
resolution: 1024x1024
source_channels: 3
target_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
```

#### sar2ir

```yaml
task_name: sar2ir
resolution: 1024x1024
source_channels: 1
target_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
```

#### sar2rgb

```yaml
task_name: sar2rgb
resolution: 1024x1024
source_channels: 1
target_channels: 3
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
```

## CUT Baseline

**Description:** Contrastive Unpaired Translation using contrastive learning.
Uses a ResNet-based generator and PatchGAN discriminator with contrastive loss.

| Task | Generator | Discriminator | Total Parameters | Resolution | Channels |
|------|-----------|---------------|------------------|------------|----------|
| sar2eo | ~10,918,016 | ~2,761,728 | ~13,679,744 | 256×256 | 1→1 |
| rgb2ir | ~10,924,288 | ~2,761,728 | ~13,686,016 | 1024×1024 | 3→1 |
| sar2ir | ~10,918,016 | ~2,761,728 | ~13,679,744 | 1024×1024 | 1→1 |
| sar2rgb | ~10,924,288 | ~2,763,776 | ~13,688,064 | 1024×1024 | 1→3 |

### CUT - Detailed Configuration

#### sar2eo

```yaml
task_name: sar2eo
resolution: 256x256
channels: 1 → 1
generator_base_filters_ngf: 64
discriminator_base_filters_ndf: 64
num_downsampling_layers: 2
num_residual_blocks: 9
discriminator_layers: 3
```

#### rgb2ir

```yaml
task_name: rgb2ir
resolution: 1024x1024
channels: 3 → 1
generator_base_filters_ngf: 64
discriminator_base_filters_ndf: 64
num_downsampling_layers: 2
num_residual_blocks: 9
discriminator_layers: 3
```

#### sar2ir

```yaml
task_name: sar2ir
resolution: 1024x1024
channels: 1 → 1
generator_base_filters_ngf: 64
discriminator_base_filters_ndf: 64
num_downsampling_layers: 2
num_residual_blocks: 9
discriminator_layers: 3
```

#### sar2rgb

```yaml
task_name: sar2rgb
resolution: 1024x1024
channels: 1 → 3
generator_base_filters_ngf: 64
discriminator_base_filters_ndf: 64
num_downsampling_layers: 2
num_residual_blocks: 9
discriminator_layers: 3
```

## Img2Img-Turbo Baseline

**Description:** Pix2Pix-Turbo based on Stable Diffusion Turbo with LoRA adapters.
Fine-tunes pretrained SD-Turbo model using LoRA for efficient adaptation.

| Task | Base Model (SD-Turbo) | LoRA Parameters | Resolution | Channels |
|------|-----------------------|-----------------|------------|----------|
| sar2eo | ~865,000,000 | ~138,240 | 512×512 | 1→1 |
| rgb2ir | ~865,000,000 | ~138,240 | 1024×1024 | 3→1 |
| sar2ir | ~865,000,000 | ~138,240 | 1024×1024 | 1→1 |
| sar2rgb | ~865,000,000 | ~138,240 | 1024×1024 | 1→3 |

### Img2Img-Turbo - Detailed Configuration

#### sar2eo

```yaml
task_name: sar2eo
resolution: 512x512
prompt: "convert SAR image to electro-optical image"
```

#### rgb2ir

```yaml
task_name: rgb2ir
resolution: 1024x1024
prompt: "convert RGB image to infrared image"
```

#### sar2ir

```yaml
task_name: sar2ir
resolution: 1024x1024
prompt: "convert SAR image to infrared image"
```

#### sar2rgb

```yaml
task_name: sar2rgb
resolution: 1024x1024
prompt: "convert SAR image to RGB image"
```

## Summary

### Parameter Count Comparison (sar2eo task)

| Baseline | Estimated Parameters | Architecture Type | Notes |
|----------|---------------------|-------------------|-------|
| DDBM | ~61,803,904 | Conditional UNet | Diffusion-based |
| BiBBDM | ~61,803,904 | Conditional UNet | Diffusion-based |
| I2SB | ~61,803,904 | Conditional UNet | Diffusion-based |
| DDIB | ~123,605,504 | Dual Unconditional UNets | Two independent models |
| CUT | ~13,679,744 | ResNet + PatchGAN | Generator + Discriminator |
| Img2Img-Turbo | Base: ~865,000,000, LoRA: ~138,240 | SD-Turbo + LoRA | Pretrained foundation model |

### Key Architecture Differences

1. **Diffusion Models (DDBM, BiBBDM, I2SB, DDIB):** Use iterative denoising process
   - DDBM: Bridge diffusion with VP/VE noise schedules
   - BiBBDM: Brownian Bridge with bidirectional translation
   - I2SB: Schrödinger Bridge formulation
   - DDIB: Two separate unconditional models with DDIM bridge

2. **GAN-based (CUT):** Direct translation with adversarial + contrastive loss
   - Faster inference (single forward pass)
   - ResNet generator + PatchGAN discriminator

3. **Foundation Model (Img2Img-Turbo):** Leverages pretrained SD-Turbo
   - Large pretrained base (~865M parameters)
   - Efficient fine-tuning via LoRA (~few K trainable parameters)
   - Single-step inference capability

### Resolution and Channel Support

- **1024×1024:** sar2ir, sar2rgb (DDBM, BiBBDM, I2SB, DDIB, CUT)
- **512×512:** Img2Img-Turbo (all tasks)
- **256×256:** sar2eo, rgb2ir (DDBM, BiBBDM, I2SB, DDIB, CUT)

Most models support flexible channel configurations (1-ch, 3-ch) through their architecture.

---
*Generated automatically by document_model_configs.py*
*Parameter counts are estimates based on model architecture configurations*