#!/usr/bin/env python3
"""Script to document model configurations from all baselines.

This script extracts configuration details by directly parsing config files
to avoid import dependencies.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Parsed model configuration."""
    task_name: str = ""
    resolution: int = 256
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 1
    num_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: str = "32,16,8"
    dropout: float = 0.0
    condition_mode: str = "concat"
    channel_mult: str = ""
    
    # Baseline-specific fields
    unet_type: str = ""
    pred_mode: str = ""
    sigma_max: float = 0.0
    sigma_min: float = 0.0
    objective: str = ""
    num_timesteps: int = 0
    mt_type: str = ""
    num_steps: int = 0
    ode_sampler: str = ""
    num_train_timesteps: int = 0
    ddim_num_steps: int = 0
    ddim_eta: float = 0.0
    ngf: int = 64
    ndf: int = 64
    n_downsampling: int = 2
    n_blocks: int = 9
    n_layers_D: int = 3
    nce_layers: str = ""
    nce_idt: bool = False
    lambda_GAN: float = 0.0
    lambda_NCE: float = 0.0
    pretrained_model_name_or_path: str = ""
    lora_rank_unet: int = 0
    lora_rank_vae: int = 0
    prompt: str = ""


def parse_config_function(config_file: Path, task: str) -> Optional[ModelConfig]:
    """Parse a config function from a Python file."""
    content = config_file.read_text()
    
    # Find the task config function
    pattern = rf'def {task}_config\(.*?\).*?:.*?cfg = TaskConfig\((.*?)\)\s+for k, v in overrides'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        return None
    
    config_str = match.group(1)
    config = ModelConfig()
    config.task_name = task
    
    # Parse key-value pairs
    for line in config_str.split('\n'):
        line = line.strip()
        # Remove inline comments
        if '#' in line:
            line = line.split('#')[0].strip()
        if '=' in line:
            try:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().rstrip(',')
                
                # Remove quotes from strings
                if value.startswith('"') or value.startswith("'"):
                    value = value[1:-1]
                
                # Parse booleans
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                # Parse numbers
                elif value.replace('.', '', 1).replace('-', '', 1).replace('_', '').isdigit():
                    if '.' in value:
                        value = float(value.replace('_', ''))
                    else:
                        value = int(value.replace('_', ''))
                
                if hasattr(config, key):
                    setattr(config, key, value)
            except Exception as e:
                pass
    
    return config


def estimate_unet_parameters(config, is_conditional=True):
    """Estimate UNet parameters based on configuration."""
    # Parse channel_mult
    if config.channel_mult:
        channel_mult = tuple(int(c) for c in config.channel_mult.split(","))
    else:
        # Default channel_mult based on resolution
        resolution_defaults = {
            512: (1, 1, 2, 2, 4, 4),
            256: (1, 1, 2, 2, 4, 4),
            1024: (1, 1, 2, 2, 4, 4),
            128: (1, 1, 2, 3, 4),
            64: (1, 2, 3, 4),
            32: (1, 2, 3, 4),
        }
        channel_mult = resolution_defaults.get(int(config.resolution), (1, 2, 3, 4))
    
    # Calculate base parameters - ensure all are ints
    num_channels = int(config.num_channels)
    model_channels = int(config.model_channels)
    num_res_blocks = int(config.num_res_blocks)
    
    # Conditional input doubles channels
    if is_conditional:
        in_channels = model_channels * 2
    else:
        in_channels = model_channels
    
    # Rough estimation
    num_levels = len(channel_mult)
    
    # Initial convolution
    params = in_channels * num_channels * 9
    
    # Encoder blocks
    for i, mult in enumerate(channel_mult):
        ch = num_channels * mult
        params += num_res_blocks * (ch * ch * 18)
        if i < len(channel_mult) - 1:
            params += ch * ch * 4
    
    # Middle block
    mid_ch = num_channels * channel_mult[-1]
    params += mid_ch * mid_ch * 18
    
    # Decoder blocks
    for i, mult in reversed(list(enumerate(channel_mult))):
        ch = num_channels * mult
        params += num_res_blocks * (ch * ch * 18)
        if i > 0:
            params += ch * ch * 4
    
    # Output convolution
    params += num_channels * model_channels * 9
    
    # Attention adds parameters
    if config.attention_resolutions:
        attn_res_list = config.attention_resolutions.split(',')
        num_attn_layers = len([r for r in attn_res_list if r.strip()])
        params += num_attn_layers * mid_ch * mid_ch * 4
    
    return params


def estimate_cut_parameters(config):
    """Estimate CUT generator and discriminator parameters."""
    ngf = int(config.ngf)
    ndf = int(config.ndf)
    n_blocks = int(config.n_blocks)
    n_downsampling = int(config.n_downsampling)
    n_layers_D = int(config.n_layers_D)
    source_channels = int(config.source_channels)
    target_channels = int(config.target_channels)
    
    # Generator (ResNet-based)
    gen_params = 0
    gen_params += source_channels * ngf * 49
    gen_params += ngf * (ngf * 2) * 9 * n_downsampling
    max_ch = ngf * (2 ** n_downsampling)
    gen_params += n_blocks * max_ch * max_ch * 18
    gen_params += (ngf * 2) * ngf * 9 * n_downsampling
    gen_params += ngf * target_channels * 49
    
    # Discriminator (PatchGAN)
    disc_params = 0
    disc_params += target_channels * ndf * 16
    for i in range(n_layers_D):
        mult = min(2 ** i, 8)
        disc_params += ndf * mult * ndf * mult * 2 * 16
    disc_params += ndf * 8 * 16
    
    return gen_params, disc_params


def format_number(num):
    """Format large numbers with commas."""
    return f"{num:,}"


def main():
    """Main function to document all model configurations."""
    
    project_root = Path.cwd()
    src_dir = project_root / 'src'
    
    results = {}
    
    baselines = {
        'DDBM': 'ddbm_baseline',
        'BiBBDM': 'bibbdm_baseline',
        'I2SB': 'i2sb_baseline',
        'DDIB': 'ddib_baseline',
        'CUT': 'cut_baseline',
        'Img2Img-Turbo': 'img2img_turbo',
    }
    
    tasks = ['sar2eo', 'rgb2ir', 'sar2ir', 'sar2rgb']
    
    # Process each baseline
    for baseline_name, baseline_dir in baselines.items():
        print(f"\n{'='*60}")
        print(f"Processing {baseline_name} baseline...")
        print(f"{'='*60}")
        
        results[baseline_name] = {}
        
        config_file = src_dir / baseline_dir / 'config.py'
        if not config_file.exists():
            print(f"Config file not found: {config_file}")
            continue
        
        # Process each task
        for task in tasks:
            print(f"\n{task}...")
            
            try:
                config = parse_config_function(config_file, task)
                if config is None:
                    print(f"  Could not parse {task} config")
                    continue
                
                # Estimate parameters based on baseline type
                if baseline_name in ['DDBM', 'BiBBDM', 'I2SB']:
                    params = estimate_unet_parameters(config, is_conditional=True)
                    results[baseline_name][task] = {
                        'config': config,
                        'estimated_params': params,
                    }
                    print(f"  Estimated: ~{format_number(params)} parameters")
                    
                elif baseline_name == 'DDIB':
                    # Two separate models
                    source_params = estimate_unet_parameters(config, is_conditional=False)
                    target_params = estimate_unet_parameters(config, is_conditional=False)
                    
                    results[baseline_name][task] = {
                        'config': config,
                        'source_estimated_params': source_params,
                        'target_estimated_params': target_params,
                        'total_estimated_params': source_params + target_params,
                    }
                    print(f"  Source: ~{format_number(source_params)} parameters")
                    print(f"  Target: ~{format_number(target_params)} parameters")
                    print(f"  Combined: ~{format_number(source_params + target_params)} parameters")
                    
                elif baseline_name == 'CUT':
                    gen_params, disc_params = estimate_cut_parameters(config)
                    results[baseline_name][task] = {
                        'config': config,
                        'generator_estimated_params': gen_params,
                        'discriminator_estimated_params': disc_params,
                        'total_estimated_params': gen_params + disc_params,
                    }
                    print(f"  Generator: ~{format_number(gen_params)} parameters")
                    print(f"  Discriminator: ~{format_number(disc_params)} parameters")
                    print(f"  Combined: ~{format_number(gen_params + disc_params)} parameters")
                    
                elif baseline_name == 'Img2Img-Turbo':
                    # SD-Turbo base model has ~865M parameters
                    sd_turbo_params = 865_000_000
                    lora_unet_rank = int(config.lora_rank_unet) if config.lora_rank_unet else 8
                    lora_vae_rank = int(config.lora_rank_vae) if config.lora_rank_vae else 4
                    lora_unet_params = lora_unet_rank * 320 * 50
                    lora_vae_params = lora_vae_rank * 128 * 20
                    
                    results[baseline_name][task] = {
                        'config': config,
                        'base_model_params': sd_turbo_params,
                        'lora_params': lora_unet_params + lora_vae_params,
                        'note': 'Uses pretrained SD-Turbo with LoRA fine-tuning',
                    }
                    print(f"  Base model (SD-Turbo): ~{format_number(sd_turbo_params)} parameters")
                    print(f"  LoRA trainable params: ~{format_number(lora_unet_params + lora_vae_params)} parameters")
                    
            except Exception as e:
                print(f"  Error: {str(e)}")
                import traceback
                traceback.print_exc()
                results[baseline_name][task] = {
                    'error': str(e)
                }
    
    # Generate markdown documentation
    print(f"\n{'='*60}")
    print("Generating markdown documentation...")
    print(f"{'='*60}")
    
    generate_markdown(results, project_root)
    
    print("\nDone! Documentation saved to ./docs/model_parameters.md")


def generate_markdown(results, project_root):
    """Generate markdown documentation from results."""
    
    md_lines = [
        "# Model Parameters Documentation",
        "",
        "This document provides detailed information about model architectures and parameter counts",
        "for all baselines in the 4th-MAVIC-T project. All models are initialized from their default",
        "configurations for each task.",
        "",
        "**Note:** Parameter counts are estimates based on model configurations. Actual counts may vary",
        "slightly depending on implementation details.",
        "",
        "**Tasks:**",
        "- `sar2eo`: SAR to EO (optical) translation",
        "- `rgb2ir`: RGB to Infrared translation",
        "- `sar2ir`: SAR to Infrared translation",
        "- `sar2rgb`: SAR to RGB translation",
        "",
    ]
    
    # Process each baseline
    for baseline_name, baseline_data in results.items():
        md_lines.append(f"## {baseline_name} Baseline")
        md_lines.append("")
        
        # Add baseline-specific description
        if baseline_name == 'DDBM':
            md_lines.append("**Description:** Denoising Diffusion Bridge Models for image-to-image translation.")
            md_lines.append("Uses a UNet architecture with conditioning via concatenation.")
        elif baseline_name == 'BiBBDM':
            md_lines.append("**Description:** Bidirectional Brownian Bridge Diffusion Models with reversible translation.")
            md_lines.append("Supports bidirectional sampling (source→target and target→source).")
        elif baseline_name == 'I2SB':
            md_lines.append("**Description:** Image-to-Image Schrödinger Bridge for paired image translation.")
            md_lines.append("Uses Schrödinger Bridge formulation with ODE/SDE samplers.")
        elif baseline_name == 'DDIB':
            md_lines.append("**Description:** Dual Diffusion Implicit Bridges - trains two independent unconditional diffusion models.")
            md_lines.append("Translation works via DDIM inversion and forward sampling through shared latent space.")
        elif baseline_name == 'CUT':
            md_lines.append("**Description:** Contrastive Unpaired Translation using contrastive learning.")
            md_lines.append("Uses a ResNet-based generator and PatchGAN discriminator with contrastive loss.")
        elif baseline_name == 'Img2Img-Turbo':
            md_lines.append("**Description:** Pix2Pix-Turbo based on Stable Diffusion Turbo with LoRA adapters.")
            md_lines.append("Fine-tunes pretrained SD-Turbo model using LoRA for efficient adaptation.")
        md_lines.append("")
        
        # Create table for this baseline
        if baseline_name == 'DDIB':
            # DDIB has two models
            md_lines.append("| Task | Source Model | Target Model | Total Parameters | Resolution | Channels |")
            md_lines.append("|------|--------------|--------------|------------------|------------|----------|")
            
            for task, task_data in baseline_data.items():
                if 'error' in task_data:
                    md_lines.append(f"| {task} | Error | Error | Error | - | - |")
                else:
                    config = task_data['config']
                    source_params = format_number(task_data['source_estimated_params'])
                    target_params = format_number(task_data['target_estimated_params'])
                    total_params = format_number(task_data['total_estimated_params'])
                    resolution = f"{config.resolution}×{config.resolution}"
                    channels = f"{config.source_channels}→{config.target_channels}"
                    md_lines.append(f"| {task} | ~{source_params} | ~{target_params} | ~{total_params} | {resolution} | {channels} |")
                    
        elif baseline_name == 'CUT':
            # CUT has generator and discriminator
            md_lines.append("| Task | Generator | Discriminator | Total Parameters | Resolution | Channels |")
            md_lines.append("|------|-----------|---------------|------------------|------------|----------|")
            
            for task, task_data in baseline_data.items():
                if 'error' in task_data:
                    md_lines.append(f"| {task} | Error | Error | Error | - | - |")
                else:
                    config = task_data['config']
                    gen_params = format_number(task_data['generator_estimated_params'])
                    disc_params = format_number(task_data['discriminator_estimated_params'])
                    total_params = format_number(task_data['total_estimated_params'])
                    resolution = f"{config.resolution}×{config.resolution}"
                    channels = f"{config.source_channels}→{config.target_channels}"
                    md_lines.append(f"| {task} | ~{gen_params} | ~{disc_params} | ~{total_params} | {resolution} | {channels} |")
                    
        elif baseline_name == 'Img2Img-Turbo':
            # Turbo has base model + LoRA
            md_lines.append("| Task | Base Model (SD-Turbo) | LoRA Parameters | Resolution | Channels |")
            md_lines.append("|------|-----------------------|-----------------|------------|----------|")
            
            for task, task_data in baseline_data.items():
                if 'error' in task_data:
                    md_lines.append(f"| {task} | Error | Error | - | - |")
                elif 'note' in task_data:
                    config = task_data['config']
                    base_params = format_number(task_data['base_model_params'])
                    lora_params = format_number(task_data['lora_params'])
                    resolution = f"{config.resolution}×{config.resolution}"
                    channels = f"{config.source_channels}→{config.target_channels}"
                    md_lines.append(f"| {task} | ~{base_params} | ~{lora_params} | {resolution} | {channels} |")
        else:
            # Standard single model
            md_lines.append("| Task | Estimated Parameters | Resolution | Channels | Model Channels |")
            md_lines.append("|------|---------------------|------------|----------|----------------|")
            
            for task, task_data in baseline_data.items():
                if 'error' in task_data:
                    md_lines.append(f"| {task} | Error | - | - | - |")
                else:
                    config = task_data['config']
                    params = format_number(task_data['estimated_params'])
                    resolution = f"{config.resolution}×{config.resolution}"
                    channels = f"{config.source_channels}→{config.target_channels}"
                    model_ch = config.model_channels
                    md_lines.append(f"| {task} | ~{params} | {resolution} | {channels} | {model_ch} |")
        
        md_lines.append("")
        
        # Add detailed configuration for each task
        md_lines.append(f"### {baseline_name} - Detailed Configuration")
        md_lines.append("")
        
        for task, task_data in baseline_data.items():
            if 'error' in task_data:
                continue
                
            config = task_data['config']
            md_lines.append(f"#### {task}")
            md_lines.append("")
            md_lines.append("```yaml")
            
            # Print key config values
            if baseline_name in ['DDBM', 'BiBBDM', 'I2SB']:
                md_lines.append(f"task_name: {config.task_name}")
                md_lines.append(f"resolution: {config.resolution}x{config.resolution}")
                md_lines.append(f"channels: {config.source_channels} → {config.target_channels}")
                md_lines.append(f"model_channels: {config.model_channels}")
                md_lines.append(f"unet_base_channels: {config.num_channels}")
                md_lines.append(f"num_res_blocks: {config.num_res_blocks}")
                md_lines.append(f"attention_resolutions: {config.attention_resolutions}")
                md_lines.append(f"dropout: {config.dropout}")
                md_lines.append(f"condition_mode: {config.condition_mode}")
                
                if baseline_name == 'DDBM':
                    if config.unet_type:
                        md_lines.append(f"unet_type: {config.unet_type}")
                    if config.pred_mode:
                        md_lines.append(f"pred_mode: {config.pred_mode}")
                    if config.sigma_max:
                        md_lines.append(f"sigma_max: {config.sigma_max}")
                        md_lines.append(f"sigma_min: {config.sigma_min}")
                elif baseline_name == 'BiBBDM':
                    if config.objective:
                        md_lines.append(f"objective: {config.objective}")
                    if config.num_timesteps:
                        md_lines.append(f"num_timesteps: {config.num_timesteps}")
                    if config.mt_type:
                        md_lines.append(f"mt_type: {config.mt_type}")
                elif baseline_name == 'I2SB':
                    if config.num_steps:
                        md_lines.append(f"num_steps: {config.num_steps}")
                    if config.ode_sampler:
                        md_lines.append(f"ode_sampler: {config.ode_sampler}")
                    
            elif baseline_name == 'DDIB':
                md_lines.append(f"task_name: {config.task_name}")
                md_lines.append(f"resolution: {config.resolution}x{config.resolution}")
                md_lines.append(f"source_channels: {config.source_channels}")
                md_lines.append(f"target_channels: {config.target_channels}")
                md_lines.append(f"unet_base_channels: {config.num_channels}")
                md_lines.append(f"num_res_blocks: {config.num_res_blocks}")
                md_lines.append(f"attention_resolutions: {config.attention_resolutions}")
                if config.num_train_timesteps:
                    md_lines.append(f"num_train_timesteps: {config.num_train_timesteps}")
                if config.ddim_num_steps:
                    md_lines.append(f"ddim_num_steps: {config.ddim_num_steps}")
                if config.ddim_eta:
                    md_lines.append(f"ddim_eta: {config.ddim_eta}")
                
            elif baseline_name == 'CUT':
                md_lines.append(f"task_name: {config.task_name}")
                md_lines.append(f"resolution: {config.resolution}x{config.resolution}")
                md_lines.append(f"channels: {config.source_channels} → {config.target_channels}")
                md_lines.append(f"generator_base_filters_ngf: {config.ngf}")
                md_lines.append(f"discriminator_base_filters_ndf: {config.ndf}")
                md_lines.append(f"num_downsampling_layers: {config.n_downsampling}")
                md_lines.append(f"num_residual_blocks: {config.n_blocks}")
                md_lines.append(f"discriminator_layers: {config.n_layers_D}")
                if config.nce_layers:
                    md_lines.append(f"nce_layers: {config.nce_layers}")
                if config.lambda_GAN:
                    md_lines.append(f"lambda_GAN: {config.lambda_GAN}")
                if config.lambda_NCE:
                    md_lines.append(f"lambda_NCE: {config.lambda_NCE}")
                
            elif baseline_name == 'Img2Img-Turbo':
                md_lines.append(f"task_name: {config.task_name}")
                md_lines.append(f"resolution: {config.resolution}x{config.resolution}")
                if config.pretrained_model_name_or_path:
                    md_lines.append(f"pretrained_model: {config.pretrained_model_name_or_path}")
                if config.lora_rank_unet:
                    md_lines.append(f"lora_rank_unet: {config.lora_rank_unet}")
                if config.lora_rank_vae:
                    md_lines.append(f"lora_rank_vae: {config.lora_rank_vae}")
                if config.prompt:
                    md_lines.append(f"prompt: \"{config.prompt}\"")
                    
            md_lines.append("```")
            md_lines.append("")
    
    # Add summary section
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append("### Parameter Count Comparison (sar2eo task)")
    md_lines.append("")
    md_lines.append("| Baseline | Estimated Parameters | Architecture Type | Notes |")
    md_lines.append("|----------|---------------------|-------------------|-------|")
    
    for baseline_name, baseline_data in results.items():
        if 'sar2eo' in baseline_data:
            task_data = baseline_data['sar2eo']
            if 'error' in task_data:
                md_lines.append(f"| {baseline_name} | Error | - | - |")
            elif 'note' in task_data:
                base = format_number(task_data['base_model_params'])
                lora = format_number(task_data['lora_params'])
                md_lines.append(f"| {baseline_name} | Base: ~{base}, LoRA: ~{lora} | SD-Turbo + LoRA | Pretrained foundation model |")
            elif 'estimated_params' in task_data:
                params = format_number(task_data['estimated_params'])
                arch_type = "Conditional UNet"
                md_lines.append(f"| {baseline_name} | ~{params} | {arch_type} | Diffusion-based |")
            elif 'total_estimated_params' in task_data:
                total = format_number(task_data['total_estimated_params'])
                if baseline_name == 'DDIB':
                    md_lines.append(f"| {baseline_name} | ~{total} | Dual Unconditional UNets | Two independent models |")
                elif baseline_name == 'CUT':
                    md_lines.append(f"| {baseline_name} | ~{total} | ResNet + PatchGAN | Generator + Discriminator |")
    
    md_lines.append("")
    md_lines.append("### Key Architecture Differences")
    md_lines.append("")
    md_lines.append("1. **Diffusion Models (DDBM, BiBBDM, I2SB, DDIB):** Use iterative denoising process")
    md_lines.append("   - DDBM: Bridge diffusion with VP/VE noise schedules")
    md_lines.append("   - BiBBDM: Brownian Bridge with bidirectional translation")
    md_lines.append("   - I2SB: Schrödinger Bridge formulation")
    md_lines.append("   - DDIB: Two separate unconditional models with DDIM bridge")
    md_lines.append("")
    md_lines.append("2. **GAN-based (CUT):** Direct translation with adversarial + contrastive loss")
    md_lines.append("   - Faster inference (single forward pass)")
    md_lines.append("   - ResNet generator + PatchGAN discriminator")
    md_lines.append("")
    md_lines.append("3. **Foundation Model (Img2Img-Turbo):** Leverages pretrained SD-Turbo")
    md_lines.append("   - Large pretrained base (~865M parameters)")
    md_lines.append("   - Efficient fine-tuning via LoRA (~few K trainable parameters)")
    md_lines.append("   - Single-step inference capability")
    md_lines.append("")
    md_lines.append("### Resolution and Channel Support")
    md_lines.append("")
    md_lines.append("- **1024×1024:** sar2ir, sar2rgb (DDBM, BiBBDM, I2SB, DDIB, CUT)")
    md_lines.append("- **512×512:** Img2Img-Turbo (all tasks)")
    md_lines.append("- **256×256:** sar2eo, rgb2ir (DDBM, BiBBDM, I2SB, DDIB, CUT)")
    md_lines.append("")
    md_lines.append("Most models support flexible channel configurations (1-ch, 3-ch) through their architecture.")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("*Generated automatically by document_model_configs.py*")
    md_lines.append(f"*Parameter counts are estimates based on model architecture configurations*")
    
    # Write to file
    output_path = project_root / "docs" / "model_parameters.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md_lines))
    print(f"Wrote documentation to: {output_path}")


if __name__ == "__main__":
    main()
