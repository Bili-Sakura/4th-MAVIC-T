#!/usr/bin/env python
"""
Example: Using DDBM with Hugging Face diffusers API

This example demonstrates how to use the DDBM pipeline with the diffusers-compatible
API for image-to-image translation tasks.

Examples:
    1. Basic inference with mock model
    2. Loading a pretrained model
    3. Using PIL images as input
    4. Standalone scheduler usage
    5. Custom sampling loops
"""

import torch
from PIL import Image
import numpy as np

# Import DDBM diffusers components
from ddbm import DDBMScheduler, DDBMPipeline, DDBMPipelineOutput


class MockDDBMUNet(torch.nn.Module):
    """
    Mock UNet for demonstration purposes.

    In practice, you would use either:
    - ddbm.UNetModel (ADM-style UNet)
    - ddbm.SongUNet (EDM-style UNet)

    Or load a pretrained model from a checkpoint.
    """

    def __init__(self, channels=3):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # Dummy param

    def forward(self, x, timesteps, xT=None):
        # Mock denoising: gradually blend towards target
        # In a real model, this would be a full UNet forward pass
        alpha = 0.1
        if xT is not None:
            return x * (1 - alpha) + xT * alpha
        return x * (1 - alpha)


def example_basic_inference():
    """
    Basic inference example using the DDBM pipeline.

    This demonstrates the simplest way to use DDBM for image-to-image translation.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Inference")
    print("=" * 60)

    # 1. Create the scheduler with VP (variance-preserving) mode
    scheduler = DDBMScheduler(
        sigma_max=1.0,          # Maximum sigma (use 1.0 for VP mode)
        sigma_min=0.0001,       # Minimum sigma
        pred_mode='vp',         # Variance-preserving schedule
        beta_d=2.0,             # Beta_d parameter
        beta_min=0.1,           # Beta_min parameter
        num_train_timesteps=40, # Default training steps
    )

    # 2. Create the model (use your trained DDBM model here)
    model = MockDDBMUNet()

    # 3. Create the pipeline
    pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

    # 4. Prepare input image (random tensor for demo)
    # In practice, this would be your source image
    source_image = torch.randn(1, 3, 64, 64)

    # 5. Run inference
    print("Running DDBM inference...")
    result = pipeline(
        source_image=source_image,
        num_inference_steps=40,    # Number of diffusion steps
        guidance=1.0,              # Guidance weight
        churn_step_ratio=0.33,     # Stochastic sampling ratio
        output_type='pil',         # Output as PIL images
    )

    # 6. Access results
    # The result is a DDBMPipelineOutput with 'images' and 'nfe' attributes
    images = result.images  # List of PIL images
    nfe = result.nfe        # Number of function evaluations

    print(f"Generated {len(images)} images")
    print(f"Number of function evaluations: {nfe}")
    print(f"Output image size: {images[0].size}")

    return images


def example_load_pretrained():
    """
    Example loading a pretrained DDBM model.

    This shows how to load model weights from a checkpoint file.
    """
    print("\n" + "=" * 60)
    print("Example 2: Loading Pretrained Model")
    print("=" * 60)

    # 1. Create scheduler
    scheduler = DDBMScheduler(pred_mode='vp', sigma_max=1.0)

    # 2. Create model architecture
    # For a real model, you would use:
    #
    # from ddbm import create_model, model_and_diffusion_defaults
    # defaults = model_and_diffusion_defaults()
    # model = create_model(
    #     image_size=64,
    #     in_channels=3,
    #     num_channels=128,
    #     num_res_blocks=2,
    #     unet_type='adm',  # or 'edm'
    #     attention_resolutions="32,16,8",
    #     condition_mode='concat',
    # )

    # 3. Load checkpoint
    # model.load_state_dict(torch.load("path/to/checkpoint.pt"))
    # model.eval()

    # For this demo, we use a mock model
    model = MockDDBMUNet()

    # 4. Create pipeline
    pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

    print("Pipeline created successfully!")
    print(f"Scheduler pred_mode: {pipeline.pred_mode}")
    print(f"Scheduler sigma_max: {pipeline.sigma_max}")

    return pipeline


def example_pil_input():
    """
    Example using PIL images as input.

    This demonstrates how to use real images with the pipeline.
    """
    print("\n" + "=" * 60)
    print("Example 3: PIL Image Input")
    print("=" * 60)

    scheduler = DDBMScheduler(pred_mode='vp', sigma_max=1.0)
    model = MockDDBMUNet()
    pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

    # Create a random PIL image (replace with Image.open("your_image.png"))
    source_img = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )

    # Run inference with PIL input
    result = pipeline(
        source_image=source_img,
        num_inference_steps=20,
        output_type='pil',
    )

    print(f"Input image size: {source_img.size}")
    print(f"Output image size: {result.images[0].size}")
    print(f"NFE: {result.nfe}")


def example_numpy_output():
    """
    Example with numpy array output.

    Useful for further processing or integration with other libraries.
    """
    print("\n" + "=" * 60)
    print("Example 4: NumPy Output")
    print("=" * 60)

    scheduler = DDBMScheduler(pred_mode='ve', sigma_max=80.0)
    model = MockDDBMUNet()
    pipeline = DDBMPipeline(unet=model, scheduler=scheduler)

    # Batch of 2 images
    source_image = torch.randn(2, 3, 32, 32)

    result = pipeline(
        source_image=source_image,
        num_inference_steps=10,
        output_type='np',  # NumPy output
    )

    print(f"Output type: {type(result.images)}")
    print(f"Output shape: {result.images.shape}")
    print(f"Output range: [{result.images.min():.3f}, {result.images.max():.3f}]")


def example_scheduler_standalone():
    """
    Example using scheduler standalone for custom sampling loops.

    This is useful when you want more control over the sampling process.
    """
    print("\n" + "=" * 60)
    print("Example 5: Standalone Scheduler")
    print("=" * 60)

    scheduler = DDBMScheduler(
        sigma_max=1.0,
        sigma_min=0.0001,
        pred_mode='vp',
    )

    # Set timesteps
    scheduler.set_timesteps(40)

    print(f"Number of sigmas: {len(scheduler.sigmas)}")
    print(f"First sigma: {scheduler.sigmas[0]:.6f}")
    print(f"Last non-zero sigma: {scheduler.sigmas[-2]:.6f}")
    print(f"Sigma range: [{scheduler.sigmas.min():.6f}, {scheduler.sigmas.max():.6f}]")

    # Example: Add noise to samples using bridge process
    batch_size = 2
    original_samples = torch.randn(batch_size, 3, 64, 64)
    noise = torch.randn_like(original_samples)
    x_T = torch.randn_like(original_samples)
    timesteps = torch.tensor([0.5, 0.8])  # Sigma values

    noisy = scheduler.add_noise(original_samples, noise, timesteps, x_T)
    print(f"Noisy samples shape: {noisy.shape}")


def example_custom_sampling_loop():
    """
    Example implementing a custom sampling loop.

    This demonstrates low-level control over the diffusion process.
    """
    print("\n" + "=" * 60)
    print("Example 6: Custom Sampling Loop")
    print("=" * 60)

    scheduler = DDBMScheduler(pred_mode='vp', sigma_max=1.0)
    model = MockDDBMUNet()

    # Set timesteps
    num_steps = 20
    scheduler.set_timesteps(num_steps)
    sigmas = scheduler.sigmas

    # Source image (x_T in bridge terminology)
    x_T = torch.randn(1, 3, 64, 64)
    x = x_T.clone()

    print(f"Starting custom sampling with {num_steps} steps...")

    # Custom sampling loop
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Get model prediction (mock)
        with torch.no_grad():
            model_output = model(x, sigma.unsqueeze(0), xT=x_T)

        # Use scheduler step (simplified)
        result = scheduler.step(
            model_output=model_output,
            timestep=i,
            sample=x,
            x_T=x_T,
            guidance=1.0,
        )
        x = result.prev_sample

        if i % 5 == 0:
            print(f"  Step {i}: sigma={sigma:.4f} -> {sigma_next:.4f}")

    print(f"Final sample shape: {x.shape}")
    print(f"Final sample range: [{x.min():.3f}, {x.max():.3f}]")


def example_config_save_load():
    """
    Example saving and loading scheduler configuration.

    This demonstrates diffusers-style configuration management.
    """
    print("\n" + "=" * 60)
    print("Example 7: Config Save/Load")
    print("=" * 60)

    import tempfile
    import os

    # Create scheduler with custom config
    scheduler = DDBMScheduler(
        sigma_max=1.0,
        sigma_min=0.001,
        pred_mode='vp',
        beta_d=2.5,
        beta_min=0.15,
    )

    # Save config to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        scheduler.save_config(tmpdir)
        print(f"Saved config to {tmpdir}")

        # List saved files
        files = os.listdir(tmpdir)
        print(f"Saved files: {files}")

        # Load config and create new scheduler
        loaded_scheduler = DDBMScheduler.from_config(tmpdir)
        print(f"Loaded scheduler pred_mode: {loaded_scheduler.pred_mode}")
        print(f"Loaded scheduler sigma_max: {loaded_scheduler.sigma_max}")
        print(f"Config matches: {loaded_scheduler.config == scheduler.config}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("DDBM Diffusers Integration Examples")
    print("=" * 60)

    example_basic_inference()
    example_load_pretrained()
    example_pil_input()
    example_numpy_output()
    example_scheduler_standalone()
    example_custom_sampling_loop()
    example_config_save_load()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
