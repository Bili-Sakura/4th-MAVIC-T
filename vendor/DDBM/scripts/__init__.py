# Copyright 2024 The DDBM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
"""
DDBM Training and Sampling Scripts.

This module provides command-line scripts for training and sampling with DDBM
using the Hugging Face diffusers-compatible API.

Available Scripts:
    - `train_ddbm_diffusers.py`: Training script using accelerate for distributed training
    - `sample_ddbm_diffusers.py`: Sampling script using the DDBMPipeline

Usage:
    # Training (single GPU)
    python scripts/train_ddbm_diffusers.py --data_dir /path/to/data

    # Training (multi-GPU with accelerate)
    accelerate launch scripts/train_ddbm_diffusers.py --data_dir /path/to/data

    # Sampling
    python scripts/sample_ddbm_diffusers.py \\
        --model_path ./outputs/model.pt \\
        --data_dir /path/to/data \\
        --output_dir ./samples
"""

