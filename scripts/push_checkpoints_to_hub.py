#!/usr/bin/env python3
"""Quick script to push checkpoint directories to Hugging Face Hub.

This script pushes accelerator checkpoints (checkpoint-1000) from the four stage1
training directories directly to the Hugging Face Hub.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.training_utils import push_checkpoint_to_hub


def push_checkpoint(
    checkpoint_base_dir: str,
    hub_model_id: str,
    step: int = 1000,
):
    """Push checkpoint directory to hub.
    
    Parameters
    ----------
    checkpoint_base_dir : str
        Base directory containing ddbm/task_name/checkpoint-{step}/
    hub_model_id : str
        Hub repository ID (e.g., "BiliSakura")
    step : int
        Checkpoint step number (default: 1000)
    """
    checkpoint_base_path = Path(checkpoint_base_dir)
    
    # Find the checkpoint directory structure: ddbm/task_name/checkpoint-{step}/
    ddbm_dir = checkpoint_base_path / "ddbm"
    if not ddbm_dir.exists():
        raise FileNotFoundError(f"ddbm directory not found in {checkpoint_base_dir}")
    
    # Find task directory
    task_dirs = list(ddbm_dir.iterdir())
    if not task_dirs:
        raise FileNotFoundError(f"No task directory found in {ddbm_dir}")
    
    task_dir = task_dirs[0]  # Should be only one task directory
    task_name = task_dir.name
    
    checkpoint_dir = task_dir / f"checkpoint-{step}"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    print(f"\n{'='*60}")
    print(f"Processing: {task_name} (step {step})")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"{'='*60}")
    
    print(f"Pushing to hub: {hub_model_id}")
    path_in_repo = f"ddbm/{task_name}/checkpoint-{step}"
    push_checkpoint_to_hub(
        save_dir=str(checkpoint_dir),
        hub_model_id=hub_model_id,
        commit_message=f"ddbm {task_name} checkpoint step {step}",
        path_in_repo=path_in_repo,
    )
    print(f"✓ Successfully pushed {task_name} checkpoint-{step} to hub")


def main():
    """Main entry point."""
    hub_model_id = "BiliSakura/4th-MAVIC-T-ckpt"
    step = 3000
    
    # Checkpoint directories
    checkpoints = [
        "/data/projects/4th-MAVIC-T/ckpt/stage1_sar2rgb",
        "/data/projects/4th-MAVIC-T/ckpt/stage1_sar2ir",
        "/data/projects/4th-MAVIC-T/ckpt/stage1_sar2eo",
        "/data/projects/4th-MAVIC-T/ckpt/stage1_rgb2ir",
    ]
    
    print(f"Hub Model ID: {hub_model_id}")
    print(f"Step: {step}")
    print(f"Checkpoints to process: {len(checkpoints)}")
    
    for checkpoint_dir in checkpoints:
        try:
            push_checkpoint(
                checkpoint_dir,
                hub_model_id=hub_model_id,
                step=step,
            )
        except Exception as e:
            print(f"✗ Error processing {checkpoint_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All checkpoints processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
