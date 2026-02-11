#!/usr/bin/env python3
"""Quick script to push TensorBoard logs to Hugging Face Hub.

This script pushes TensorBoard log directories from stage1 training runs
to the Hugging Face Hub, alongside checkpoints.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.training_utils import push_checkpoint_to_hub
from src.utils.paths import path_from_root


def push_logs(
    logs_base_dir: str,
    hub_model_id: str,
):
    """Push TensorBoard logs directory to hub.

    Parameters
    ----------
    logs_base_dir : str
        Path to the logs directory, e.g.
        ckpt/exp3/stage1_sar2eo/ddbm/sar2eo/logs
    hub_model_id : str
        Hub repository ID (e.g., "BiliSakura/4th-MAVIC-T-ckpt")
    """
    logs_path = Path(logs_base_dir)
    if not logs_path.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_base_dir}")

    # Infer task name from path: .../ddbm/<task_name>/logs
    # e.g. ckpt/exp3/stage1_sar2eo/ddbm/sar2eo/logs -> sar2eo
    parts = logs_path.parts
    try:
        ddbm_idx = parts.index("ddbm")
        task_name = parts[ddbm_idx + 1]
    except (ValueError, IndexError):
        task_name = Path(logs_base_dir).parent.name

    print(f"\n{'='*60}")
    print(f"Processing: {task_name}")
    print(f"Logs: {logs_path}")
    print(f"{'='*60}")

    print(f"Pushing to hub: {hub_model_id}")
    path_in_repo = f"ddbm/{task_name}/logs"
    push_checkpoint_to_hub(
        save_dir=str(logs_path),
        hub_model_id=hub_model_id,
        commit_message=f"ddbm {task_name} TensorBoard logs",
        path_in_repo=path_in_repo,
    )
    print(f"✓ Successfully pushed {task_name} logs to hub")


def main():
    """Main entry point."""
    hub_model_id = "BiliSakura/4th-MAVIC-T-ckpt"

    # TensorBoard log directories (relative to PROJECT_ROOT)
    logs_dirs = [
        str(path_from_root("ckpt/exp3/stage1_sar2eo/ddbm/sar2eo/logs")),
        str(path_from_root("ckpt/exp3/stage1_sar2ir/ddbm/sar2ir/logs")),
    ]

    print(f"Hub Model ID: {hub_model_id}")
    print(f"Logs to process: {len(logs_dirs)}")

    for logs_dir in logs_dirs:
        try:
            push_logs(
                logs_dir,
                hub_model_id=hub_model_id,
            )
        except Exception as e:
            print(f"✗ Error processing {logs_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("All logs processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
