"""Shared training utilities for MAVIC-T baselines.

Provides:
* :func:`create_optimizer` – build Prodigy or Adam/AdamW from config.
* :func:`save_checkpoint_diffusers` – save model weights in diffusers-style
  directory layout (``unet/``, ``scheduler/``, ``model_index.json``) using
  safetensors.
* :func:`push_checkpoint_to_hub` – upload a diffusers-style checkpoint
  directory to the Hugging Face Hub.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_muon_optimizers():
    """Load Muon optimizer classes, raising ImportError when unavailable."""
    try:
        import muon as muon_module
    except ImportError as exc:
        raise ImportError(
            "Muon optimizer requires the Muon package (https://github.com/KellerJordan/Muon). "
            "Install it with: pip install git+https://github.com/KellerJordan/Muon.git"
        ) from exc
    if not hasattr(muon_module, "Muon") or not hasattr(muon_module, "SingleDeviceMuon"):
        raise ImportError(
            "The installed 'muon' package does not expose Muon optimizers. "
            "Install the optimizer build with: pip install git+https://github.com/KellerJordan/Muon.git"
        )
    return muon_module.Muon, muon_module.SingleDeviceMuon


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def create_optimizer(
    params: Iterable[torch.nn.Parameter],
    optimizer_type: str = "prodigy",
    lr: float = 1.0,
    weight_decay: float = 0.0,
    betas: tuple = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """Create an optimizer from a string identifier.

    Parameters
    ----------
    params : iterable of ``torch.nn.Parameter``
        Model parameters to optimise.
    optimizer_type : str
        ``"prodigy"`` (default) or ``"adamw"`` / ``"adam"`` / ``"muon"``.
    lr : float
        Learning rate.  For Prodigy the recommended value is ``1.0``.
    weight_decay : float
        Weight-decay coefficient.
    betas : tuple
        Beta coefficients for Adam-family optimizers.

    Returns
    -------
    torch.optim.Optimizer
    """
    name = optimizer_type.lower()
    if name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError:
            raise ImportError(
                "prodigyopt is required for the Prodigy optimizer. "
                "Install it with: pip install prodigyopt"
            )
        return Prodigy(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif name == "muon":
        Muon, SingleDeviceMuon = _load_muon_optimizers()
        # Muon constructor asserts params is a list and sorts it by size, so materialize any generator.
        params_list = list(params)
        try:
            use_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except (RuntimeError, AttributeError) as exc:
            logger.debug("Muon optimizer using single-device fallback: %s", exc)
            use_distributed = False
        # Muon paper recommends momentum around 0.95; fall back to that if betas is None.
        momentum = betas[0] if betas else 0.95
        if use_distributed:
            return Muon(params_list, lr=lr, weight_decay=weight_decay, momentum=momentum)
        return SingleDeviceMuon(params_list, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}")


# ---------------------------------------------------------------------------
# Diffusers-style checkpoint saving (safetensors)
# ---------------------------------------------------------------------------

def _save_safetensors(state_dict: Dict[str, torch.Tensor], path: str) -> None:
    """Save a state dict in safetensors format."""
    from safetensors.torch import save_file
    # Ensure all tensors are contiguous and on CPU, filter out non-tensor values
    cpu_sd = {
        k: v.contiguous().cpu() 
        for k, v in state_dict.items() 
        if isinstance(v, torch.Tensor)
    }
    if not cpu_sd:
        logger.warning(f"No tensor values found in state_dict for {path}, skipping save.")
        return
    save_file(cpu_sd, path)


def save_checkpoint_diffusers(
    save_dir: str,
    model: torch.nn.Module,
    scheduler: Optional[Any] = None,
    *,
    model_name: str = "unet",
    extra_state_dicts: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    model_index: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a checkpoint in diffusers-style directory layout.

    Creates the following structure under *save_dir*::

        save_dir/
            model_index.json
            <model_name>/
                config.json
                diffusion_pytorch_model.safetensors
            scheduler/
                scheduler_config.json

    Parameters
    ----------
    save_dir : str
        Root directory for the checkpoint.
    model : torch.nn.Module
        The main model whose ``state_dict()`` is saved.
    scheduler : optional
        A diffusers-compatible scheduler with a ``save_config`` method.
    model_name : str
        Sub-directory name for the main model (default ``"unet"``).
    extra_state_dicts : dict, optional
        Additional ``{folder_name: state_dict}`` to save alongside the main
        model (e.g. ``{"vae": vae.state_dict()}``).
    model_index : dict, optional
        Custom content for ``model_index.json``.  When *None* a minimal
        default is written.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- main model ----
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    _save_safetensors(model.state_dict(), os.path.join(model_dir, "diffusion_pytorch_model.safetensors"))

    # Write a minimal config.json for the model sub-folder
    model_config: Dict[str, Any] = {}
    if hasattr(model, "config") and hasattr(model.config, "to_dict"):
        model_config = model.config.to_dict()
    elif hasattr(model, "config") and isinstance(model.config, dict):
        model_config = model.config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2, default=str)

    # ---- scheduler ----
    if scheduler is not None:
        sched_dir = os.path.join(save_dir, "scheduler")
        os.makedirs(sched_dir, exist_ok=True)
        if hasattr(scheduler, "save_config"):
            scheduler.save_config(sched_dir)
        elif hasattr(scheduler, "state_dict"):
            sched_config: Dict[str, Any] = {}
            if hasattr(scheduler, "config") and hasattr(scheduler.config, "to_dict"):
                sched_config = scheduler.config.to_dict()
            with open(os.path.join(sched_dir, "scheduler_config.json"), "w") as f:
                json.dump(sched_config, f, indent=2, default=str)

    # ---- extra state dicts ----
    if extra_state_dicts:
        for folder, sd in extra_state_dicts.items():
            d = os.path.join(save_dir, folder)
            os.makedirs(d, exist_ok=True)
            _save_safetensors(sd, os.path.join(d, "diffusion_pytorch_model.safetensors"))

    # ---- model_index.json ----
    if model_index is None:
        model_index = {
            "_class_name": type(model).__name__,
            "_diffusers_version": "0.36.0",
            model_name: [type(model).__module__, type(model).__name__],
        }
        if scheduler is not None:
            model_index["scheduler"] = [type(scheduler).__module__, type(scheduler).__name__]
    with open(os.path.join(save_dir, "model_index.json"), "w") as f:
        json.dump(model_index, f, indent=2, default=str)

    logger.info(f"Saved diffusers-style checkpoint to {save_dir}")


# ---------------------------------------------------------------------------
# Hub upload
# ---------------------------------------------------------------------------

def push_checkpoint_to_hub(
    save_dir: str,
    hub_model_id: str,
    commit_message: str = "Update checkpoint",
    token: Optional[str] = None,
    path_in_repo: Optional[str] = None,
) -> None:
    """Upload a diffusers-style checkpoint directory to the Hugging Face Hub.

    Parameters
    ----------
    save_dir : str
        Local directory containing the checkpoint.
    hub_model_id : str
        Repository ID on the Hub (e.g. ``"user/model-name"``).
    commit_message : str
        Git commit message for the upload.
    token : str, optional
        Hugging Face API token.  When *None* the token cached by
        ``huggingface-cli login`` is used.
    path_in_repo : str, optional
        Destination subfolder inside the Hub repository.  Use this to
        organise checkpoints by baseline and task, e.g.
        ``"ddbm/sar2eo/checkpoint-epoch-5"``.  When *None* the files are
        uploaded to the repository root (legacy behaviour).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub is not installed – skipping push to hub.")
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=hub_model_id, exist_ok=True)
    api.upload_folder(
        repo_id=hub_model_id,
        folder_path=save_dir,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    logger.info(f"Pushed checkpoint to hub: {hub_model_id} (path_in_repo={path_in_repo})")
