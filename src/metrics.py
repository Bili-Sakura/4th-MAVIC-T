"""MAVIC-T official evaluation metrics.

This module implements the three evaluation metrics used by the 4th MAVIC-T
challenge (https://www.codabench.org/competitions/12566/) together with the
composite task-score and overall-score formulas.

Metrics
-------
- **LPIPS** – Learned Perceptual Image Patch Similarity (VGG-16).
- **FID**  – Fréchet Inception Distance (InceptionV3 features).
- **L1**   – Mean pixel-wise absolute difference.

Scoring
-------
``task_score = (2/π · arctan(FID)  +  LPIPS  +  L1) / 3``

``overall_score = mean(task_scores)`` with a penalty of **1** added for each
unattempted domain (out of the four: sar2eo, sar2rgb, sar2ir, rgb2ir).

Training loss
-------------
:class:`MavicCriterion` provides a differentiable combination of LPIPS and L1
so that the model can be directly optimised toward the evaluation metric.
FID is distribution-level and cannot be used as a per-sample loss, so it is
excluded from the training criterion but included in evaluation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg


# ---------------------------------------------------------------------------
# L1 metric
# ---------------------------------------------------------------------------

def compute_l1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute pixel-wise error.

    Parameters
    ----------
    predictions, targets : torch.Tensor
        Image tensors in ``[0, 1]`` range with shape ``(N, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Scalar L1 value.
    """
    return F.l1_loss(predictions, targets)


# ---------------------------------------------------------------------------
# LPIPS metric (wraps torchmetrics)
# ---------------------------------------------------------------------------

class LPIPS(nn.Module):
    """VGG-16 based Learned Perceptual Image Patch Similarity.

    Thin wrapper around ``torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity``
    that accepts images in ``[0, 1]`` and re-scales them to ``[-1, 1]`` as
    expected by the underlying network.

    Parameters
    ----------
    net_type : str
        Backbone network (``"vgg"`` for the official MAVIC-T evaluation).
    """

    def __init__(self, net_type: str = "vgg") -> None:
        super().__init__()
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self._lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LPIPS.

        Parameters
        ----------
        predictions, targets : torch.Tensor
            Image tensors in ``[0, 1]`` range with shape ``(N, C, H, W)``.
            For grayscale (C=1), channels are replicated to 3.

        Returns
        -------
        torch.Tensor
            Scalar LPIPS value (lower is better).
        """
        # LPIPS expects [-1, 1] and 3-channel images
        preds = predictions * 2 - 1
        tgts = targets * 2 - 1
        if preds.shape[1] == 1:
            preds = preds.expand(-1, 3, -1, -1)
            tgts = tgts.expand(-1, 3, -1, -1)
        return self._lpips(preds, tgts)


# ---------------------------------------------------------------------------
# FID metric (pure-PyTorch, using torchvision InceptionV3)
# ---------------------------------------------------------------------------

class FIDStatistics:
    """Container for the mean and covariance of InceptionV3 activations."""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        self.mu = np.atleast_1d(mu)
        self.sigma = np.atleast_2d(sigma)

    def frechet_distance(self, other: "FIDStatistics", eps: float = 1e-6) -> float:
        """Compute the Fréchet distance to *other* statistics."""
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        assert mu1.shape == mu2.shape, (
            f"Mean vectors have different lengths: {mu1.shape} vs {mu2.shape}"
        )
        assert sigma1.shape == sigma2.shape, (
            f"Covariance matrices differ: {sigma1.shape} vs {sigma2.shape}"
        )

        diff = mu1 - mu2

        covmean_result = linalg.sqrtm(sigma1.dot(sigma2))
        if isinstance(covmean_result, tuple):
            covmean = covmean_result[0]
        else:
            covmean = covmean_result
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        return float(
            diff.dot(diff)
            + np.trace(sigma1)
            + np.trace(sigma2)
            - 2 * np.trace(covmean)
        )


@torch.no_grad()
def compute_inception_features(
    images: torch.Tensor,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract pool-3 features from InceptionV3 for FID computation.

    Parameters
    ----------
    images : torch.Tensor
        ``(N, C, H, W)`` float tensor in ``[0, 1]``.
    batch_size : int
        Inference batch size.
    device : torch.device, optional
        Defaults to CUDA if available.

    Returns
    -------
    np.ndarray
        ``(N, 2048)`` feature matrix.
    """
    from torchvision.models import inception_v3, Inception_V3_Weights

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = nn.Identity()  # pool-3 features before classification head
    model = model.to(device).eval()

    features_list: list[np.ndarray] = []
    n = images.shape[0]
    for i in range(0, n, batch_size):
        batch = images[i : i + batch_size].to(device)
        # InceptionV3 expects 3-channel, 299×299
        if batch.shape[1] == 1:
            batch = batch.expand(-1, 3, -1, -1)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        feats = model(batch)
        features_list.append(feats.cpu().numpy())

    return np.concatenate(features_list, axis=0)


def compute_fid(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> float:
    """Compute FID between two sets of images.

    Parameters
    ----------
    predictions, targets : torch.Tensor
        ``(N, C, H, W)`` float tensors in ``[0, 1]``.

    Returns
    -------
    float
        FID value (lower is better).
    """
    feats_pred = compute_inception_features(predictions, batch_size, device)
    feats_tgt = compute_inception_features(targets, batch_size, device)

    stats_pred = FIDStatistics(
        mu=np.mean(feats_pred, axis=0),
        sigma=np.cov(feats_pred, rowvar=False),
    )
    stats_tgt = FIDStatistics(
        mu=np.mean(feats_tgt, axis=0),
        sigma=np.cov(feats_tgt, rowvar=False),
    )
    return stats_pred.frechet_distance(stats_tgt)


# ---------------------------------------------------------------------------
# Task score & overall score
# ---------------------------------------------------------------------------

def task_score(fid: float, lpips: float, l1: float) -> float:
    """Compute the MAVIC-T task score.

    ``score = (2/π · arctan(FID) + LPIPS + L1) / 3``

    Lower is better (all three components are distances).
    """
    normalised_fid = (2.0 / math.pi) * math.atan(fid)
    return (normalised_fid + lpips + l1) / 3.0


def overall_score(
    task_scores: Dict[str, float],
    all_tasks: Sequence[str] = ("sar2eo", "sar2rgb", "sar2ir", "rgb2ir"),
) -> float:
    """Compute the MAVIC-T overall score.

    ``overall = mean(task_scores) + penalty``

    A penalty of **1** is added for each unattempted task.
    Lower is better.
    """
    attempted = [task_scores[t] for t in all_tasks if t in task_scores]
    unattempted = sum(1 for t in all_tasks if t not in task_scores)

    if not attempted:
        return float(len(all_tasks))

    return sum(attempted) / len(all_tasks) + unattempted


# ---------------------------------------------------------------------------
# Differentiable training criterion (LPIPS + L1)
# ---------------------------------------------------------------------------

class MavicCriterion(nn.Module):
    """Differentiable training loss matching the MAVIC-T evaluation metric.

    The official evaluation combines LPIPS, FID, and L1.  FID is a
    distribution-level metric and cannot be used per-sample, so this criterion
    uses only the two sample-level components:

    ``loss = lpips_weight · LPIPS(pred, target) + l1_weight · L1(pred, target)``

    Parameters
    ----------
    lpips_weight : float
        Weight for the LPIPS term (default ``1.0``).
    l1_weight : float
        Weight for the L1 term (default ``1.0``).
    net_type : str
        LPIPS backbone (default ``"vgg"``).
    """

    def __init__(
        self,
        lpips_weight: float = 1.0,
        l1_weight: float = 1.0,
        net_type: str = "vgg",
    ) -> None:
        super().__init__()
        self.lpips_weight = lpips_weight
        self.l1_weight = l1_weight
        self._lpips = LPIPS(net_type=net_type)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined loss.

        Parameters
        ----------
        predictions, targets : torch.Tensor
            ``(N, C, H, W)`` tensors in ``[0, 1]`` range.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        l1 = compute_l1(predictions, targets)
        lpips_val = self._lpips(predictions, targets)
        return self.lpips_weight * lpips_val + self.l1_weight * l1
