"""Tests for src.metrics – the MAVIC-T evaluation metric module."""

import math

import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# L1
# ---------------------------------------------------------------------------

class TestComputeL1:
    def test_identical_images(self):
        from src.metrics import compute_l1
        img = torch.rand(4, 3, 64, 64)
        assert compute_l1(img, img).item() == pytest.approx(0.0, abs=1e-6)

    def test_known_value(self):
        from src.metrics import compute_l1
        a = torch.zeros(1, 1, 2, 2)
        b = torch.ones(1, 1, 2, 2)
        assert compute_l1(a, b).item() == pytest.approx(1.0, abs=1e-6)

    def test_output_is_scalar(self):
        from src.metrics import compute_l1
        val = compute_l1(torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8))
        assert val.dim() == 0


# ---------------------------------------------------------------------------
# FIDStatistics
# ---------------------------------------------------------------------------

class TestFIDStatistics:
    def test_identical_distributions(self):
        from src.metrics import FIDStatistics
        rng = np.random.RandomState(0)
        feats = rng.randn(100, 64)
        mu = feats.mean(axis=0)
        sigma = np.cov(feats, rowvar=False)
        s1 = FIDStatistics(mu, sigma)
        s2 = FIDStatistics(mu, sigma)
        assert s1.frechet_distance(s2) == pytest.approx(0.0, abs=1e-4)

    def test_different_distributions(self):
        from src.metrics import FIDStatistics
        rng = np.random.RandomState(42)
        f1 = rng.randn(200, 32)
        f2 = rng.randn(200, 32) + 5  # shifted
        s1 = FIDStatistics(f1.mean(0), np.cov(f1, rowvar=False))
        s2 = FIDStatistics(f2.mean(0), np.cov(f2, rowvar=False))
        assert s1.frechet_distance(s2) > 0

    def test_symmetry(self):
        from src.metrics import FIDStatistics
        rng = np.random.RandomState(1)
        f1 = rng.randn(100, 16)
        f2 = rng.randn(100, 16) + 1
        s1 = FIDStatistics(f1.mean(0), np.cov(f1, rowvar=False))
        s2 = FIDStatistics(f2.mean(0), np.cov(f2, rowvar=False))
        assert s1.frechet_distance(s2) == pytest.approx(
            s2.frechet_distance(s1), abs=1e-4
        )


# ---------------------------------------------------------------------------
# Task score & overall score
# ---------------------------------------------------------------------------

class TestTaskScore:
    def test_perfect_score(self):
        from src.metrics import task_score
        # All zeros → score should be 0
        assert task_score(0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_formula(self):
        from src.metrics import task_score
        fid, lpips, l1 = 10.0, 0.3, 0.15
        expected = ((2.0 / math.pi) * math.atan(fid) + lpips + l1) / 3.0
        assert task_score(fid, lpips, l1) == pytest.approx(expected)

    def test_monotonicity_fid(self):
        from src.metrics import task_score
        # Higher FID → higher (worse) score
        assert task_score(100, 0, 0) > task_score(1, 0, 0)


class TestOverallScore:
    def test_all_tasks_attempted(self):
        from src.metrics import overall_score
        scores = {"sar2eo": 0.1, "sar2rgb": 0.2, "sar2ir": 0.3, "rgb2ir": 0.4}
        assert overall_score(scores) == pytest.approx(0.25)

    def test_missing_task_penalty(self):
        from src.metrics import overall_score
        # 3 tasks attempted, 1 missing → penalty of 1
        scores = {"sar2eo": 0.0, "sar2rgb": 0.0, "sar2ir": 0.0}
        assert overall_score(scores) == pytest.approx(1.0)

    def test_no_tasks_attempted(self):
        from src.metrics import overall_score
        assert overall_score({}) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# LPIPS wrapper
# ---------------------------------------------------------------------------

class TestLPIPS:
    def test_identical_images(self):
        from src.metrics import LPIPS
        lpips = LPIPS(net_type="vgg")
        img = torch.rand(2, 3, 64, 64)
        val = lpips(img, img)
        assert val.item() == pytest.approx(0.0, abs=0.05)

    def test_grayscale_support(self):
        from src.metrics import LPIPS
        lpips = LPIPS(net_type="vgg")
        img = torch.rand(2, 1, 64, 64)
        val = lpips(img, img)
        assert val.dim() == 0  # returns scalar


# ---------------------------------------------------------------------------
# MavicCriterion
# ---------------------------------------------------------------------------

class TestMavicCriterion:
    def test_identical_images_low_loss(self):
        from src.metrics import MavicCriterion
        criterion = MavicCriterion(lpips_weight=1.0, l1_weight=1.0)
        img = torch.rand(2, 3, 64, 64)
        loss = criterion(img, img)
        assert loss.item() < 0.1

    def test_different_images_higher_loss(self):
        from src.metrics import MavicCriterion
        criterion = MavicCriterion(lpips_weight=1.0, l1_weight=1.0)
        a = torch.zeros(2, 3, 64, 64)
        b = torch.ones(2, 3, 64, 64)
        loss = criterion(a, b)
        assert loss.item() > 0.5
