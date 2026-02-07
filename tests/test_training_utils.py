import types

import torch

import src.training_utils as training_utils
from src.training_utils import create_optimizer


def test_create_optimizer_muon_single_device():
    model = torch.nn.Linear(2, 2)
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_type="muon",
        lr=0.01,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    assert optimizer.__class__.__name__ == "SingleDeviceMuon"
    assert optimizer.defaults["lr"] == 0.01
    assert optimizer.defaults["weight_decay"] == 0.0
    assert optimizer.defaults["momentum"] == 0.9


def test_create_optimizer_muon_distributed(monkeypatch):
    dummy_dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 8,
    )
    monkeypatch.setattr(training_utils, "dist", dummy_dist, raising=False)

    model = torch.nn.Linear(2, 2)
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_type="muon",
        lr=0.02,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    assert optimizer.__class__.__name__ == "Muon"
    assert optimizer.defaults["lr"] == 0.02
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["momentum"] == 0.9


def test_create_optimizer_muon_world_size_one(monkeypatch):
    dummy_dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 1,
    )
    monkeypatch.setattr(training_utils, "dist", dummy_dist, raising=False)

    model = torch.nn.Linear(1, 1)
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_type="muon",
        lr=0.015,
        weight_decay=0.02,
        betas=(0.95, 0.999),
    )

    assert optimizer.__class__.__name__ == "SingleDeviceMuon"
    assert optimizer.defaults["lr"] == 0.015
    assert optimizer.defaults["weight_decay"] == 0.02
    assert optimizer.defaults["momentum"] == 0.95
