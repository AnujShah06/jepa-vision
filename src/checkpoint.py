"""
checkpoint.py -- save and load JEPA training state.

A checkpoint captures everything needed to (a) resume training or
(b) hand a trained model to evaluation. That means model weights, optimizer
state, scheduler state, the step counter, and a config dict that describes
the model (so it can be reconstructed without guessing hyperparameters).

Checkpoints go under runs/ -- git-ignored.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    epoch: int = 0,
    step: int = 0,
    extra: dict | None = None,
) -> Path:
    """Write a checkpoint. `extra` may hold metrics, e.g. best val loss."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "extra": extra or {},
    }
    torch.save(blob, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    map_location: str = "cpu",
) -> dict:
    """
    Load a checkpoint.

    If `model` is provided, loads the saved state dict into it (strict=False
    so checkpoints saved before a parameter was added still load).
    Returns {"model", "step", "optimizer_state", "scheduler_state", "extra"}.
    """
    blob = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        missing, unexpected = model.load_state_dict(
            blob["model_state"], strict=False
        )
        if missing:
            print(f"[checkpoint] using defaults for absent keys: {list(missing)}")
        if unexpected:
            print(f"[checkpoint] ignoring unknown keys: {list(unexpected)}")

    return {
        "model": model,
        "epoch": blob.get("epoch", 0),
        "step": blob.get("step", 0),
        "optimizer_state": blob.get("optimizer_state"),
        "scheduler_state": blob.get("scheduler_state"),
        "extra": blob.get("extra", {}),
    }
