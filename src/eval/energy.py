# TODO Step 1.4 — multi-mask rewrite.
# This file is a port of the cocktail-JEPA recipe_energy / energy_over_loader
# harness.  The slot-iteration logic (mask each slot in turn) maps to
# patch-block sampling in the vision version.  Step 1.4 replaces the
# deterministic per-slot loop with multi-mask averaging (K=8 independent
# block samples) and adds per-patch energy maps for spatial heatmaps.
# Until then, this file is kept as an import stub so tests/test_imports.py
# can verify the package structure.

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn


@torch.no_grad()
def recipe_energy(
    model: nn.Module,
    batch: dict,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Energy for each recipe in a batch: mean latent prediction error over
    all real ingredient slots.

    TODO Step 1.4: replace with vision patch-block energy (multi-mask).
    """
    raise NotImplementedError("Step 1.4: rewrite for vision patch-block energy")


@torch.no_grad()
def energy_over_loader(
    model: nn.Module,
    loader,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[str]]:
    """
    Compute energy for every sample delivered by a DataLoader.

    TODO Step 1.4: rewrite for vision (multi-mask averaging, heatmap output).
    """
    raise NotImplementedError("Step 1.4: rewrite for vision patch-block energy")
