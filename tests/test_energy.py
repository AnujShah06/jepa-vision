"""
Tests for src/eval/energy.py.

Uses a tiny VisionJEPA (smoke config: d=64, 2 enc layers) so no real
checkpoint is needed.  Tests verify:
  - image_energy returns finite, non-negative values
  - output shapes are correct (energy [B], patch_energy [B, N])
  - results are deterministic given a fixed seed
  - energy_heatmap returns the expected pixel-space shape
"""

from __future__ import annotations

import torch
import numpy as np
import pytest

from src.eval.energy import energy_heatmap, image_energy
from src.models.jepa import VisionJEPA, VisionJEPAConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model() -> VisionJEPA:
    """Small VisionJEPA that fits in CPU RAM for unit testing."""
    cfg = VisionJEPAConfig(
        img_size=96,
        patch_size=8,      # → 12×12 = 144 patches
        d_model=64,
        enc_layers=2,
        enc_heads=4,
        pred_layers=2,
        pred_width=64,     # == d_model → Identity projections
        pred_heads=4,
        sigreg_weight=1.0,
        sigreg_projections=16,
        ema_decay=0.996,
        use_ema=True,
        dropout=0.0,
    )
    model = VisionJEPA(cfg).eval()
    return model


@pytest.fixture(scope="module")
def dummy_images() -> torch.Tensor:
    """Small batch of random images at production resolution."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 96, 96)   # B=4


# ---------------------------------------------------------------------------
# image_energy tests
# ---------------------------------------------------------------------------

def test_energy_shapes(tiny_model, dummy_images):
    B = dummy_images.shape[0]
    N = (tiny_model.cfg.img_size // tiny_model.cfg.patch_size) ** 2  # 144

    result = image_energy(tiny_model, dummy_images, K=2, seed=7, device="cpu")

    assert "energy" in result
    assert "patch_energy" in result
    assert result["energy"].shape == (B,), \
        f"Expected energy shape ({B},), got {result['energy'].shape}"
    assert result["patch_energy"].shape == (B, N), \
        f"Expected patch_energy shape ({B},{N}), got {result['patch_energy'].shape}"


def test_energy_finite(tiny_model, dummy_images):
    result = image_energy(tiny_model, dummy_images, K=4, seed=0, device="cpu")
    assert torch.isfinite(result["energy"]).all(), "Some energy values are not finite"
    assert torch.isfinite(result["patch_energy"]).all(), \
        "Some patch_energy values are not finite"


def test_energy_non_negative(tiny_model, dummy_images):
    result = image_energy(tiny_model, dummy_images, K=4, seed=0, device="cpu")
    assert (result["energy"] >= 0).all(), "Energy should be non-negative (smooth-L1)"
    assert (result["patch_energy"] >= 0).all(), \
        "patch_energy should be non-negative (smooth-L1)"


def test_energy_deterministic(tiny_model, dummy_images):
    """Same seed → identical results."""
    r1 = image_energy(tiny_model, dummy_images, K=4, seed=99, device="cpu")
    r2 = image_energy(tiny_model, dummy_images, K=4, seed=99, device="cpu")
    assert torch.allclose(r1["energy"], r2["energy"]), \
        "Energy not deterministic with same seed"
    assert torch.allclose(r1["patch_energy"], r2["patch_energy"]), \
        "patch_energy not deterministic with same seed"


def test_energy_seed_changes_result(tiny_model, dummy_images):
    """Different seeds → different mask samples → different energy values."""
    r1 = image_energy(tiny_model, dummy_images, K=4, seed=0, device="cpu")
    r2 = image_energy(tiny_model, dummy_images, K=4, seed=1, device="cpu")
    assert not torch.allclose(r1["energy"], r2["energy"]), \
        "Different seeds produced identical energies (unlikely unless K is tiny)"


def test_energy_k_reduces_variance(tiny_model, dummy_images):
    """
    More mask samples should reduce variance of energy estimates across
    independent random seeds — law of large numbers check.
    """
    n_trials = 10
    energies_k1  = []
    energies_k16 = []
    for s in range(n_trials):
        e1  = image_energy(tiny_model, dummy_images, K=1,  seed=s, device="cpu")
        e16 = image_energy(tiny_model, dummy_images, K=16, seed=s, device="cpu")
        energies_k1.append(e1["energy"])
        energies_k16.append(e16["energy"])

    std_k1  = torch.stack(energies_k1).std(dim=0).mean().item()
    std_k16 = torch.stack(energies_k16).std(dim=0).mean().item()
    assert std_k16 < std_k1, \
        f"K=16 std ({std_k16:.4f}) should be < K=1 std ({std_k1:.4f})"


def test_energy_patch_coverage(tiny_model, dummy_images):
    """
    With K=8 mask samples, most of the 144-patch grid should be covered
    (i.e., patch_energy > 0 at most positions).
    """
    result = image_energy(tiny_model, dummy_images, K=8, seed=0, device="cpu")
    frac_covered = (result["patch_energy"] > 0).float().mean().item()
    assert frac_covered > 0.5, \
        f"Only {frac_covered:.0%} of patches covered with K=8 masks"


# ---------------------------------------------------------------------------
# energy_heatmap tests
# ---------------------------------------------------------------------------

def test_heatmap_shape(tiny_model, dummy_images):
    """Heatmap should be [H, W, 3] uint8 matching the source image size."""
    result = image_energy(tiny_model, dummy_images, K=2, seed=0, device="cpu")
    # Use the first image and its patch energy
    patch_e = result["patch_energy"][0]    # [144]
    src_img = dummy_images[0]              # [3, 96, 96]

    overlay = energy_heatmap(patch_e, src_img, n_h=12, n_w=12)

    assert isinstance(overlay, np.ndarray), "energy_heatmap should return ndarray"
    assert overlay.shape == (96, 96, 3), \
        f"Expected (96, 96, 3), got {overlay.shape}"
    assert overlay.dtype == np.uint8, \
        f"Expected uint8, got {overlay.dtype}"


def test_heatmap_value_range(tiny_model, dummy_images):
    """Pixel values must be in [0, 255]."""
    result = image_energy(tiny_model, dummy_images, K=2, seed=0, device="cpu")
    patch_e = result["patch_energy"][0]
    src_img = dummy_images[0]
    overlay = energy_heatmap(patch_e, src_img, n_h=12, n_w=12)

    assert int(overlay.min()) >= 0
    assert int(overlay.max()) <= 255


def test_heatmap_2d_input(tiny_model, dummy_images):
    """energy_heatmap also accepts a [n_h, n_w] shaped input."""
    result = image_energy(tiny_model, dummy_images, K=2, seed=0, device="cpu")
    patch_e_2d = result["patch_energy"][0].reshape(12, 12)
    src_img = dummy_images[0]
    overlay = energy_heatmap(patch_e_2d, src_img, n_h=12, n_w=12)
    assert overlay.shape == (96, 96, 3)
