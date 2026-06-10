"""Tests for the I-JEPA block masking module."""

import random

import pytest
import torch

from src.data.masking import IJEPAMaskCollator, MaskResult, sample_block_mask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_H, N_W = 12, 12
N = N_H * N_W  # 144


def _make_mask(seed: int | None = None, **kwargs) -> MaskResult:
    rng = random.Random(seed)
    return sample_block_mask(N_H, N_W, rng=rng, **kwargs)


def _many_masks(n: int = 200) -> list[MaskResult]:
    rng = random.Random(42)
    return [sample_block_mask(N_H, N_W, rng=rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

def test_no_context_target_overlap():
    """Context and target patches must be disjoint for every sample."""
    for m in _many_masks():
        ctx = set(m.context_patches)
        tgt = set(m.target_patches)
        assert ctx.isdisjoint(tgt), (
            f"Overlap found: {ctx & tgt}"
        )


def test_context_non_empty():
    """Context must always contain at least one patch."""
    for m in _many_masks():
        assert len(m.context_patches) > 0, "Empty context detected"


def test_target_non_empty():
    """Target union must always contain at least one patch."""
    for m in _many_masks():
        assert len(m.target_patches) > 0, "Empty target union detected"


def test_all_indices_valid():
    """Every returned patch index must be in [0, N)."""
    for m in _many_masks():
        for idx in m.target_patches + m.context_patches:
            assert 0 <= idx < N, f"Invalid patch index {idx}"
        for blk in m.target_blocks:
            for idx in blk:
                assert 0 <= idx < N, f"Invalid index {idx} in target block"


def test_target_patches_is_union_of_blocks():
    """target_patches must equal the union of the individual target blocks."""
    for m in _many_masks(50):
        union = set()
        for blk in m.target_blocks:
            union.update(blk)
        assert set(m.target_patches) == union


def test_context_patches_subset_of_context_block():
    """context_patches must be a subset of the sampled context block."""
    rng = random.Random(0)
    for _ in range(200):
        m = sample_block_mask(N_H, N_W, rng=rng)
        # fallback path may widen context beyond context_block, so only check
        # when the context block was large enough not to trigger the fallback
        if set(m.context_block) - set(m.target_patches):
            # fallback not triggered; context must be a strict subset
            assert set(m.context_patches).issubset(set(m.context_block)), (
                "context_patches outside context_block"
            )


def test_correct_number_of_target_blocks():
    """Must always produce exactly n_targets target blocks."""
    for n_targets in (1, 2, 4):
        for m in [sample_block_mask(N_H, N_W, n_targets=n_targets,
                                    rng=random.Random(i))
                  for i in range(50)]:
            assert len(m.target_blocks) == n_targets


# ---------------------------------------------------------------------------
# Coverage / area bounds
# ---------------------------------------------------------------------------

def test_each_target_block_area_in_bounds():
    """
    Each individual target block should cover approximately 15-20% of patches.
    We allow ±30% relative slack to absorb integer rounding of h and w.
    """
    lo = 0.15 * N * 0.70   # ~15 patches
    hi = 0.20 * N * 1.30   # ~37 patches
    rng = random.Random(7)
    for _ in range(300):
        m = sample_block_mask(N_H, N_W, rng=rng)
        for blk in m.target_blocks:
            area = len(blk)
            assert lo <= area <= hi, (
                f"Target block area {area} outside [{lo:.1f}, {hi:.1f}]"
            )


def test_context_block_area_in_bounds():
    """
    Context block (before target removal) should cover most patches.

    Nominal target is 85-100% = 122-144 patches.  However, clamping block
    dimensions to the grid boundary (e.g. h clamped to n_h=12 while w is
    computed from aspect=1.5) can reduce the actual area to ~75% (108/144).
    The lower bound here captures the worst-case of this integer-rounding +
    grid-clamping effect.  The important property is that context is always
    large -- a context block of ~75% still gives the predictor plenty to
    work with.
    """
    lo = int(0.70 * N)   # 101 patches -- conservative floor after clamping
    hi = N
    rng = random.Random(99)
    for _ in range(300):
        m = sample_block_mask(N_H, N_W, rng=rng)
        area = len(m.context_block)
        assert lo <= area <= hi, (
            f"Context block area {area} outside [{lo}, {hi}]"
        )


def test_target_union_below_ceiling():
    """Union of 4 target blocks should never exceed ~80% of patches."""
    ceiling = 0.85 * N   # conservative: 4 × 20% with no overlap = 80%
    rng = random.Random(5)
    for _ in range(300):
        m = sample_block_mask(N_H, N_W, rng=rng)
        assert len(m.target_patches) <= ceiling, (
            f"Target union {len(m.target_patches)} exceeds ceiling {ceiling}"
        )


# ---------------------------------------------------------------------------
# Sorted outputs
# ---------------------------------------------------------------------------

def test_outputs_are_sorted():
    """target_patches, context_patches, and context_block must be sorted."""
    for m in _many_masks(50):
        assert m.target_patches == sorted(m.target_patches)
        assert m.context_patches == sorted(m.context_patches)
        assert m.context_block == sorted(m.context_block)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def test_collator_batch_format():
    """IJEPAMaskCollator produces correct keys and tensor types."""
    collator = IJEPAMaskCollator(n_h=N_H, n_w=N_W, seed=0)
    fake_img = torch.zeros(3, 96, 96)
    batch = collator([(fake_img, -1)] * 8)

    assert set(batch.keys()) == {"imgs", "target_idx", "context_idx"}
    assert batch["imgs"].shape == (8, 3, 96, 96)
    assert batch["target_idx"].dtype == torch.long
    assert batch["context_idx"].dtype == torch.long


def test_collator_no_overlap():
    """Collator output target/context must be disjoint."""
    collator = IJEPAMaskCollator(n_h=N_H, n_w=N_W, seed=42)
    fake_img = torch.zeros(3, 96, 96)
    for _ in range(50):
        batch = collator([(fake_img, -1)] * 4)
        tgt = set(batch["target_idx"].tolist())
        ctx = set(batch["context_idx"].tolist())
        assert tgt.isdisjoint(ctx), "Collator produced overlapping target/context"


def test_collator_different_masks_across_calls():
    """Successive calls should (usually) give different masks."""
    collator = IJEPAMaskCollator(n_h=N_H, n_w=N_W, seed=0)
    fake_img = torch.zeros(3, 96, 96)
    masks = [
        collator([(fake_img, -1)] * 2)["target_idx"].tolist()
        for _ in range(20)
    ]
    unique = {tuple(m) for m in masks}
    assert len(unique) > 1, "Collator returned identical masks on every call"
