"""Tests for the I-JEPA block masking module."""

import random

import pytest
import torch

from src.data.masking import IJEPAMaskCollator, MaskResult, sample_block_mask

# ---------------------------------------------------------------------------
# Configs under test
# ---------------------------------------------------------------------------

N_H, N_W = 12, 12
N = N_H * N_W  # 144

# Each config matches one of the YAML configs in configs/ so that tests fail
# fast if a config is added that violates a structural invariant.
CONFIGS: dict[str, dict] = {
    "ref": dict(
        n_targets=4,
        target_scale=(0.15, 0.20),
        context_scale=(0.85, 1.00),
    ),
    "hardmask": dict(
        n_targets=4,
        target_scale=(0.20, 0.25),
        context_scale=(0.75, 0.90),
    ),
}
CFG_IDS = list(CONFIGS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _many_masks(cfg: dict, n: int = 200, seed: int = 42) -> list[MaskResult]:
    rng = random.Random(seed)
    return [sample_block_mask(N_H, N_W, rng=rng, **cfg) for _ in range(n)]


# ---------------------------------------------------------------------------
# Structural invariants — apply to every config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_no_context_target_overlap(cfg):
    """Context and target patches must be disjoint for every sample."""
    for m in _many_masks(cfg):
        ctx = set(m.context_patches)
        tgt = set(m.target_patches)
        assert ctx.isdisjoint(tgt), f"Overlap found: {ctx & tgt}"


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_context_non_empty(cfg):
    """Context must always contain at least one patch."""
    for m in _many_masks(cfg):
        assert len(m.context_patches) > 0, "Empty context detected"


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_target_non_empty(cfg):
    """Target union must always contain at least one patch."""
    for m in _many_masks(cfg):
        assert len(m.target_patches) > 0, "Empty target union detected"


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_all_indices_valid(cfg):
    """Every returned patch index must be in [0, N)."""
    for m in _many_masks(cfg):
        for idx in m.target_patches + m.context_patches:
            assert 0 <= idx < N, f"Invalid patch index {idx}"
        for blk in m.target_blocks:
            for idx in blk:
                assert 0 <= idx < N, f"Invalid index {idx} in target block"


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_target_patches_is_union_of_blocks(cfg):
    """target_patches must equal the union of the individual target blocks."""
    for m in _many_masks(cfg, n=50):
        union: set[int] = set()
        for blk in m.target_blocks:
            union.update(blk)
        assert set(m.target_patches) == union


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_context_patches_subset_of_context_block(cfg):
    """context_patches must be a subset of the sampled context block.

    Only checks when the fallback path was not triggered (i.e. ctx_block
    minus target_union is non-empty).
    """
    rng = random.Random(0)
    for _ in range(200):
        m = sample_block_mask(N_H, N_W, rng=rng, **cfg)
        if set(m.context_block) - set(m.target_patches):
            assert set(m.context_patches).issubset(set(m.context_block)), (
                "context_patches outside context_block"
            )


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_correct_number_of_target_blocks(cfg):
    """Must always produce exactly cfg['n_targets'] target blocks."""
    rng = random.Random(0)
    for _ in range(50):
        m = sample_block_mask(N_H, N_W, rng=rng, **cfg)
        assert len(m.target_blocks) == cfg["n_targets"]


# Parametrised over n_targets too — exercises the n_targets path itself.
@pytest.mark.parametrize("n_targets", [1, 2, 3, 4])
def test_n_targets_dial(n_targets):
    rng = random.Random(0)
    for _ in range(30):
        m = sample_block_mask(N_H, N_W, n_targets=n_targets, rng=rng)
        assert len(m.target_blocks) == n_targets


# ---------------------------------------------------------------------------
# Coverage / area bounds — per config
# ---------------------------------------------------------------------------

# Relative slack on individual block area to absorb integer rounding of h, w.
_BLOCK_SLACK = 0.30


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_each_target_block_area_in_bounds(cfg):
    """Each target block area within scale × (1 ± _BLOCK_SLACK) of nominal."""
    lo = cfg["target_scale"][0] * N * (1 - _BLOCK_SLACK)
    hi = cfg["target_scale"][1] * N * (1 + _BLOCK_SLACK)
    for m in _many_masks(cfg, n=300, seed=7):
        for blk in m.target_blocks:
            area = len(blk)
            assert lo <= area <= hi, (
                f"Target block area {area} outside [{lo:.1f}, {hi:.1f}]"
            )


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_context_block_area_in_bounds(cfg):
    """
    Context block (before target removal) covers most patches.

    Lower bound: scale_lo × (1 − _BLOCK_SLACK) absorbs integer rounding and
    grid-edge clamping. The important property is that context is always
    large enough that the predictor sees a meaningful slice of the image —
    the per-config minimum is what mask_stats.py uses to GATE adoption.
    """
    lo = int(cfg["context_scale"][0] * N * (1 - _BLOCK_SLACK))
    hi = N
    for m in _many_masks(cfg, n=300, seed=99):
        area = len(m.context_block)
        assert lo <= area <= hi, (
            f"Context block area {area} outside [{lo}, {hi}]"
        )


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_target_union_below_ceiling(cfg):
    """Union of n_targets blocks cannot exceed scale_hi × n_targets × (1 + slack)
    (this is the no-overlap maximum, conservatively widened)."""
    ceiling = min(N, cfg["target_scale"][1] * cfg["n_targets"] * N * (1 + _BLOCK_SLACK))
    for m in _many_masks(cfg, n=300, seed=5):
        assert len(m.target_patches) <= ceiling, (
            f"Target union {len(m.target_patches)} exceeds ceiling {ceiling:.1f}"
        )


# ---------------------------------------------------------------------------
# Sorted outputs — config-agnostic structural property
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_outputs_are_sorted(cfg):
    for m in _many_masks(cfg, n=50):
        assert m.target_patches  == sorted(m.target_patches)
        assert m.context_patches == sorted(m.context_patches)
        assert m.context_block   == sorted(m.context_block)


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_collator_batch_format(cfg):
    """IJEPAMaskCollator produces correct keys and tensor types for any config."""
    collator = IJEPAMaskCollator(n_h=N_H, n_w=N_W, seed=0, **cfg)
    fake_img = torch.zeros(3, 96, 96)
    batch = collator([(fake_img, -1)] * 8)

    assert set(batch.keys()) == {"imgs", "target_idx", "context_idx"}
    assert batch["imgs"].shape == (8, 3, 96, 96)
    assert batch["target_idx"].dtype  == torch.long
    assert batch["context_idx"].dtype == torch.long


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_collator_no_overlap(cfg):
    """Collator output target/context must be disjoint for any config."""
    collator = IJEPAMaskCollator(n_h=N_H, n_w=N_W, seed=42, **cfg)
    fake_img = torch.zeros(3, 96, 96)
    for _ in range(50):
        batch = collator([(fake_img, -1)] * 4)
        tgt = set(batch["target_idx"].tolist())
        ctx = set(batch["context_idx"].tolist())
        assert tgt.isdisjoint(ctx), "Collator produced overlapping target/context"


@pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CFG_IDS)
def test_collator_different_masks_across_calls(cfg):
    """Successive calls should (usually) give different masks."""
    collator = IJEPAMaskCollator(n_h=N_H, n_w=N_W, seed=0, **cfg)
    fake_img = torch.zeros(3, 96, 96)
    masks = [
        collator([(fake_img, -1)] * 2)["target_idx"].tolist()
        for _ in range(20)
    ]
    unique = {tuple(m) for m in masks}
    assert len(unique) > 1, "Collator returned identical masks on every call"
