"""
masking.py -- I-JEPA block masking for 96×96 images.

Algorithm (Assran et al. 2023, §3.2):
  1. Sample N_TARGETS=4 target blocks independently:
       each block covers target_scale (15-20%) of total patch area,
       with aspect ratio (h/w) drawn from target_aspect (0.75-1.5).
       Blocks may overlap; we take their union.
  2. Sample 1 context block:
       covers context_scale (85-100%) of patch area.
  3. context_patches = context_block - target_union.
       If removing target patches would empty the context, fall back to
       all non-target patches (guarantees context is never empty).

Why block masking instead of random:
  Random masking is easy -- nearby unmasked patches let the predictor
  interpolate rather than reason. Block masking forces coherent region
  prediction from distant context, producing stronger representations.

Grid convention:
  Patches are indexed row-major: patch index = row * n_w + col,
  where row in [0, n_h), col in [0, n_w).
  For the default 96×96 image with 8×8 patches: n_h = n_w = 12, N = 144.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MaskResult:
    """Output of sample_block_mask for a single image."""
    target_patches: list[int]          # sorted union of all target blocks
    context_patches: list[int]         # sorted context block minus target union
    target_blocks: list[list[int]]     # per-block patch lists (for visualisation)
    context_block: list[int]           # full context block before target removal
    n_h: int
    n_w: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_one_block(
    n_h: int,
    n_w: int,
    scale_lo: float,
    scale_hi: float,
    aspect_lo: float,
    aspect_hi: float,
    rng: random.Random,
) -> list[int]:
    """
    Sample a single rectangular block from the n_h × n_w patch grid.

    scale  : fraction of total N patches the block should cover.
    aspect : h/w ratio (height-to-width).

    Retries up to MAX_TRIES times to land valid integer dimensions; if all
    retries fail (can only happen at the extreme corners of the aspect/scale
    space near tiny grids), falls back to a single random patch.
    """
    N = n_h * n_w
    MAX_TRIES = 10

    for _ in range(MAX_TRIES):
        target_area = N * rng.uniform(scale_lo, scale_hi)
        ar = rng.uniform(aspect_lo, aspect_hi)        # h / w
        h = max(1, min(n_h, round(math.sqrt(target_area * ar))))
        w = max(1, min(n_w, round(math.sqrt(target_area / ar))))
        if h >= 1 and w >= 1:
            break
    else:
        h, w = 1, 1  # fallback: single patch

    top = rng.randint(0, n_h - h)
    left = rng.randint(0, n_w - w)

    return [
        (top + di) * n_w + (left + dj)
        for di in range(h)
        for dj in range(w)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_block_mask(
    n_h: int = 12,
    n_w: int = 12,
    n_targets: int = 4,
    target_scale: tuple[float, float] = (0.15, 0.20),
    target_aspect: tuple[float, float] = (0.75, 1.50),
    context_scale: tuple[float, float] = (0.85, 1.00),
    context_aspect: tuple[float, float] = (0.75, 1.50),
    rng: random.Random | None = None,
) -> MaskResult:
    """
    Sample one I-JEPA mask: 4 target blocks + 1 context block.

    Returns a MaskResult with:
      target_patches   sorted union of all target blocks
      context_patches  sorted context block minus target patches
      target_blocks    list of per-block patch lists (length = n_targets)
      context_block    full context block before removing target patches
    """
    if rng is None:
        rng = random.Random()

    N = n_h * n_w

    # 1. Sample n_targets independent target blocks
    blocks: list[list[int]] = []
    target_union: set[int] = set()
    for _ in range(n_targets):
        blk = _sample_one_block(n_h, n_w, *target_scale, *target_aspect, rng)
        blocks.append(blk)
        target_union.update(blk)

    # 2. Sample one context block
    ctx_blk = _sample_one_block(n_h, n_w, *context_scale, *context_aspect, rng)
    ctx_set: set[int] = set(ctx_blk)

    # 3. Remove target patches from context
    ctx_after = ctx_set - target_union
    if not ctx_after:
        # Fallback: all non-target patches (guarantees context non-empty
        # unless the targets cover every patch, which can't happen at
        # target_scale ≤ 0.20 × 4 = 80% on a well-sized grid).
        ctx_after = set(range(N)) - target_union
        if not ctx_after:
            ctx_after = {0}  # last-resort single patch (shouldn't reach here)

    return MaskResult(
        target_patches=sorted(target_union),
        context_patches=sorted(ctx_after),
        target_blocks=blocks,
        context_block=sorted(ctx_set),
        n_h=n_h,
        n_w=n_w,
    )


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class IJEPAMaskCollator:
    """
    DataLoader collate_fn that adds I-JEPA block masks to each batch.

    Samples ONE mask set per batch (all images share the same target/context
    split for efficient batched GPU computation). Mask diversity comes from
    refreshing the sample every batch. Per-image masking (Step 1.3 upgrade)
    requires variable-length handling or padding in the model's forward pass.

    Batch output keys:
      imgs        FloatTensor [B, 3, H, W]  normalised images
      target_idx  LongTensor  [N_tgt]       patch indices to predict
      context_idx LongTensor  [N_ctx]       patch indices the encoder sees

    The collator is stateful (holds an rng) so successive __call__ invocations
    give independent mask samples. Thread-safety note: DataLoader with
    num_workers > 0 serialises the collate function via pickling; the rng
    state is not shared across workers, which is fine -- each worker draws
    from its own independent sequence.
    """

    def __init__(
        self,
        n_h: int = 12,
        n_w: int = 12,
        n_targets: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.20),
        target_aspect: tuple[float, float] = (0.75, 1.50),
        context_scale: tuple[float, float] = (0.85, 1.00),
        context_aspect: tuple[float, float] = (0.75, 1.50),
        seed: int | None = None,
    ) -> None:
        self.n_h = n_h
        self.n_w = n_w
        self.n_targets = n_targets
        self.target_scale = target_scale
        self.target_aspect = target_aspect
        self.context_scale = context_scale
        self.context_aspect = context_aspect
        self.rng = random.Random(seed)

    def sample_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one (target_idx, context_idx) pair as LongTensors."""
        result = sample_block_mask(
            n_h=self.n_h,
            n_w=self.n_w,
            n_targets=self.n_targets,
            target_scale=self.target_scale,
            target_aspect=self.target_aspect,
            context_scale=self.context_scale,
            context_aspect=self.context_aspect,
            rng=self.rng,
        )
        target_idx = torch.tensor(result.target_patches, dtype=torch.long)
        context_idx = torch.tensor(result.context_patches, dtype=torch.long)
        return target_idx, context_idx

    def __call__(self, samples: list) -> dict:
        imgs = torch.stack([s[0] for s in samples])  # [B, 3, H, W]
        target_idx, context_idx = self.sample_masks()
        return {"imgs": imgs, "target_idx": target_idx, "context_idx": context_idx}
