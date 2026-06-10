"""
visualize_masks.py -- visual unit test for I-JEPA block masking.

Renders 10 sampled masks overlaid on real STL-10 unlabeled images and saves
to reports/figures/mask_samples.png.  Eyeball the output:
  - Dark-grey patches  = outside context block (encoder never sees)
  - Blue patches       = context (encoder input)
  - Red/orange/green/purple = 4 target blocks (predictor must reconstruct)
  - Black grid lines separate patches

Usage:
    uv run python scripts/visualize_masks.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Subset
from torchvision.datasets import STL10

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.masking import sample_block_mask

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_PATH = Path(__file__).parent.parent / "reports" / "figures" / "mask_samples.png"
N_IMAGES = 10
N_H, N_W = 12, 12
PATCH_PX = 8   # each patch is 8×8 pixels

# RGBA colours for context + 4 target blocks
CTX_COLOR    = np.array([0.25, 0.45, 0.85, 0.55])  # blue
OUTSIDE_CLR  = np.array([0.10, 0.10, 0.10, 0.45])  # dark grey
TARGET_COLORS = [
    np.array([0.95, 0.20, 0.20, 0.65]),  # red    -- block 0
    np.array([0.95, 0.60, 0.10, 0.65]),  # orange -- block 1
    np.array([0.15, 0.75, 0.20, 0.65]),  # green  -- block 2
    np.array([0.75, 0.15, 0.85, 0.65]),  # purple -- block 3
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _denorm(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation; return [H, W, 3] uint8."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _make_overlay(mask_result, img_array: np.ndarray) -> np.ndarray:
    """
    Alpha-composite a colour-coded patch overlay onto img_array [H, W, 3] uint8.
    Returns [H, W, 3] uint8.
    """
    H, W = img_array.shape[:2]
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    ctx_set   = set(mask_result.context_patches)
    tgt_union = set(mask_result.target_patches)

    # map each target patch to the colour of its last-owning block
    patch_color: dict[int, np.ndarray] = {}
    for blk_idx, blk in enumerate(mask_result.target_blocks):
        for pidx in blk:
            patch_color[pidx] = TARGET_COLORS[blk_idx % len(TARGET_COLORS)]

    for row in range(N_H):
        for col in range(N_W):
            pidx = row * N_W + col
            r0, r1 = row * PATCH_PX, (row + 1) * PATCH_PX
            c0, c1 = col * PATCH_PX, (col + 1) * PATCH_PX
            if pidx in tgt_union:
                overlay[r0:r1, c0:c1] = patch_color.get(pidx, TARGET_COLORS[0])
            elif pidx in ctx_set:
                overlay[r0:r1, c0:c1] = CTX_COLOR
            else:
                overlay[r0:r1, c0:c1] = OUTSIDE_CLR

    # 1-pixel black grid lines
    for r in range(0, H, PATCH_PX):
        overlay[r, :] = [0, 0, 0, 0.7]
    for c in range(0, W, PATCH_PX):
        overlay[:, c] = [0, 0, 0, 0.7]

    base  = img_array.astype(np.float32) / 255.0
    alpha = overlay[:, :, 3:4]
    blended = base * (1 - alpha) + overlay[:, :, :3] * alpha
    return (blended.clip(0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = STL10(root=str(DATA_DIR), split="unlabeled",
               transform=transform, download=True)
    # spread sample indices across the dataset for visual variety
    step = max(1, len(ds) // N_IMAGES)
    indices = list(range(0, len(ds), step))[:N_IMAGES]
    ds_sub = Subset(ds, indices)

    rng = random.Random(1234)

    ncols = 5
    nrows = math.ceil(N_IMAGES / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_idx, (img_t, _) in enumerate(ds_sub):
        img_np = _denorm(img_t)
        mask   = sample_block_mask(N_H, N_W, rng=rng)
        blended = _make_overlay(mask, img_np)

        ax = axes_flat[ax_idx]
        ax.imshow(blended)
        ax.set_title(
            f"ctx={len(mask.context_patches)} tgt={len(mask.target_patches)}",
            fontsize=7,
        )
        ax.axis("off")

    for ax in axes_flat[N_IMAGES:]:
        ax.set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=CTX_COLOR[:3],   alpha=float(CTX_COLOR[3]),
                       label="context (encoder input)"),
        *[mpatches.Patch(facecolor=c[:3], alpha=float(c[3]),
                         label=f"target block {i}")
          for i, c in enumerate(TARGET_COLORS)],
        mpatches.Patch(facecolor=OUTSIDE_CLR[:3], alpha=float(OUTSIDE_CLR[3]),
                       label="outside (ignored)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("I-JEPA Block Masking — 10 sample masks on STL-10 images",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
