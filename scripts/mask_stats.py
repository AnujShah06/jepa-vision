"""
mask_stats.py — Step 1.6 part 1: mask-statistics simulation.

Compares mask-yield distributions between:
  (a) current production config  : 4 tgt, target 0.15-0.20, context 0.85-1.00
  (b) proposed hard config        : 4 tgt, target 0.20-0.25, context 0.75-0.90

For 10,000 samples per config, reports per config:
  - realized target-union patches  (mean / p5 / p95)
  - realized context-block patches (mean / p5 / p95)  -- pre-removal size
  - realized context-after-removal (mean / p5 / p95)  -- what training sees
  - fallback trigger rate          (% of samples where ctx − tgt was empty)

GATE (hard config only):
  Pass iff context-after-removal p5 >= 20 patches AND fallback rate <= 2%.
  If gate fails on (b), apply levers in order:
    lever 1 : 3 target blocks instead of 4
    lever 2 : context 0.80-0.95
  Adopt the first candidate that passes.

Also renders 10 hard-config masks (using the adopted hard variant) to
reports/figures/mask_samples_hard.png.

Usage:
    uv run python scripts/mask_stats.py --n_samples 10000 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass, field
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
sys.path.insert(0, str(Path(__file__).parent))

from src.data.masking import sample_block_mask
from visualize_masks import (
    CTX_COLOR,
    OUTSIDE_CLR,
    TARGET_COLORS,
    _denorm,
    _make_overlay,
)

DATA_DIR = Path(__file__).parent.parent / "data"
FIG_HARD = Path(__file__).parent.parent / "reports" / "figures" / "mask_samples_hard.png"

N_H, N_W = 12, 12
N_TOTAL  = N_H * N_W   # 144

P5_FLOOR     = 20      # ctx-after-removal p5 must be >= this
FALLBACK_CAP = 0.02    # fallback rate must be <= this


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class MaskCfg:
    name: str
    n_targets: int = 4
    target_scale:  tuple[float, float] = (0.15, 0.20)
    target_aspect: tuple[float, float] = (0.75, 1.50)
    context_scale: tuple[float, float] = (0.85, 1.00)
    context_aspect: tuple[float, float] = (0.75, 1.50)

    def kwargs(self) -> dict:
        return {
            "n_targets":      self.n_targets,
            "target_scale":   self.target_scale,
            "target_aspect":  self.target_aspect,
            "context_scale":  self.context_scale,
            "context_aspect": self.context_aspect,
        }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(cfg: MaskCfg, n_samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    target_sizes:    list[int] = []
    ctx_block_sizes: list[int] = []
    ctx_after_sizes: list[int] = []
    fallback_count = 0

    for _ in range(n_samples):
        m = sample_block_mask(n_h=N_H, n_w=N_W, rng=rng, **cfg.kwargs())
        tgt  = set(m.target_patches)
        cblk = set(m.context_block)
        target_sizes.append(len(tgt))
        ctx_block_sizes.append(len(cblk))
        ctx_after_sizes.append(len(m.context_patches))
        if not (cblk - tgt):
            fallback_count += 1

    def _stats(xs: list[int]) -> dict:
        a = np.asarray(xs)
        return {
            "mean": float(a.mean()),
            "p5":   float(np.percentile(a, 5)),
            "p95":  float(np.percentile(a, 95)),
        }

    return {
        "name":          cfg.name,
        "cfg":           asdict(cfg),
        "target":        _stats(target_sizes),
        "ctx_block":     _stats(ctx_block_sizes),
        "ctx_after":     _stats(ctx_after_sizes),
        "fallback_rate": fallback_count / n_samples,
        "n_samples":     n_samples,
    }


def gate(stats: dict) -> tuple[bool, str]:
    p5 = stats["ctx_after"]["p5"]
    fb = stats["fallback_rate"]
    if p5 < P5_FLOOR:
        return False, f"ctx_after p5={p5:.1f} < {P5_FLOOR}"
    if fb > FALLBACK_CAP:
        return False, f"fallback_rate={fb*100:.2f}% > {FALLBACK_CAP*100:.0f}%"
    return True, f"ctx_after p5={p5:.1f}  fallback={fb*100:.2f}%"


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_stats(s: dict) -> None:
    cfg = s["cfg"]
    print(f"  {s['name']}")
    print(f"    n_targets={cfg['n_targets']}  "
          f"target_scale={tuple(cfg['target_scale'])}  "
          f"context_scale={tuple(cfg['context_scale'])}")
    print(f"    target_union  mean={s['target']['mean']:6.2f}  "
          f"p5={s['target']['p5']:5.1f}  p95={s['target']['p95']:5.1f}")
    print(f"    ctx_block     mean={s['ctx_block']['mean']:6.2f}  "
          f"p5={s['ctx_block']['p5']:5.1f}  p95={s['ctx_block']['p95']:5.1f}")
    print(f"    ctx_after     mean={s['ctx_after']['mean']:6.2f}  "
          f"p5={s['ctx_after']['p5']:5.1f}  p95={s['ctx_after']['p95']:5.1f}")
    print(f"    fallback      {s['fallback_rate']*100:5.2f}%")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_masks(cfg: MaskCfg, out_path: Path, seed: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = STL10(root=str(DATA_DIR), split="unlabeled", transform=tfm, download=False)
    step = max(1, len(ds) // 10)
    indices = list(range(0, len(ds), step))[:10]
    ds_sub = Subset(ds, indices)

    rng = random.Random(seed)
    fig, axes = plt.subplots(2, 5, figsize=(5 * 2.2, 2 * 2.4))
    axes_flat = axes.flatten()

    for ax_idx, (img_t, _) in enumerate(ds_sub):
        img_np  = _denorm(img_t)
        mask    = sample_block_mask(N_H, N_W, rng=rng, **cfg.kwargs())
        blended = _make_overlay(mask, img_np)
        ax = axes_flat[ax_idx]
        ax.imshow(blended)
        ax.set_title(
            f"ctx={len(mask.context_patches)} tgt={len(mask.target_patches)}",
            fontsize=7,
        )
        ax.axis("off")

    legend = [
        mpatches.Patch(facecolor=CTX_COLOR[:3], alpha=float(CTX_COLOR[3]),
                       label="context (encoder input)"),
        *[mpatches.Patch(facecolor=c[:3], alpha=float(c[3]),
                         label=f"target block {i}")
          for i, c in enumerate(TARGET_COLORS[: cfg.n_targets])],
        mpatches.Patch(facecolor=OUTSIDE_CLR[:3], alpha=float(OUTSIDE_CLR[3]),
                       label="outside (ignored)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Hard masking — {cfg.name}\n"
        f"target {cfg.target_scale}, context {cfg.context_scale}, "
        f"{cfg.n_targets} target blocks",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    print(f"[mask_stats] n_samples={args.n_samples}  seed={args.seed}")
    print(f"[mask_stats] grid {N_H}×{N_W} = {N_TOTAL} patches")
    print(f"[mask_stats] gate: ctx_after p5 >= {P5_FLOOR}  AND  fallback <= {FALLBACK_CAP*100:.0f}%")
    print()
    print("=" * 72)

    # ---- (a) current production -------------------------------------------
    cur = MaskCfg(
        name="(a) current  [phase1_ref]",
        n_targets=4,
        target_scale=(0.15, 0.20),
        context_scale=(0.85, 1.00),
    )
    s_cur = simulate(cur, args.n_samples, args.seed)
    print_stats(s_cur)
    print()

    # ---- (b) proposed hard, gated, with lever fallbacks -------------------
    candidates: list[MaskCfg] = [
        MaskCfg(
            name="(b1) hard  [4 tgt, target 0.20-0.25, ctx 0.75-0.90]",
            n_targets=4,
            target_scale=(0.20, 0.25),
            context_scale=(0.75, 0.90),
        ),
        MaskCfg(
            name="(b2) hard  [3 tgt, target 0.20-0.25, ctx 0.75-0.90]  -- lever 1 (3 targets)",
            n_targets=3,
            target_scale=(0.20, 0.25),
            context_scale=(0.75, 0.90),
        ),
        MaskCfg(
            name="(b3) hard  [3 tgt, target 0.20-0.25, ctx 0.80-0.95]  -- lever 1+2 (3 targets, narrower ctx)",
            n_targets=3,
            target_scale=(0.20, 0.25),
            context_scale=(0.80, 0.95),
        ),
    ]

    adopted: MaskCfg | None = None
    adopted_stats: dict | None = None
    all_stats: list[dict] = []

    for cand in candidates:
        s = simulate(cand, args.n_samples, args.seed)
        all_stats.append(s)
        print_stats(s)
        passed, reason = gate(s)
        verdict = f"GATE: PASS  ({reason})" if passed else f"GATE: FAIL — {reason}"
        print(f"    {verdict}")
        print()
        if passed and adopted is None:
            adopted = cand
            adopted_stats = s
            break    # first passing config wins

    print("=" * 72)
    if adopted is None:
        print("[mask_stats] NO CANDIDATE PASSED THE GATE — escalate to user, "
              "do NOT write hardmask config or render figure.")
        sys.exit(1)

    print(f"[mask_stats] ADOPTED: {adopted.name}")
    print(f"  n_targets    = {adopted.n_targets}")
    print(f"  target_scale = {adopted.target_scale}")
    print(f"  context_scale= {adopted.context_scale}")
    assert adopted_stats is not None
    print(f"  ctx_after p5 = {adopted_stats['ctx_after']['p5']:.1f}  "
          f"(>= {P5_FLOOR})")
    print(f"  fallback     = {adopted_stats['fallback_rate']*100:.2f}%  "
          f"(<= {FALLBACK_CAP*100:.0f}%)")

    # ---- render -----------------------------------------------------------
    render_masks(adopted, FIG_HARD, args.seed + 1)
    print(f"[mask_stats] figure → {FIG_HARD}")

    if adopted is not candidates[0]:
        print()
        print("[mask_stats] NOTE: base hard config (b1) did not pass the gate; "
              "adopted variant is a lever-adjusted version. "
              "Log this adjustment in DECISIONS.md.")


if __name__ == "__main__":
    main()
