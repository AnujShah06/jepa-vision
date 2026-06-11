"""
k_sweep.py -- Step 1.4 K-sweep: energy variance and clean-vs-corrupted
separation on the STL-10 formal validation split for K ∈ {1, 4, 8, 16}.

Validation only.  The corruption/OOD test benchmark is quarantined until
all three production seeds exist.

Usage:
    uv run python scripts/k_sweep.py \\
        --ckpt runs/tkqjawa0/best.ckpt \\
        --out reports/figures/k_sweep.png

The script:
  1. Loads the model from the checkpoint.
  2. Loads the formal 1,000-image validation split (data/splits/stl10_val_idx.json).
  3. Creates corrupted counterparts: gaussian_noise at severity 3.
  4. For each K ∈ {1, 4, 8, 16}: computes energy for both sets and measures
       - AUROC (clean = negative class, corrupted = positive class)
       - std of clean-image energy scores (proxy for per-sample noise floor)
  5. Saves a 2-panel plot to --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint
from src.data.stl10 import get_val_loader
from src.eval.energy import image_energy
from src.eval.evaluate import auroc
from src.models.jepa import VisionJEPA, VisionJEPAConfig


# ---------------------------------------------------------------------------
# Corruption helper
# ---------------------------------------------------------------------------

def _corrupt_gaussian(imgs: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    Apply gaussian_noise corruption (imagecorruptions) to a batch of
    ImageNet-normalised tensors.  Corruption operates in uint8 space,
    so we denormalise, corrupt, then renormalise.

    Returns a tensor with the same shape and dtype as `imgs`.
    """
    from imagecorruptions import corrupt

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Denormalise → [0, 1] → uint8
    imgs_01 = (imgs.cpu().float() * STD + MEAN).clamp(0.0, 1.0)
    imgs_u8 = (imgs_01 * 255).byte().numpy()   # [B, 3, H, W]

    corrupted = []
    for i in range(imgs_u8.shape[0]):
        hwc = imgs_u8[i].transpose(1, 2, 0)          # [H, W, 3]
        out = corrupt(hwc, corruption_name="gaussian_noise", severity=severity)
        corrupted.append(out.transpose(2, 0, 1))      # back to [3, H, W]

    corrupted_u8 = np.stack(corrupted, axis=0)        # [B, 3, H, W]
    corrupted_f  = torch.from_numpy(corrupted_u8).float() / 255.0
    corrupted_norm = (corrupted_f - MEAN.expand_as(corrupted_f)
                      ) / STD.expand_as(corrupted_f)
    return corrupted_norm.to(imgs.dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      default="runs/tkqjawa0/best.ckpt")
    parser.add_argument("--val_split", default=None,
                        help="Path to val split JSON (default: data/splits/stl10_val_idx.json)")
    parser.add_argument("--severity",  type=int, default=3,
                        help="imagecorruptions severity (1-5)")
    parser.add_argument("--out",       default="reports/figures/k_sweep.png")
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    print(f"[k_sweep] device={device}  ckpt={args.ckpt}")

    # -- load model ----------------------------------------------------------
    model = VisionJEPA(VisionJEPAConfig())
    load_checkpoint(args.ckpt, model=model, map_location=device)
    model.to(device).eval()

    # -- load formal validation split (1000 images, data/splits/stl10_val_idx.json) --
    data_dir = Path(__file__).parent.parent / "data"
    loader = get_val_loader(
        data_dir,
        split_file=args.val_split,
        batch_size=1000,   # load all at once
        num_workers=0,
    )
    imgs_batch = next(iter(loader))
    clean_imgs = imgs_batch[0]                           # [N, 3, H, W]
    print(f"[k_sweep] loaded {clean_imgs.shape[0]} formal val images "
          f"(data/splits/stl10_val_idx.json)")

    # -- create corrupted counterparts ----------------------------------------
    print(f"[k_sweep] applying gaussian_noise severity={args.severity} ...")
    corrupt_imgs = _corrupt_gaussian(clean_imgs, severity=args.severity)
    print("[k_sweep] corruption done")

    # -- K sweep -------------------------------------------------------------
    K_vals  = [1, 4, 8, 16]
    results = {}   # K -> {"auroc": float, "clean_std": float,
                   #        "clean_mean": float, "corrupt_mean": float}

    for K in K_vals:
        print(f"[k_sweep] K={K} ...", end="  ", flush=True)

        clean_result  = image_energy(model, clean_imgs,  K=K,
                                     seed=args.seed, device=device)
        corrupt_result = image_energy(model, corrupt_imgs, K=K,
                                      seed=args.seed, device=device)

        clean_e  = clean_result["energy"].cpu()
        corrupt_e = corrupt_result["energy"].cpu()

        auc = auroc(neg_scores=clean_e, pos_scores=corrupt_e)
        results[K] = {
            "auroc":        auc,
            "clean_std":    float(clean_e.std()),
            "clean_mean":   float(clean_e.mean()),
            "corrupt_mean": float(corrupt_e.mean()),
        }
        print(f"AUROC={auc:.3f}  clean_mean={results[K]['clean_mean']:.4f}  "
              f"corrupt_mean={results[K]['corrupt_mean']:.4f}  "
              f"clean_std={results[K]['clean_std']:.4f}")

    # -- plot ----------------------------------------------------------------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    k_x     = K_vals
    auroc_y = [results[k]["auroc"]     for k in K_vals]
    std_y   = [results[k]["clean_std"] for k in K_vals]

    # Panel 1: AUROC vs K
    ax = axes[0]
    ax.plot(k_x, auroc_y, marker="o", linewidth=2, color="steelblue")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xlabel("K (mask samples per image)")
    ax.set_ylabel("AUROC")
    ax.set_title("Clean vs. gaussian-noise AUROC")
    ax.set_xticks(k_x)
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate each point
    for k, a in zip(k_x, auroc_y):
        ax.annotate(f"{a:.3f}", (k, a),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)

    # Panel 2: clean energy std vs K
    ax = axes[1]
    ax.plot(k_x, std_y, marker="s", linewidth=2, color="darkorange")
    ax.set_xlabel("K (mask samples per image)")
    ax.set_ylabel("Std dev of energy (clean images)")
    ax.set_title("Energy variance vs. K (clean images)")
    ax.set_xticks(k_x)
    ax.grid(True, alpha=0.3)

    for k, s in zip(k_x, std_y):
        ax.annotate(f"{s:.4f}", (k, s),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)

    fig.suptitle(
        f"K-sweep  |  n={clean_imgs.shape[0]} (formal val)  corruption=gaussian_noise(sev={args.severity})"
        f"  ckpt={Path(args.ckpt).parent.name}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[k_sweep] figure saved → {args.out}")

    # -- text summary --------------------------------------------------------
    print("\n" + "=" * 55)
    print(f"{'K':>4}  {'AUROC':>7}  {'clean μ':>9}  {'corrupt μ':>10}  {'clean σ':>9}")
    print("-" * 55)
    for k in K_vals:
        r = results[k]
        print(f"{k:>4}  {r['auroc']:>7.4f}  {r['clean_mean']:>9.4f}  "
              f"{r['corrupt_mean']:>10.4f}  {r['clean_std']:>9.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
