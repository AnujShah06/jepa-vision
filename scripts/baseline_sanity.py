"""
baseline_sanity.py -- Step 1.5 baseline AUROC sanity check.

Runs all four baselines + trained JEPA on the formal val split with
gaussian_noise severity=3. Validation only; test set is quarantined.

Usage:
    uv run python scripts/baseline_sanity.py \\
        --jepa_ckpt runs/tkqjawa0/best.ckpt \\
        --severity 3

Output: console table of AUROC + energy statistics per baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint
from src.data.stl10 import get_val_loader
from src.eval.baselines import (
    extract_encoder_features,
    fit_mahalanobis,
    mae_energy,
    mahalanobis_energy,
    pixel_stats_energy,
    random_init_energy,
)
from src.eval.energy import image_energy
from src.eval.evaluate import auroc
from src.models.jepa import VisionJEPA, VisionJEPAConfig
from src.models.mae import PixelMAE, PixelMAEConfig


# ---------------------------------------------------------------------------
# Corruption helper (identical to k_sweep.py)
# ---------------------------------------------------------------------------

def _corrupt_gaussian(imgs: torch.Tensor, severity: int = 3) -> torch.Tensor:
    from imagecorruptions import corrupt
    import numpy as np

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    imgs_01 = (imgs.cpu().float() * STD + MEAN).clamp(0.0, 1.0)
    imgs_u8 = (imgs_01 * 255).byte().numpy()

    corrupted = []
    for i in range(imgs_u8.shape[0]):
        hwc = imgs_u8[i].transpose(1, 2, 0)
        out = corrupt(hwc, corruption_name="gaussian_noise", severity=severity)
        corrupted.append(out.transpose(2, 0, 1))

    corrupted_f = torch.from_numpy(
        __import__("numpy").stack(corrupted).astype("float32")
    ) / 255.0
    return ((corrupted_f - MEAN.expand_as(corrupted_f))
            / STD.expand_as(corrupted_f)).to(imgs.dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", default="runs/tkqjawa0/best.ckpt",
                        help="Trained JEPA checkpoint (seed-0)")
    parser.add_argument("--mae_ckpt",  default=None,
                        help="Trained MAE checkpoint (optional; if omitted uses random-init MAE)")
    parser.add_argument("--severity",  type=int, default=3)
    parser.add_argument("--K",         type=int, default=8)
    parser.add_argument("--n_fit",     type=int, default=2048,
                        help="Images used to fit the Mahalanobis Gaussian")
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    data_dir = Path(__file__).parent.parent / "data"

    print(f"[sanity] device={device}  JEPA_ckpt={args.jepa_ckpt}  K={args.K}")

    # ── load val images ────────────────────────────────────────────────────
    loader = get_val_loader(data_dir, batch_size=1000, num_workers=0)
    batch  = next(iter(loader))
    clean_imgs = batch[0]                             # [1000, 3, 96, 96]
    print(f"[sanity] loaded {clean_imgs.shape[0]} formal val images")

    print(f"[sanity] applying gaussian_noise sev={args.severity} ...")
    corrupt_imgs = _corrupt_gaussian(clean_imgs, severity=args.severity)

    # ── load trained JEPA ─────────────────────────────────────────────────
    jepa = VisionJEPA(VisionJEPAConfig()).eval()
    load_checkpoint(args.jepa_ckpt, model=jepa, map_location=device)
    jepa.to(device)

    # ── helper: compute AUROC + stats for a pair of energy vectors ─────────
    def _report(name: str, clean_e: torch.Tensor, corrupt_e: torch.Tensor) -> dict:
        auc  = auroc(neg_scores=clean_e, pos_scores=corrupt_e)
        return {
            "name":        name,
            "auroc":       auc,
            "clean_mean":  float(clean_e.mean()),
            "corrupt_mean":float(corrupt_e.mean()),
            "clean_std":   float(clean_e.std()),
        }

    rows = []

    # ── 1. Pixel statistics ────────────────────────────────────────────────
    print("[sanity] 1/5  pixel stats ...", end="  ", flush=True)
    rows.append(_report(
        "Pixel std (trivial floor)",
        pixel_stats_energy(clean_imgs),
        pixel_stats_energy(corrupt_imgs),
    ))
    print(f"AUROC={rows[-1]['auroc']:.4f}")

    # ── 2. Random-init encoder ─────────────────────────────────────────────
    print(f"[sanity] 2/5  random-init encoder K={args.K} ...", end="  ", flush=True)
    rows.append(_report(
        f"Random-init JEPA (K={args.K})",
        random_init_energy(clean_imgs,  K=args.K, seed=args.seed, device=device),
        random_init_energy(corrupt_imgs, K=args.K, seed=args.seed, device=device),
    ))
    print(f"AUROC={rows[-1]['auroc']:.4f}")

    # ── 3. Mahalanobis ────────────────────────────────────────────────────
    print(f"[sanity] 3/5  Mahalanobis (fit on {args.n_fit} unlabeled) ...",
          end="  ", flush=True)
    from src.data.stl10 import get_eval_loader
    fit_loader = get_eval_loader(
        data_dir, split="unlabeled",
        n_images=args.n_fit,
        batch_size=256, num_workers=0,
    )
    # get_eval_loader(split="unlabeled") hits the 100k set — use STL10 unlabeled loader
    # but get_eval_loader expects labeled split; fall back to direct construction:
    from torch.utils.data import DataLoader, Subset
    import torchvision.transforms as T
    from torchvision.datasets import STL10

    transform = T.Compose([
        T.Resize(96, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(96),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    unlabeled_ds = STL10(root=str(data_dir), split="unlabeled",
                         transform=transform, download=True)
    unlabeled_ds = Subset(unlabeled_ds, list(range(args.n_fit)))
    fit_loader   = DataLoader(unlabeled_ds, batch_size=256, num_workers=0)

    fit_feats = extract_encoder_features(jepa, fit_loader, device, n_samples=args.n_fit)
    maha_mean, maha_prec = fit_mahalanobis(fit_feats)
    rows.append(_report(
        "Mahalanobis (trained features)",
        mahalanobis_energy(clean_imgs,  jepa, maha_mean, maha_prec, device=device),
        mahalanobis_energy(corrupt_imgs, jepa, maha_mean, maha_prec, device=device),
    ))
    print(f"AUROC={rows[-1]['auroc']:.4f}")

    # ── 4. MAE (untrained or loaded) ──────────────────────────────────────
    mae_label = f"MAE reconstruction (K={args.K}"
    mae_model = PixelMAE(PixelMAEConfig()).eval()
    if args.mae_ckpt:
        load_checkpoint(args.mae_ckpt, model=mae_model, map_location=device)
        mae_label += ", trained)"
    else:
        mae_label += ", untrained)"
    mae_model.to(device)

    print(f"[sanity] 4/5  MAE K={args.K} [{mae_label}] ...", end="  ", flush=True)
    rows.append(_report(
        mae_label,
        mae_energy(mae_model, clean_imgs,  K=args.K, seed=args.seed, device=device),
        mae_energy(mae_model, corrupt_imgs, K=args.K, seed=args.seed, device=device),
    ))
    print(f"AUROC={rows[-1]['auroc']:.4f}")

    # ── 5. JEPA (trained) — reference ────────────────────────────────────
    print(f"[sanity] 5/5  JEPA trained K={args.K} (reference) ...",
          end="  ", flush=True)
    jepa.eval()
    rows.append(_report(
        f"JEPA energy K={args.K} (trained, seed-0)",
        image_energy(jepa, clean_imgs,  K=args.K, seed=args.seed, device=device)["energy"],
        image_energy(jepa, corrupt_imgs, K=args.K, seed=args.seed, device=device)["energy"],
    ))
    print(f"AUROC={rows[-1]['auroc']:.4f}")

    # ── print table ───────────────────────────────────────────────────────
    W = 42
    print()
    print("=" * 75)
    print(f"  BASELINE SANITY  |  gaussian_noise sev={args.severity}  |  n=1000 val  |  K={args.K}")
    print("=" * 75)
    print(f"  {'Baseline':<{W}}  {'AUROC':>7}  {'clean μ':>9}  {'corrupt μ':>10}  {'clean σ':>8}")
    print("-" * 75)
    for r in rows[:-1]:
        print(f"  {r['name']:<{W}}  {r['auroc']:>7.4f}  "
              f"{r['clean_mean']:>9.4f}  {r['corrupt_mean']:>10.4f}  {r['clean_std']:>8.4f}")
    print("  " + "-" * 71)
    r = rows[-1]
    print(f"  {r['name']:<{W}}  {r['auroc']:>7.4f}  "
          f"{r['clean_mean']:>9.4f}  {r['corrupt_mean']:>10.4f}  {r['clean_std']:>8.4f}")
    print("=" * 75)
    print()
    print("Interpretation: AUROC > 0.5 = energy detects corruption.")
    print("Gate 1B requires JEPA clearly above random-init and pixel baselines.")


if __name__ == "__main__":
    main()
