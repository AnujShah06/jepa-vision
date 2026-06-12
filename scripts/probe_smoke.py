"""
probe_smoke.py — Step 1.5d smoke test.

Runs the full probe harness at n=200 only (seed-0 JEPA encoder) to confirm
the grid code works end-to-end before the 3-seed full run.

Validation only — test set is quarantined.

Usage:
    uv run python scripts/probe_smoke.py \\
        --jepa_ckpt runs/tkqjawa0/best.ckpt \\
        --probe_epochs 30 \\
        --scratch_epochs 40

Full grid (run after all 3 seeds are trained):
    uv run python scripts/probe_sweep.py --jepa_ckpts <s0> <s1> <s2>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint
from src.data.stl10 import get_val_loader
from src.eval.probe import (
    ScratchClassifier,
    extract_features,
    get_probe_pool,
    stratified_sample,
    train_probe,
    train_scratch,
)
from src.models.jepa import VisionJEPA, VisionJEPAConfig

DATA_DIR = Path(__file__).parent.parent / "data"


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_loader(data_dir: Path, dataset_indices: list[int],
                 batch_size: int = 64) -> DataLoader:
    """Loader for a fixed subset of the STL-10 labeled train split."""
    import torchvision.transforms as T
    from torchvision.datasets import STL10

    transform = T.Compose([
        T.Resize(96, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(96),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = STL10(root=str(data_dir), split="train", transform=transform, download=False)
    return DataLoader(Subset(ds, dataset_indices), batch_size=batch_size,
                      shuffle=True, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt",      default="runs/tkqjawa0/best.ckpt")
    parser.add_argument("--n",              type=int, default=200)
    parser.add_argument("--probe_epochs",   type=int, default=30,
                        help="Linear probe epochs (smoke-test default = 30)")
    parser.add_argument("--scratch_epochs", type=int, default=40,
                        help="From-scratch epochs (smoke-test default = 40; full run = 200)")
    parser.add_argument("--seed",           type=int, default=0)
    parser.add_argument("--lr_list",        type=float, nargs="+",
                        default=[1e-3, 3e-4, 1e-4])
    args = parser.parse_args()

    device = _pick_device()
    n      = args.n
    assert n % 10 == 0, "--n must be divisible by 10 (stratified, 10 classes)"
    n_per_class = n // 10

    print(f"[smoke] device={device}  n={n} ({n_per_class}/class)  "
          f"ckpt={args.jepa_ckpt}")

    # ── load JEPA encoder ─────────────────────────────────────────────────
    jepa = VisionJEPA(VisionJEPAConfig()).eval()
    load_checkpoint(args.jepa_ckpt, model=jepa, map_location=device)
    jepa.to(device)

    # ── probe pool: labels and stratified subset ──────────────────────────
    print("[smoke] loading probe pool labels ...")
    probe_indices, probe_labels = get_probe_pool(DATA_DIR)
    print(f"[smoke] probe pool: {len(probe_indices)} images, "
          f"{len(set(probe_labels))} classes")

    train_subset = stratified_sample(probe_labels, n_per_class=n_per_class,
                                     seed=args.seed)
    # train_subset is indices into probe_indices/probe_labels; map to dataset indices
    train_ds_idx = [probe_indices[i] for i in train_subset]

    print(f"[smoke] sampled {len(train_ds_idx)} train images "
          f"({n_per_class}/class, seed={args.seed})")

    # ── val split ─────────────────────────────────────────────────────────
    val_loader  = get_val_loader(DATA_DIR, batch_size=256, num_workers=0)

    # ── extract features ──────────────────────────────────────────────────
    train_loader_plain = _make_loader(DATA_DIR, train_ds_idx)
    print("[smoke] extracting train features ...", end="  ", flush=True)
    train_feats, train_labels = extract_features(jepa, train_loader_plain, device)
    print(f"shape={train_feats.shape}")

    print("[smoke] extracting val features ...", end="  ", flush=True)
    val_feats, val_labels = extract_features(jepa, val_loader, device)
    print(f"shape={val_feats.shape}")

    # ── linear probe ──────────────────────────────────────────────────────
    print(f"[smoke] training linear probe ({args.probe_epochs} epochs) ...",
          end="  ", flush=True)
    _, probe_acc = train_probe(
        train_feats, train_labels, val_feats, val_labels,
        epochs=args.probe_epochs, lr=1e-3, device=device,
    )
    print(f"val_acc={probe_acc:.4f}")

    # ── from-scratch ──────────────────────────────────────────────────────
    train_loader_scratch = _make_loader(DATA_DIR, train_ds_idx, batch_size=64)
    print(f"[smoke] training from-scratch ({args.scratch_epochs} epochs × "
          f"{len(args.lr_list)} lrs) ...", end="  ", flush=True)
    scratch_acc, best_lr = train_scratch(
        train_loader_scratch, val_loader,
        lr_list=args.lr_list,
        epochs=args.scratch_epochs,
        device=device,
        seed=args.seed,
    )
    print(f"val_acc={scratch_acc:.4f}  best_lr={best_lr:.0e}")

    # ── summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  PROBE SMOKE TEST  |  n={n}  |  val only  |  seed-{args.seed}")
    print("=" * 60)
    print(f"  {'Method':<35}  {'Val Acc':>7}")
    print("-" * 60)
    print(f"  {'Frozen probe (linear)':<35}  {probe_acc:>7.4f}")
    print(f"  {f'From-scratch (best lr={best_lr:.0e})':<35}  {scratch_acc:>7.4f}")
    print("=" * 60)
    print()
    print("Note: smoke-test epoch budget is intentionally small.")
    print("Probe > scratch expected once JEPA is well-trained (3 seeds).")
    print()
    print("Feature shapes verified:")
    print(f"  train feats: {list(train_feats.shape)}")
    print(f"  val feats:   {list(val_feats.shape)}")


if __name__ == "__main__":
    main()
