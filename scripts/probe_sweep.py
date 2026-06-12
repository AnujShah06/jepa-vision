"""
probe_sweep.py — Full-budget probe grid on a single JEPA encoder.

Runs the label-scarce transfer evaluation (PLAYBOOK §1.5d) at
n ∈ {40, 200, 400, 4000}, validation only, and writes a Markdown report.

  (A) Frozen-encoder linear probe (100 epochs by default)
  (B) From-scratch ViT-Tiny (200 epochs, lr ∈ {1e-3, 3e-4, 1e-4} swept on val)

Test set stays sealed — this script never touches it.

Usage:
    uv run python scripts/probe_sweep.py \\
        --jepa_ckpt runs/tkqjawa0/best.ckpt \\
        --out reports/probe_seed0_val.md \\
        --seed 0

Args:
    --jepa_ckpt      Path to trained JEPA checkpoint.
    --out            Output Markdown path (default: reports/probe_seed0_val.md).
    --seed           Probe sampling seed (default: 0).
    --probe_epochs   Linear probe training epochs (default: 100).
    --scratch_epochs From-scratch training epochs (default: 200).
    --n_list         Space-separated n values (default: 40 200 400 4000).
    --batch_size     DataLoader batch size for scratch training (default: 128).
    --no_amp         Disable bfloat16 AMP (use for debugging).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint
from src.data.stl10 import get_val_loader
from src.eval.probe import (
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


def _make_train_loader(
    data_dir: Path, dataset_indices: list[int], batch_size: int
) -> DataLoader:
    """Loader for a fixed subset of STL-10 labeled train (actual images for scratch)."""
    import torchvision.transforms as T
    from torchvision.datasets import STL10

    transform = T.Compose([
        T.RandomResizedCrop(96, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = STL10(root=str(data_dir), split="train", transform=transform, download=False)
    return DataLoader(
        Subset(ds, dataset_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt",      default="runs/tkqjawa0/best.ckpt")
    parser.add_argument("--out",            default="reports/probe_seed0_val.md")
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--probe_epochs",   type=int,   default=100)
    parser.add_argument("--scratch_epochs", type=int,   default=200)
    parser.add_argument("--n_list",         type=int,   nargs="+",
                        default=[40, 200, 400, 4000])
    parser.add_argument("--batch_size",     type=int,   default=128)
    parser.add_argument("--no_amp",         action="store_true")
    args = parser.parse_args()

    device  = _pick_device()
    use_amp = not args.no_amp
    t0_all  = time.time()

    print(f"[sweep] device={device}  amp={'on' if use_amp else 'off'}  "
          f"probe_ep={args.probe_epochs}  scratch_ep={args.scratch_epochs}")
    print(f"[sweep] ckpt={args.jepa_ckpt}  n_list={args.n_list}  seed={args.seed}")

    # ── load JEPA encoder ─────────────────────────────────────────────────
    jepa = VisionJEPA(VisionJEPAConfig()).eval()
    load_checkpoint(args.jepa_ckpt, model=jepa, map_location=device)
    jepa.to(device)
    print("[sweep] encoder loaded")

    # ── val features (extracted once) ────────────────────────────────────
    val_loader_plain = get_val_loader(DATA_DIR, batch_size=256, num_workers=0)
    print("[sweep] extracting val features ...", end="  ", flush=True)
    t0 = time.time()
    val_feats, val_labels = extract_features(jepa, val_loader_plain, device)
    print(f"{val_feats.shape}  ({_fmt_time(time.time()-t0)})")

    # ── probe pool: all 4000 images' features + labels ────────────────────
    print("[sweep] loading probe pool labels ...", end="  ", flush=True)
    probe_indices, probe_labels = get_probe_pool(DATA_DIR)
    print(f"{len(probe_indices)} images")

    print("[sweep] extracting all probe features ...", end="  ", flush=True)
    t0 = time.time()

    import torchvision.transforms as T
    from torchvision.datasets import STL10
    _transform_plain = T.Compose([
        T.Resize(96, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(96),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _probe_ds = STL10(root=str(DATA_DIR), split="train",
                      transform=_transform_plain, download=False)
    probe_feat_loader = DataLoader(
        Subset(_probe_ds, probe_indices),
        batch_size=256, shuffle=False, num_workers=0,
    )
    all_probe_feats, all_probe_labels = extract_features(jepa, probe_feat_loader, device)
    print(f"{all_probe_feats.shape}  ({_fmt_time(time.time()-t0)})")

    # ── val loader for scratch evaluation ─────────────────────────────────
    val_loader_scratch = get_val_loader(DATA_DIR, batch_size=256, num_workers=0)

    # ── grid ─────────────────────────────────────────────────────────────
    rows: list[dict] = []

    for n in args.n_list:
        assert n % 10 == 0, f"n={n} must be divisible by 10"
        n_per_class = n // 10
        print(f"\n[sweep] ── n={n} ({n_per_class}/class) ──────────────────────")

        # Stratified indices into the probe pool (indices into all_probe_feats)
        subset_idx = stratified_sample(
            probe_labels, n_per_class=n_per_class, seed=args.seed
        )
        # Map back to dataset-level indices for the scratch DataLoader
        train_ds_idx = [probe_indices[i] for i in subset_idx]

        train_feats  = all_probe_feats[subset_idx]
        train_labels = all_probe_labels[subset_idx]

        # (A) Linear probe
        print(f"  [probe] {args.probe_epochs} epochs ...", end="  ", flush=True)
        t0 = time.time()
        _, probe_acc = train_probe(
            train_feats, train_labels, val_feats, val_labels,
            epochs=args.probe_epochs,
            lr=1e-3,
            device=device,
        )
        print(f"val_acc={probe_acc:.4f}  ({_fmt_time(time.time()-t0)})")

        # (B) From-scratch
        train_loader_scratch = _make_train_loader(DATA_DIR, train_ds_idx, args.batch_size)
        print(f"  [scratch] {args.scratch_epochs} ep × 3 lrs ...", end="  ", flush=True)
        t0 = time.time()
        scratch_acc, best_lr = train_scratch(
            train_loader_scratch, val_loader_scratch,
            lr_list=(1e-3, 3e-4, 1e-4),
            epochs=args.scratch_epochs,
            device=device,
            seed=args.seed,
            use_amp=use_amp,
        )
        print(f"val_acc={scratch_acc:.4f}  best_lr={best_lr:.0e}  ({_fmt_time(time.time()-t0)})")

        gap = probe_acc - scratch_acc
        rows.append({
            "n":          n,
            "probe_acc":  probe_acc,
            "scratch_acc": scratch_acc,
            "gap":        gap,
            "best_lr":    best_lr,
        })

    total_time = time.time() - t0_all

    # ── print table ───────────────────────────────────────────────────────
    print()
    _print_table(rows, args)
    print(f"\n[sweep] total wall time: {_fmt_time(total_time)}")

    # ── save report ───────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(out_path, rows, args, total_time)
    print(f"[sweep] saved → {out_path}")


def _print_table(rows: list[dict], args) -> None:
    print("=" * 72)
    print(f"  PROBE GRID (val only)  |  ckpt={args.jepa_ckpt}  |  seed={args.seed}")
    print(f"  probe={args.probe_epochs}ep  scratch={args.scratch_epochs}ep × lr{{1e-3,3e-4,1e-4}}")
    print("=" * 72)
    print(f"  {'n':>5}  {'Probe':>7}  {'Scratch':>8}  {'Gap':>7}  {'BestLR':>8}")
    print("-" * 72)
    for r in rows:
        sign = "+" if r["gap"] >= 0 else ""
        print(f"  {r['n']:>5}  {r['probe_acc']:>7.4f}  {r['scratch_acc']:>8.4f}  "
              f"{sign}{r['gap']:>6.4f}  {r['best_lr']:>8.0e}")
    print("=" * 72)


def _write_report(path: Path, rows: list[dict], args, total_time: float) -> None:
    lines = [
        "# Probe grid — seed 0, validation only",
        "",
        f"Checkpoint: `{args.jepa_ckpt}`  |  seed: {args.seed}  |  "
        f"wall time: {_fmt_time(total_time)}",
        "",
        "**Probe:** frozen JEPA context encoder → mean-pool → linear head, "
        f"{args.probe_epochs} epochs, lr=1e-3.",
        "",
        "**From-scratch:** identical ViT-Tiny trained end-to-end, "
        f"{args.scratch_epochs} epochs, lr swept over {{1e-3, 3e-4, 1e-4}}, "
        "best on val.",
        "",
        "Val split: formal 1 000-image stratified set (sealed; not used for "
        "training).  Test set: quarantined.",
        "",
        "| n | Frozen probe | From-scratch | Gap (P−S) | Best LR |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        sign = "+" if r["gap"] >= 0 else ""
        lines.append(
            f"| {r['n']} | {r['probe_acc']:.4f} | {r['scratch_acc']:.4f} | "
            f"{sign}{r['gap']:.4f} | {r['best_lr']:.0e} |"
        )
    lines += [
        "",
        "**Gap = Frozen probe − From-scratch.**  Positive = JEPA pretraining helps.",
        "",
        "---",
        "",
        f"*Generated by `scripts/probe_sweep.py`.*",
        f"*probe_epochs={args.probe_epochs}, scratch_epochs={args.scratch_epochs}, "
        f"batch_size={args.batch_size}, AMP={'on' if not args.no_amp else 'off'}*",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
