"""
probe_vs_epoch.py — Step 1.5e: probe accuracy vs pretraining epoch curve.

Discriminates between:
  H1 (curve rising at ep 150) → encoder needs more training
  H2 (curve plateaued ~100-120) → masking task too easy

For each checkpoint in {30, 60, 90, 120, 150}:
  - Extract TARGET encoder features + CONTEXT encoder features (both n=4000 and n=200)
  - Z-score normalise from training set statistics
  - Run locked probe protocol (lr sweep {3e-3,1e-3,3e-4} × 200 epochs) at each n
  - Record target val acc, context val acc, target-vs-context gap

Output:
  reports/figures/probe_vs_epoch.png  — both n curves + target/context lines
  reports/probe_vs_epoch.md           — table + pre-registered reading

Usage:
    uv run python scripts/probe_vs_epoch.py \\
        --ckpt_dir /tmp/ckpts_probe_diag \\
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint
from src.data.stl10 import get_val_loader
from src.eval.probe import get_probe_pool, stratified_sample, train_probe
from src.models.jepa import VisionJEPA, VisionJEPAConfig

DATA_DIR = Path(__file__).parent.parent / "data"
REPORTS  = Path(__file__).parent.parent / "reports"


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def _extract_all_patches(model, encoder_attr: str, loader, device: str) -> tuple:
    """Return (tokens [N,144,d], labels [N]) on CPU for entire loader."""
    model.eval()
    encoder = getattr(model, encoder_attr)
    all_tok, all_lbl = [], []
    for batch in loader:
        imgs = batch[0].to(device)
        lbl  = batch[1]
        tok  = encoder(model.patch_embed(imgs) + model.pos_embed)  # [B,144,d]
        all_tok.append(tok.cpu())
        all_lbl.append(lbl)
    return torch.cat(all_tok), torch.cat(all_lbl).long()


def _mean_pool_zscore(tokens_tr, tokens_vl):
    """Mean-pool patches then z-score from train statistics."""
    tr = tokens_tr.mean(1)   # [N_tr, d]
    vl = tokens_vl.mean(1)   # [N_vl, d]
    mu  = tr.mean(0)
    sig = tr.std(0).clamp(min=1e-6)
    return (tr - mu) / sig, (vl - mu) / sig


def _best_probe(train_f, train_l, val_f, val_l,
                lr_list, epochs, device) -> tuple[float, float]:
    best_acc, best_lr = -1.0, lr_list[0]
    for lr in lr_list:
        _, acc = train_probe(train_f, train_l, val_f, val_l,
                             epochs=epochs, lr=lr, device=device)
        if acc > best_acc:
            best_acc, best_lr = acc, lr
    return best_acc, best_lr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="/tmp/ckpts_probe_diag",
                        help="Directory with epoch_00XX.ckpt files")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--epochs_list", type=int, nargs="+",
                        default=[30, 60, 90, 120, 150])
    args = parser.parse_args()

    device   = _pick_device()
    ckpt_dir = Path(args.ckpt_dir)
    LR_LIST  = (3e-3, 1e-3, 3e-4)
    PROBE_EP = 200

    print(f"[pvep] device={device}  ckpt_dir={ckpt_dir}  seed={args.seed}")

    # ── probe pool ────────────────────────────────────────────────────────
    probe_indices, probe_labels = get_probe_pool(DATA_DIR)

    # n=4000: full probe pool
    idx4000 = stratified_sample(probe_labels, n_per_class=400, seed=args.seed)
    # n=200: 20/class
    idx200  = stratified_sample(probe_labels, n_per_class=20,  seed=args.seed)

    # ── build image loaders (plain, no masking) ───────────────────────────
    import torchvision.transforms as T
    from torchvision.datasets import STL10
    from torch.utils.data import DataLoader, Subset

    _tfm = T.Compose([
        T.Resize(96, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(96),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _ds = STL10(root=str(DATA_DIR), split="train", transform=_tfm, download=False)

    ds4000 = [probe_indices[i] for i in idx4000]
    ds200  = [probe_indices[i] for i in idx200]

    loader4000 = DataLoader(Subset(_ds, ds4000), batch_size=256,
                            shuffle=False, num_workers=0)
    loader200  = DataLoader(Subset(_ds, ds200),  batch_size=256,
                            shuffle=False, num_workers=0)
    val_loader = get_val_loader(DATA_DIR, batch_size=256, num_workers=0)

    # ── iterate checkpoints ───────────────────────────────────────────────
    rows: list[dict] = []

    for ep in args.epochs_list:
        ckpt_path = ckpt_dir / f"epoch_{ep:04d}.ckpt"
        # epoch_0150.ckpt exists in both /tmp dir and runs/tkqjawa0/
        if not ckpt_path.exists():
            # fall back to runs/ for ep 150
            alt = Path("runs/tkqjawa0/epoch_0150.ckpt")
            if ep == 150 and alt.exists():
                ckpt_path = alt
            else:
                print(f"[pvep] MISSING {ckpt_path}, skipping")
                continue

        print(f"\n[pvep] ── ep={ep}  ({ckpt_path.name}) ──────────────────")
        jepa = VisionJEPA(VisionJEPAConfig()).eval()
        load_checkpoint(str(ckpt_path), model=jepa, map_location=device)
        jepa.to(device)

        t0 = time.time()
        # extract target + context tokens for n=4000, n=200, val
        tgt4000, lbl4000 = _extract_all_patches(jepa, "target_encoder",  loader4000, device)
        ctx4000, _       = _extract_all_patches(jepa, "context_encoder", loader4000, device)
        tgt200,  lbl200  = _extract_all_patches(jepa, "target_encoder",  loader200,  device)
        ctx200,  _       = _extract_all_patches(jepa, "context_encoder", loader200,  device)
        tgt_val, lbl_val = _extract_all_patches(jepa, "target_encoder",  val_loader, device)
        ctx_val, _       = _extract_all_patches(jepa, "context_encoder", val_loader, device)
        print(f"  features extracted ({time.time()-t0:.1f}s)")

        # z-score from n=4000 training set (primary protocol)
        tgt4000_z, tgt_val_z4 = _mean_pool_zscore(tgt4000, tgt_val)
        ctx4000_z, ctx_val_z4 = _mean_pool_zscore(ctx4000, ctx_val)
        tgt200_z,  tgt_val_z2 = _mean_pool_zscore(tgt200,  tgt_val)
        ctx200_z,  ctx_val_z2 = _mean_pool_zscore(ctx200,  ctx_val)

        # probes
        print(f"  probing n=4000 (target) ...", end="  ", flush=True)
        t_acc4000, t_lr4000 = _best_probe(tgt4000_z, lbl4000, tgt_val_z4, lbl_val,
                                           LR_LIST, PROBE_EP, device)
        print(f"val={t_acc4000:.4f} lr={t_lr4000:.0e}")

        print(f"  probing n=4000 (context)...", end="  ", flush=True)
        c_acc4000, c_lr4000 = _best_probe(ctx4000_z, lbl4000, ctx_val_z4, lbl_val,
                                           LR_LIST, PROBE_EP, device)
        print(f"val={c_acc4000:.4f} lr={c_lr4000:.0e}")

        print(f"  probing n=200  (target) ...", end="  ", flush=True)
        t_acc200, t_lr200 = _best_probe(tgt200_z, lbl200, tgt_val_z2, lbl_val,
                                         LR_LIST, PROBE_EP, device)
        print(f"val={t_acc200:.4f} lr={t_lr200:.0e}")

        print(f"  probing n=200  (context)...", end="  ", flush=True)
        c_acc200, c_lr200 = _best_probe(ctx200_z, lbl200, ctx_val_z2, lbl_val,
                                         LR_LIST, PROBE_EP, device)
        print(f"val={c_acc200:.4f} lr={c_lr200:.0e}")

        rows.append({
            "ep":         ep,
            "t4000":      t_acc4000,
            "c4000":      c_acc4000,
            "gap4000":    t_acc4000 - c_acc4000,
            "t200":       t_acc200,
            "c200":       c_acc200,
            "gap200":     t_acc200 - c_acc200,
        })

    # ── print table ───────────────────────────────────────────────────────
    _print_table(rows)

    # ── plot ─────────────────────────────────────────────────────────────
    fig_path = REPORTS / "figures" / "probe_vs_epoch.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _plot(rows, fig_path)
    print(f"\n[pvep] figure → {fig_path}")

    # ── write report ──────────────────────────────────────────────────────
    md_path = REPORTS / "probe_vs_epoch.md"
    _write_md(rows, md_path, args)
    print(f"[pvep] report → {md_path}")


def _print_table(rows: list[dict]) -> None:
    print()
    print("=" * 78)
    print("  PROBE vs EPOCH  |  seed-0  |  locked protocol (target+zscore, lr-swept, 200ep)")
    print("=" * 78)
    print(f"  {'ep':>4}  {'tgt n=4000':>10}  {'ctx n=4000':>10}  {'gap4k':>6}"
          f"  {'tgt n=200':>9}  {'ctx n=200':>9}  {'gap200':>6}")
    print("-" * 78)
    for r in rows:
        print(f"  {r['ep']:>4}  {r['t4000']:>10.4f}  {r['c4000']:>10.4f}  "
              f"{r['gap4000']:>+6.4f}  {r['t200']:>9.4f}  {r['c200']:>9.4f}  "
              f"{r['gap200']:>+6.4f}")
    print("=" * 78)
    # reading
    if rows:
        ep150 = next((r for r in rows if r["ep"] == 150), rows[-1])
        prev  = rows[-2] if len(rows) >= 2 else None
        delta = (ep150["t4000"] - prev["t4000"]) if prev else 0.0
        print(f"\n  Δ (n=4000 target) ep{prev['ep'] if prev else '?'}→150: {delta:+.4f}")
        if   delta > 0.015:
            verdict = "RISING  → H1 (too short); next: resume seed-0 to 300 epochs"
        elif delta < 0.005:
            verdict = "PLATEAU → H2 (masking too easy); next: harder masking run"
        else:
            verdict = "AMBIGUOUS → queue both runs, duration first"
        print(f"  Pre-registered reading: {verdict}")
    print("=" * 78)


def _plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps    = [r["ep"]    for r in rows]
    t4000  = [r["t4000"] for r in rows]
    c4000  = [r["c4000"] for r in rows]
    t200   = [r["t200"]  for r in rows]
    c200   = [r["c200"]  for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.suptitle("Probe accuracy vs pretraining epoch — seed 0, val only", fontsize=12)

    for ax, t_vals, c_vals, n_label in [
        (axes[0], t4000, c4000, "n=4000"),
        (axes[1], t200,  c200,  "n=200"),
    ]:
        ax.plot(eps, t_vals, "o-", color="#2166ac", lw=2, ms=6, label="target encoder")
        ax.plot(eps, c_vals, "s--", color="#d6604d", lw=1.5, ms=5, label="context encoder")
        ax.axhline(0.70, color="#888", lw=1, ls=":", label="Gate 1B floor (0.70)")
        ax.set_xlabel("Pretraining epoch")
        ax.set_ylabel("Val accuracy")
        ax.set_title(f"Linear probe ({n_label}, locked protocol)")
        ax.set_xticks(eps)
        ax.legend(fontsize=9)
        ax.set_ylim(0.0, 0.85)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_md(rows: list[dict], out_path: Path, args) -> None:
    # pre-registered reading
    reading = "insufficient data"
    verdict_detail = ""
    if rows:
        ep150 = next((r for r in rows if r["ep"] == 150), rows[-1])
        prev  = rows[-2] if len(rows) >= 2 else None
        delta = (ep150["t4000"] - prev["t4000"]) if prev else 0.0
        if delta > 0.015:
            reading = "H1 — curve still rising at epoch 150"
            verdict_detail = (
                f"Δ(ep{prev['ep'] if prev else '?'}→150) = {delta:+.4f} > 0.015 threshold.  "
                "Next action: resume seed-0 from `runs/tkqjawa0/epoch_0150.ckpt` to 300 epochs."
            )
        elif delta < 0.005:
            reading = "H2 — curve plateaued by ~100-120 epochs"
            verdict_detail = (
                f"Δ(ep{prev['ep'] if prev else '?'}→150) = {delta:+.4f} < 0.005 threshold.  "
                "Next action: one from-scratch run with harder masking "
                "(target scale 0.20-0.25, context 0.75-0.90)."
            )
        else:
            reading = "AMBIGUOUS — borderline rise"
            verdict_detail = (
                f"Δ(ep{prev['ep'] if prev else '?'}→150) = {delta:+.4f}, between thresholds.  "
                "Next action: queue both runs, duration first (reuses existing checkpoint)."
            )

    lines = [
        "# Probe accuracy vs pretraining epoch — seed 0, val only",
        "",
        "Locked probe protocol: target encoder, mean pool, z-score, "
        "lr swept {3e-3,1e-3,3e-4}, 200 epochs.",
        "",
        "| Epoch | tgt n=4000 | ctx n=4000 | gap (T−C) | tgt n=200 | ctx n=200 | gap (T−C) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['ep']} | {r['t4000']:.4f} | {r['c4000']:.4f} | "
            f"{r['gap4000']:+.4f} | {r['t200']:.4f} | {r['c200']:.4f} | "
            f"{r['gap200']:+.4f} |"
        )
    lines += [
        "",
        f"**Pre-registered reading: {reading}**",
        "",
        verdict_detail,
        "",
        "---",
        "",
        "Gate 1B floor (≥0.70): shown as dashed line in figure.",
        "Figure: `reports/figures/probe_vs_epoch.png`",
        "",
        f"*Generated by `scripts/probe_vs_epoch.py`, seed={args.seed}*",
    ]
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
