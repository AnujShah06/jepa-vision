"""
probe_diag.py — Step 1.5d probe-path diagnostics on seed-0 val, n=4000 only.

Tests four axes against the baseline (context encoder, mean pool, lr=1e-3, 100ep):
  (1) target_encoder vs context_encoder features (EMA encoder often stronger)
  (2) z-score feature standardisation before the linear head
  (3) pooling variants: mean+max concat, last-2-layer concat
  (4) lr sweep {3e-3, 1e-3, 3e-4} × 200 epochs on the best encoder

Runs entirely on precomputed features — no gradient through the encoder.
Validation only; test set quarantined.

Usage:
    uv run python scripts/probe_diag.py --jepa_ckpt runs/tkqjawa0/best.ckpt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint
from src.data.stl10 import get_val_loader
from src.eval.probe import get_probe_pool, stratified_sample, train_probe
from src.models.jepa import VisionJEPA, VisionJEPAConfig

DATA_DIR = Path(__file__).parent.parent / "data"


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Feature extraction with multiple pooling variants in one pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_variants(
    model: VisionJEPA,
    encoder_attr: str,      # "context_encoder" or "target_encoder"
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One forward pass; returns three feature tensors and labels.

    Returns:
        final_tokens  : [N, n_patches, d_model] — encoder final output
        layer5_tokens : [N, n_patches, d_model] — second-to-last block output
                        (before final LayerNorm; used for last-2-layer concat)
        labels        : [N] long
    """
    model.eval()
    encoder = getattr(model, encoder_attr)

    # Hook on the second-to-last transformer block
    layer5_cache: list[torch.Tensor] = []
    def _hook(m: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        layer5_cache.append(out.detach().cpu())
    handle = encoder.blocks[-2].register_forward_hook(_hook)

    all_final:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        imgs   = batch[0].to(device)
        labels = batch[1]

        tokens = model.patch_embed(imgs) + model.pos_embed   # [B, N, d]
        final  = encoder(tokens)                              # [B, N, d]
        all_final.append(final.detach().cpu())
        all_labels.append(labels)

    handle.remove()

    final_t  = torch.cat(all_final)       # [N_total, N_patches, d]
    layer5_t = torch.cat(layer5_cache)    # [N_total, N_patches, d]
    labels_t = torch.cat(all_labels).long()
    return final_t, layer5_t, labels_t


def _pool(final: torch.Tensor, layer5: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool [N, patches, d] → [N, feat_dim] according to mode."""
    if mode == "mean":
        return final.mean(1)
    if mode == "mean_max":
        return torch.cat([final.mean(1), final.max(1).values], dim=1)
    if mode == "l2concat":
        # layer5 tokens (pre-final-norm) + final tokens, each mean-pooled
        return torch.cat([layer5.mean(1), final.mean(1)], dim=1)
    raise ValueError(f"Unknown pooling mode: {mode}")


def _zscore(
    train_f: torch.Tensor, val_f: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-score normalise val features with statistics from train features."""
    mu  = train_f.mean(0)
    sig = train_f.std(0).clamp(min=1e-6)
    return (train_f - mu) / sig, (val_f - mu) / sig


def _best_probe(
    train_f: torch.Tensor,
    train_l: torch.Tensor,
    val_f:   torch.Tensor,
    val_l:   torch.Tensor,
    lr_list: tuple[float, ...],
    epochs:  int,
    device:  str,
) -> tuple[float, float]:
    """Train probe for each lr, return (best_val_acc, best_lr)."""
    best_acc = -1.0
    best_lr  = lr_list[0]
    for lr in lr_list:
        _, acc = train_probe(train_f, train_l, val_f, val_l,
                             epochs=epochs, lr=lr, device=device)
        if acc > best_acc:
            best_acc, best_lr = acc, lr
    return best_acc, best_lr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_ckpt", default="runs/tkqjawa0/best.ckpt")
    parser.add_argument("--n",         type=int, default=4000)
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    device = _pick_device()
    n      = args.n
    assert n % 10 == 0
    n_per_class = n // 10

    print(f"[diag] device={device}  ckpt={args.jepa_ckpt}  n={n}")

    # ── load model ────────────────────────────────────────────────────────
    jepa = VisionJEPA(VisionJEPAConfig()).eval()
    load_checkpoint(args.jepa_ckpt, model=jepa, map_location=device)
    jepa.to(device)

    # ── probe pool ────────────────────────────────────────────────────────
    probe_indices, probe_labels = get_probe_pool(DATA_DIR)
    subset_idx   = stratified_sample(probe_labels, n_per_class=n_per_class, seed=args.seed)
    train_ds_idx = [probe_indices[i] for i in subset_idx]

    import torchvision.transforms as T
    from torchvision.datasets import STL10
    _tfm = T.Compose([
        T.Resize(96, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(96),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _ds = STL10(root=str(DATA_DIR), split="train", transform=_tfm, download=False)
    train_loader = DataLoader(Subset(_ds, train_ds_idx),
                              batch_size=256, shuffle=False, num_workers=0)
    val_loader   = get_val_loader(DATA_DIR, batch_size=256, num_workers=0)

    # ── extract tokens: context encoder ───────────────────────────────────
    print("[diag] extracting context encoder tokens ...", end="  ", flush=True)
    t0 = time.time()
    ctx_final, ctx_l5, train_labels = extract_variants(jepa, "context_encoder",
                                                        train_loader, device)
    ctx_val_final, ctx_val_l5, val_labels = extract_variants(jepa, "context_encoder",
                                                              val_loader, device)
    print(f"{ctx_final.shape}  ({time.time()-t0:.1f}s)")

    # ── extract tokens: target encoder ────────────────────────────────────
    print("[diag] extracting target encoder tokens ...", end="  ", flush=True)
    t0 = time.time()
    tgt_final, tgt_l5, _ = extract_variants(jepa, "target_encoder",
                                             train_loader, device)
    tgt_val_final, tgt_val_l5, _ = extract_variants(jepa, "target_encoder",
                                                     val_loader, device)
    print(f"{tgt_final.shape}  ({time.time()-t0:.1f}s)")

    # ── run variants ──────────────────────────────────────────────────────
    LR_FIXED    = (1e-3,)
    LR_SWEEP    = (3e-3, 1e-3, 3e-4)
    EP_SHORT    = 100
    EP_LONG     = 200

    results: list[dict] = []

    def _run(label: str, train_f: torch.Tensor, val_f: torch.Tensor,
             lr_list: tuple, epochs: int) -> None:
        t0 = time.time()
        print(f"  {label:<42}", end="  ", flush=True)
        acc, lr = _best_probe(train_f, train_labels, val_f, val_labels,
                              lr_list=lr_list, epochs=epochs, device=device)
        dt = time.time() - t0
        print(f"val_acc={acc:.4f}  lr={lr:.0e}  ({dt:.0f}s)")
        results.append({"label": label, "acc": acc, "epochs": epochs,
                        "best_lr": lr, "n_lrs": len(lr_list)})

    print("\n[diag] running probe variants ...")

    # 1. Baseline (reproduce grid result)
    _run("context mean (baseline)",
         _pool(ctx_final, ctx_l5, "mean"),
         _pool(ctx_val_final, ctx_val_l5, "mean"),
         LR_FIXED, EP_SHORT)

    # 2. Target encoder, mean pool
    _run("target mean",
         _pool(tgt_final, tgt_l5, "mean"),
         _pool(tgt_val_final, tgt_val_l5, "mean"),
         LR_FIXED, EP_SHORT)

    # 3. Context + z-score
    ctx_mean_tr = _pool(ctx_final, ctx_l5, "mean")
    ctx_mean_vl = _pool(ctx_val_final, ctx_val_l5, "mean")
    ctx_z_tr, ctx_z_vl = _zscore(ctx_mean_tr, ctx_mean_vl)
    _run("context mean + z-score", ctx_z_tr, ctx_z_vl, LR_FIXED, EP_SHORT)

    # 4. Target + z-score
    tgt_mean_tr = _pool(tgt_final, tgt_l5, "mean")
    tgt_mean_vl = _pool(tgt_val_final, tgt_val_l5, "mean")
    tgt_z_tr, tgt_z_vl = _zscore(tgt_mean_tr, tgt_mean_vl)
    _run("target mean + z-score", tgt_z_tr, tgt_z_vl, LR_FIXED, EP_SHORT)

    # 5. Context mean+max concat
    _run("context mean+max concat",
         _pool(ctx_final, ctx_l5, "mean_max"),
         _pool(ctx_val_final, ctx_val_l5, "mean_max"),
         LR_FIXED, EP_SHORT)

    # 6. Context last-2-layer concat
    _run("context last-2-layer concat",
         _pool(ctx_final, ctx_l5, "l2concat"),
         _pool(ctx_val_final, ctx_val_l5, "l2concat"),
         LR_FIXED, EP_SHORT)

    # 7. Target mean, lr sweep, 200 ep
    _run("target mean  |  lr-sweep  |  200ep",
         tgt_mean_tr, tgt_mean_vl, LR_SWEEP, EP_LONG)

    # 8. Target mean + z-score, lr sweep, 200 ep  (best-combo candidate)
    _run("target mean+zscore  |  lr-sweep  |  200ep",
         tgt_z_tr, tgt_z_vl, LR_SWEEP, EP_LONG)

    # ── print table ───────────────────────────────────────────────────────
    W = 44
    print()
    print("=" * 70)
    print(f"  PROBE DIAGNOSTICS  |  n={n}  |  seed-{args.seed}  |  val only")
    print("=" * 70)
    print(f"  {'Variant':<{W}}  {'Val Acc':>7}  {'Ep':>4}  {'Best LR':>8}")
    print("-" * 70)
    for r in results:
        print(f"  {r['label']:<{W}}  {r['acc']:>7.4f}  "
              f"{r['epochs']:>4}  {r['best_lr']:>8.0e}")
    print("=" * 70)
    best = max(results, key=lambda r: r["acc"])
    baseline = results[0]["acc"]
    print(f"\n  Best: '{best['label']}'  {best['acc']:.4f}  "
          f"(+{best['acc']-baseline:+.4f} vs baseline)")
    gate_str = "PASS" if best["acc"] >= 0.70 else f"FAIL — gap {best['acc']-0.70:+.4f}"
    print(f"  Gate 1B floor (>=0.70): {gate_str}")


if __name__ == "__main__":
    main()
