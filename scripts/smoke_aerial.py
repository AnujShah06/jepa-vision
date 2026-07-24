"""
smoke_aerial.py -- Phase-2 Gate-0 smoke on RESISC45.

Pre-registered pass bar (DECISIONS.md [2.1]):
  eff_rank >= 30/64, loss monotone-ish over 2 epochs, no NaN.

Usage:
    uv run python scripts/smoke_aerial.py
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.resisc45 import get_resisc45_pretrain_loader
from src.diagnostics import collapse_report
from src.models.jepa import VisionJEPA, VisionJEPAConfig
from src.models.loss import jepa_loss

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 2
BATCH  = 64     # smaller than production for a fast smoke


def main() -> None:
    print(f"[smoke] device={DEVICE}  epochs={EPOCHS}  batch={BATCH}")

    # -- tiny model (d=64, 2 enc layers — matches Phase-1 Gate-0 convention) ---
    cfg = VisionJEPAConfig(
        img_size=96, patch_size=8,
        d_model=64,  enc_layers=2, enc_heads=4,
        pred_layers=2, pred_width=32, pred_heads=4,
    )
    model = VisionJEPA(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] trainable params: {n_params:,}")

    # -- RESISC45 train loader at 96×96 --
    t0 = time.time()
    loader = get_resisc45_pretrain_loader(
        batch_size=BATCH,
        num_workers=0,          # single-process for smoke
        seed=0,
        drop_last=True,
    )
    print(f"[smoke] dataloader: {len(loader)} batches/epoch  "
          f"({BATCH * len(loader):,} images/epoch)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.04,
    )

    epoch_losses = []
    for epoch in range(EPOCHS):
        t_ep = time.time()
        total_loss = 0.0
        n_batches = 0
        emb_chunks = []

        model.train()
        for batch in loader:
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            out = model(batch)
            loss = out["loss"]
            assert not torch.isnan(loss), f"NaN loss at epoch {epoch}, batch {n_batches}"

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            if len(emb_chunks) < 8:
                emb_chunks.append(
                    out["context_emb"].detach().cpu().reshape(-1, cfg.d_model)
                )

        avg_loss = total_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)

        # collapse diagnostics
        diag = collapse_report(torch.cat(emb_chunks))
        eff_rank = diag["effective_rank"]
        t_ep_elapsed = time.time() - t_ep

        print(f"  ep {epoch}  loss={avg_loss:.4f}  eff_rank={eff_rank:.1f}/{cfg.d_model}"
              f"  t={t_ep_elapsed:.1f}s")

    total_elapsed = time.time() - t0

    # -- Gate-0 verdict -------------------------------------------------------
    loss_monotone = epoch_losses[-1] <= epoch_losses[0]  # ep1 ≤ ep0
    rank_ok = eff_rank >= 30      # pre-registered bar: 30/64

    print("\n[smoke] Gate-0 summary:")
    print(f"  loss ep0={epoch_losses[0]:.4f}  ep1={epoch_losses[-1]:.4f}  "
          f"monotone={'YES' if loss_monotone else 'NO (flag)'}")
    print(f"  final eff_rank={eff_rank:.1f}/64  bar=30  {'PASS' if rank_ok else 'FAIL'}")
    print(f"  wall time: {total_elapsed:.1f}s  "
          f"throughput: {BATCH * len(loader) * EPOCHS / total_elapsed:.0f} img/s")
    print(f"  projected 150-epoch production s/epoch: "
          f"~{total_elapsed/EPOCHS * (256/BATCH) * (22400/(BATCH*len(loader))):.0f}s")
    print(f"  NaN check: PASS (no NaN encountered)")

    gate = rank_ok and (not torch.isnan(torch.tensor(epoch_losses[-1])))
    print(f"\n[smoke] Gate-0: {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("  STOP — do not launch encoder A until gate passes.")
        sys.exit(1)
    print("  Encoder A is cleared for launch.")


if __name__ == "__main__":
    main()
