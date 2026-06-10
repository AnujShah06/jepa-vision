"""
smoke_test.py -- Gate 0 end-to-end plumbing test.

Tiny config (d=64, 2 layers), 100 STL-10 unlabeled images, 2 epochs.
Logs loss components + collapse diagnostics to W&B project "jepa-vision".
The goal is correct plumbing, not learning -- loss decreasing is NOT required.

Usage:
    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import torch
import wandb

# make src importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.stl10 import get_smoke_loader
from src.diagnostics import collapse_report
from src.loop import _ema_momentum
from src.models.jepa import VisionJEPA, VisionJEPAConfig


def main() -> None:
    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[smoke_test] device: {device}")

    # ------------------------------------------------------------------
    # Tiny config -- all overrides from the production defaults
    # ------------------------------------------------------------------
    cfg = VisionJEPAConfig(
        img_size=96,
        patch_size=8,
        d_model=64,
        enc_layers=2,
        enc_heads=4,       # 64/4 = 16 per head
        pred_layers=2,
        pred_heads=4,
        sigreg_weight=0.1,
        sigreg_projections=16,
        ema_decay=0.996,
        use_ema=True,
        dropout=0.0,
    )
    n_patches = (cfg.img_size // cfg.patch_size) ** 2  # 144

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    run = wandb.init(
        project="jepa-vision",
        name="gate0-smoke",
        config=dataclasses.asdict(cfg),
        tags=["smoke", "gate0"],
    )
    assert run is not None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_dir = Path(__file__).parent.parent / "data"
    print("[smoke_test] loading STL-10 unlabeled (downloads ~2.6 GB on first run)…")
    loader = get_smoke_loader(
        data_dir,
        n_images=100,
        batch_size=16,
        n_h=cfg.img_size // cfg.patch_size,
        n_w=cfg.img_size // cfg.patch_size,
        seed=0,
    )
    print(f"[smoke_test] {len(loader)} batches per epoch")

    # ------------------------------------------------------------------
    # Model + optimiser
    # ------------------------------------------------------------------
    model = VisionJEPA(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke_test] trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=0.05,
    )

    n_epochs = 2
    ema_start, ema_end = 0.996, 1.0
    total_steps = n_epochs * len(loader)
    step = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pred = 0.0
        epoch_sigreg = 0.0
        ctx_emb_chunks: list[torch.Tensor] = []

        for batch in loader:
            # move tensors to device
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            out = model(batch)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            decay = _ema_momentum(step, total_steps, ema_start, ema_end)
            model.ema_update(decay=decay)

            epoch_loss += float(loss.item())
            epoch_pred += float(out["pred_loss"].item())
            epoch_sigreg += float(out["sigreg_term"].item())

            # collect context embeddings on CPU for collapse_report
            ctx_emb_chunks.append(
                out["context_emb"].detach().cpu().reshape(-1, cfg.d_model)
            )
            step += 1

        n = len(loader)
        ctx_all = torch.cat(ctx_emb_chunks, dim=0)  # always on CPU
        diag = collapse_report(ctx_all)

        log_row = {
            "epoch": epoch,
            "loss": epoch_loss / n,
            "pred_loss": epoch_pred / n,
            "sigreg_term": epoch_sigreg / n,
            **diag,
        }
        wandb.log(log_row, step=epoch)

        print(
            f"  epoch {epoch} | loss={epoch_loss/n:.4f}  pred={epoch_pred/n:.4f}"
            f"  sigreg={epoch_sigreg/n:.4f}  eff_rank={diag['effective_rank']:.2f}"
            f"  spread={diag['embedding_spread']:.4f}"
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n[smoke_test] W&B run: {run.url}")
    wandb.finish()
    print("[smoke_test] Gate 0 PASSED -- plumbing is green.")


if __name__ == "__main__":
    main()
