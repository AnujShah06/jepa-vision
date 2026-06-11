"""
loop.py -- production JEPA training loop.

AdamW + cosine-with-warmup LR + EMA target-encoder update + AMP (bfloat16
on MPS/CUDA) + periodic checkpointing + W&B artifact upload.

collapse diagnostics are logged every epoch from training-batch embeddings;
there is no separate pretraining validation set, so "val" is unused for the
reference run.  The val_loader parameter is retained for future probe evals.

Resume protocol
---------------
train() accepts start_epoch / start_step / optimizer_state / scheduler_state
so train.py can pass restored state after loading a checkpoint.
The scheduler is always created with full-run total_steps; restoring its
state_dict advances last_epoch to the correct position so the LR continues
smoothly from where it left off.
"""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.checkpoint import save_checkpoint
from src.diagnostics import collapse_report
from src.models.jepa import VisionJEPA

if TYPE_CHECKING:
    import wandb as _wandb


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainConfig:
    """Training-loop hyperparameters."""
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 0.04
    warmup_epochs: int = 10       # LR warms up linearly for this many epochs
    grad_clip: float = 1.0
    ckpt_every: int = 10          # save periodic checkpoint every N epochs
    diag_every: int = 1           # compute collapse diagnostics every N epochs
    ema_start: float = 0.996
    ema_end: float = 1.0
    use_amp: bool = True          # bfloat16 autocast (MPS) or float16 (CUDA)


# ---------------------------------------------------------------------------
# LR / EMA schedule helpers
# ---------------------------------------------------------------------------

def _ema_momentum(step: int, total: int, start: float, end: float) -> float:
    """Linear ramp from start to end over total steps."""
    if total <= 1:
        return end
    return start + (end - start) * min(1.0, step / (total - 1))


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to zero."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    import math
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _make_amp_ctx(device: str, use_amp: bool):
    """Return an autocast context (or nullcontext when AMP is off/unavailable)."""
    if not use_amp:
        return contextlib.nullcontext()
    amp_device = "mps" if device == "mps" else ("cuda" if device.startswith("cuda") else "cpu")
    amp_dtype = torch.bfloat16  # bfloat16 works on both MPS and modern CUDA
    try:
        ctx = torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=True)
        # quick validation — will raise if backend doesn't support it
        with ctx:
            torch.zeros(1, device=device)
        return ctx
    except Exception:
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Evaluation (used when val_loader is provided)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: VisionJEPA,
    loader: DataLoader,
    device: str,
    amp_ctx=None,
) -> dict:
    """
    Run the JEPA objective over `loader` in eval mode.

    Returns mean loss components plus collapse diagnostics from up to 16
    batches of context embeddings.
    """
    if amp_ctx is None:
        amp_ctx = contextlib.nullcontext()

    model.eval()
    sums: dict[str, float] = {"loss": 0.0, "pred_loss": 0.0, "sigreg_term": 0.0}
    n_batches = 0
    emb_chunks: list[torch.Tensor] = []

    for batch in loader:
        batch = _to_device(batch, device)
        with amp_ctx:
            out = model(batch)
        for k in sums:
            if k in out:
                sums[k] += float(out[k].item())
        n_batches += 1
        if len(emb_chunks) < 16:
            emb_chunks.append(
                out["context_emb"].detach().cpu().reshape(-1, model.cfg.d_model)
            )

    metrics = {k: v / max(1, n_batches) for k, v in sums.items()}
    if emb_chunks:
        metrics.update(collapse_report(torch.cat(emb_chunks)))
    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: VisionJEPA,
    train_loader: DataLoader,
    cfg: TrainConfig,
    device: str,
    run,                                  # wandb.Run (or None for dry-runs)
    run_dir: Path,
    start_epoch: int = 0,
    start_step: int = 0,
    optimizer_state: dict | None = None,
    scheduler_state: dict | None = None,
    val_loader: DataLoader | None = None,
) -> tuple[Path, Path]:
    """
    Train VisionJEPA.  Returns (best_ckpt_path, last_periodic_ckpt_path).

    Checkpoints are saved to run_dir/ every cfg.ckpt_every epochs and
    uploaded as W&B artifacts (if run is not None).  Collapse diagnostics
    are logged every cfg.diag_every epochs from training-batch embeddings.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    amp_ctx = _make_amp_ctx(device, cfg.use_amp)
    n_batches = len(train_loader)
    total_steps = cfg.epochs * n_batches
    warmup_steps = cfg.warmup_epochs * n_batches

    # -- optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # -- LR scheduler (always full-run total_steps for correct resume) -----
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_warmup(s, warmup_steps, total_steps),
    )
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    # -- bookkeeping -------------------------------------------------------
    step = start_step
    best_pred_loss = float("inf")
    best_path = run_dir / "best.ckpt"
    prev_periodic: Path | None = None

    wandb_run_id = run.id if run is not None else None

    # -- epoch loop --------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        epoch_sums: dict[str, float] = {"loss": 0.0, "pred_loss": 0.0, "sigreg_term": 0.0}
        emb_chunks: list[torch.Tensor] = []
        model.train()

        for batch in train_loader:
            batch = _to_device(batch, device)

            with amp_ctx:
                out = model(batch)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            ema_m = _ema_momentum(step, total_steps, cfg.ema_start, cfg.ema_end)
            model.ema_update(decay=ema_m)

            for k in epoch_sums:
                if k in out:
                    epoch_sums[k] += float(out[k].item())
            step += 1

            # collect embeddings for diagnostics (cap at 8 batches to save memory)
            if len(emb_chunks) < 8 and (epoch % cfg.diag_every == 0):
                emb_chunks.append(
                    out["context_emb"].detach().cpu().reshape(-1, model.cfg.d_model)
                )

        # -- epoch metrics ------------------------------------------------
        avgs = {k: v / max(1, n_batches) for k, v in epoch_sums.items()}
        log_row: dict = {
            "epoch": epoch,
            "train/loss": avgs["loss"],
            "train/pred_loss": avgs["pred_loss"],
            "train/sigreg_term": avgs["sigreg_term"],
            "train/lr": scheduler.get_last_lr()[0],
            "train/ema_momentum": _ema_momentum(step, total_steps,
                                                cfg.ema_start, cfg.ema_end),
        }

        if emb_chunks and epoch % cfg.diag_every == 0:
            diag = collapse_report(torch.cat(emb_chunks))
            log_row.update({f"diag/{k}": v for k, v in diag.items()})

        # optional validation
        if val_loader is not None:
            val = evaluate(model, val_loader, device, amp_ctx)
            log_row.update({f"val/{k}": v for k, v in val.items()})

        if run is not None:
            run.log(log_row, step=epoch)

        print(
            f"  ep {epoch:4d}  loss={avgs['loss']:.4f}  "
            f"pred={avgs['pred_loss']:.4f}  "
            f"sigreg={avgs['sigreg_term']:.4f}  "
            + (f"  eff_rank={log_row.get('diag/effective_rank', 0):.1f}" if emb_chunks else ""),
            flush=True,
        )

        # -- best checkpoint (by pred_loss, no val set for pretraining) ---
        if avgs["pred_loss"] < best_pred_loss:
            best_pred_loss = avgs["pred_loss"]
            save_checkpoint(
                best_path, model, optimizer, scheduler,
                epoch=epoch, step=step,
                extra={"pred_loss": best_pred_loss, "wandb_run_id": wandb_run_id},
            )

        # -- periodic checkpoint ------------------------------------------
        if (epoch + 1) % cfg.ckpt_every == 0:
            new_periodic = run_dir / f"epoch_{epoch+1:04d}.ckpt"
            save_checkpoint(
                new_periodic, model, optimizer, scheduler,
                epoch=epoch, step=step,
                extra={"pred_loss": avgs["pred_loss"], "wandb_run_id": wandb_run_id},
            )

            # upload as W&B artifact for easy retrieval
            if run is not None:
                import wandb
                artifact = wandb.Artifact(
                    name=f"ckpt-{run.id}-ep{epoch+1}",
                    type="model",
                    metadata={"epoch": epoch + 1, "step": step,
                               "pred_loss": avgs["pred_loss"]},
                )
                artifact.add_file(str(new_periodic))
                run.log_artifact(artifact)

            # keep only the latest periodic checkpoint on disk (saves space)
            if prev_periodic is not None and prev_periodic.exists():
                prev_periodic.unlink()
            prev_periodic = new_periodic

    last_path = prev_periodic or best_path
    return best_path, last_path
