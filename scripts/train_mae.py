"""
train_mae.py -- training entry point for the pixel-MAE baseline.

Usage:
    uv run python scripts/train_mae.py --config configs/mae_baseline.yaml --seed 0
    uv run python scripts/train_mae.py --config configs/mae_baseline.yaml --seed 0 \\
        --resume runs/<run-id>/best.ckpt

Sibling to scripts/train.py; reuses checkpoint.py and the same data loaders.
Logs: train/loss, train/lr to W&B.  No collapse diagnostics (no EMA encoder).
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint, save_checkpoint
from src.data.stl10 import get_pretrain_loader, get_smoke_loader
from src.models.mae import PixelMAE, PixelMAEConfig


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_mae_cfg(cfg: dict) -> PixelMAEConfig:
    m = cfg["model"]
    return PixelMAEConfig(
        img_size=m["img_size"],
        patch_size=m["patch_size"],
        d_model=m["d_model"],
        enc_layers=m["enc_layers"],
        enc_heads=m["enc_heads"],
        dec_width=m.get("dec_width", 128),
        dec_layers=m.get("dec_layers", 2),
        dec_heads=m.get("dec_heads", 2),
        mask_ratio=m.get("mask_ratio", 0.75),
        dropout=m.get("dropout", 0.0),
        norm_pix_loss=m.get("norm_pix_loss", True),
    )


def _build_loader(cfg: dict, seed: int | None):
    d = cfg["data"]
    data_dir = Path(__file__).parent.parent / "data"
    n_images  = d.get("n_images")
    batch_size = d.get("batch_size", 256)
    num_workers = d.get("num_workers", 4)
    loader_seed = seed if seed is not None else d.get("seed")
    m = cfg["model"]
    n_h = m["img_size"] // m["patch_size"]

    if n_images is not None:
        return get_smoke_loader(data_dir, n_images=n_images, batch_size=batch_size,
                                n_h=n_h, n_w=n_h, seed=loader_seed)
    return get_pretrain_loader(data_dir, batch_size=batch_size, n_h=n_h, n_w=n_h,
                               num_workers=num_workers, seed=loader_seed, drop_last=True)


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_amp_ctx(device: str, use_amp: bool):
    if not use_amp:
        return contextlib.nullcontext()
    amp_device = "mps" if device == "mps" else ("cuda" if device.startswith("cuda") else "cpu")
    try:
        ctx = torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=True)
        with ctx:
            torch.zeros(1, device=device)
        return ctx
    except Exception:
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_mae(model, loader, cfg, device, run, run_dir,
              start_epoch=0, start_step=0,
              optimizer_state=None, scheduler_state=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).train()

    t = cfg["train"]
    n_batches   = len(loader)
    total_steps = t["epochs"] * n_batches
    warmup_steps = t["warmup_epochs"] * n_batches

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=t["lr"], weight_decay=t["weight_decay"])
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    def _cosine_warmup(s):
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine_warmup)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    amp_ctx = _make_amp_ctx(device, t.get("use_amp", True))
    ckpt_every = t.get("ckpt_every", 10)
    grad_clip  = t.get("grad_clip", 1.0)

    step = start_step
    best_loss = float("inf")
    best_path = run_dir / "best.ckpt"
    prev_periodic: Path | None = None
    wandb_run_id = run.id if run is not None else None

    for epoch in range(start_epoch, t["epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch["imgs"]
            imgs = imgs.to(device)

            with amp_ctx:
                out = model(imgs, seed=step)     # vary mask per step
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, n_batches)
        log_row = {
            "epoch":      epoch,
            "train/loss": avg_loss,
            "train/lr":   scheduler.get_last_lr()[0],
        }
        if run is not None:
            run.log(log_row, step=epoch)

        print(f"  ep {epoch:4d}  loss={avg_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(best_path, model, optimizer, scheduler,
                            epoch=epoch, step=step,
                            extra={"loss": best_loss, "wandb_run_id": wandb_run_id})

        if (epoch + 1) % ckpt_every == 0:
            new_periodic = run_dir / f"epoch_{epoch+1:04d}.ckpt"
            save_checkpoint(new_periodic, model, optimizer, scheduler,
                            epoch=epoch, step=step,
                            extra={"loss": avg_loss, "wandb_run_id": wandb_run_id})
            if run is not None:
                import wandb
                artifact = wandb.Artifact(
                    name=f"mae-ckpt-{run.id}-ep{epoch+1}", type="model",
                    metadata={"epoch": epoch + 1, "step": step, "loss": avg_loss},
                )
                artifact.add_file(str(new_periodic))
                run.log_artifact(artifact)
            if prev_periodic is not None and prev_periodic.exists():
                prev_periodic.unlink()
            prev_periodic = new_periodic

    last_path = prev_periodic or best_path
    return best_path, last_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--resume",   default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg    = _load_yaml(args.config)
    seed   = args.seed
    device = _pick_device()
    print(f"[train_mae] device={device}  config={args.config}  seed={seed}")

    if seed is not None:
        torch.manual_seed(seed)

    mae_cfg = _build_mae_cfg(cfg)
    model = PixelMAE(mae_cfg)
    trainable = sum(p.numel() for p in model.parameters())
    print(f"[train_mae] params {trainable:,}")

    loader = _build_loader(cfg, seed)
    print(f"[train_mae] {len(loader)} batches/epoch")

    start_epoch = 0
    start_step  = 0
    opt_state   = None
    sched_state = None
    resume_wandb_id = None

    if args.resume:
        ckpt = load_checkpoint(args.resume, model=model)
        start_epoch = ckpt["epoch"] + 1
        start_step  = ckpt["step"]
        opt_state   = ckpt["optimizer_state"]
        sched_state = ckpt["scheduler_state"]
        resume_wandb_id = ckpt["extra"].get("wandb_run_id")
        print(f"[train_mae] resuming epoch {start_epoch}, step {start_step}")

    run = None
    if not args.no_wandb:
        try:
            import wandb
            wc = cfg.get("wandb", {})
            tags = list(wc.get("tags", []))
            if seed is not None:
                tags.append(f"seed-{seed}")
            run = wandb.init(
                id=resume_wandb_id,
                project=wc.get("project", "jepa-vision"),
                entity=wc.get("entity", "entropy_chess"),
                config=cfg,
                tags=tags,
                resume="allow" if resume_wandb_id else None,
            )
            print(f"[train_mae] W&B run: {run.url}")
        except ImportError:
            print("[train_mae] wandb not installed")

    if run is not None:
        run_dir = Path("runs") / run.id
    else:
        run_dir = Path("runs") / f"mae-dry-{int(time.time())}"

    best_path, last_path = train_mae(
        model=model, loader=loader, cfg=cfg, device=device, run=run,
        run_dir=run_dir, start_epoch=start_epoch, start_step=start_step,
        optimizer_state=opt_state, scheduler_state=sched_state,
    )
    print(f"[train_mae] done. best={best_path}  last={last_path}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
