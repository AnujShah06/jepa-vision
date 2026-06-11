"""
train.py -- production entry point for jepa-vision Phase 1.

Usage:
    # fresh run
    uv run python scripts/train.py --config configs/phase1_ref.yaml --seed 0

    # resume from checkpoint
    uv run python scripts/train.py --config configs/phase1_ref.yaml --seed 0 \\
        --resume runs/jepa-vision-s0/best.ckpt

    # overfit check (uses configs/overfit.yaml)
    uv run python scripts/train.py --config configs/overfit.yaml --seed 0

The config YAML drives everything; --seed overrides data.seed so three seeds
can share the same config file.

Run directory: runs/<wandb-run-id>/  (or runs/dry-<timestamp>/ when --no-wandb)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint
from src.data.stl10 import get_pretrain_loader, get_smoke_loader
from src.loop import TrainConfig, train
from src.models.jepa import VisionJEPA, VisionJEPAConfig


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_model_cfg(cfg: dict) -> VisionJEPAConfig:
    m = cfg["model"]
    return VisionJEPAConfig(
        img_size=m["img_size"],
        patch_size=m["patch_size"],
        d_model=m["d_model"],
        enc_layers=m["enc_layers"],
        enc_heads=m["enc_heads"],
        pred_layers=m["pred_layers"],
        pred_width=m["pred_width"],
        pred_heads=m["pred_heads"],
        sigreg_weight=m.get("sigreg_weight", 1.0),
        sigreg_projections=m.get("sigreg_projections", 64),
        ema_decay=m.get("ema_decay", 0.996),
        use_ema=m.get("use_ema", True),
        dropout=m.get("dropout", 0.0),
    )


def _build_train_cfg(cfg: dict) -> TrainConfig:
    t = cfg["train"]
    return TrainConfig(
        epochs=t["epochs"],
        lr=t["lr"],
        weight_decay=t["weight_decay"],
        warmup_epochs=t["warmup_epochs"],
        grad_clip=t["grad_clip"],
        ckpt_every=t["ckpt_every"],
        diag_every=t.get("diag_every", 1),
        ema_start=t.get("ema_start", 0.996),
        ema_end=t.get("ema_end", 1.0),
        use_amp=t.get("use_amp", True),
    )


def _build_loader(cfg: dict, seed: int | None):
    d = cfg["data"]
    data_dir = Path(__file__).parent.parent / "data"
    n_images = d.get("n_images")
    augment = d.get("augment", True)
    batch_size = d.get("batch_size", 256)
    num_workers = d.get("num_workers", 4)
    loader_seed = seed if seed is not None else d.get("seed")

    m = cfg["model"]
    n_h = m["img_size"] // m["patch_size"]

    if n_images is not None:
        # small subset -- use smoke loader (no augmentation regardless of flag)
        return get_smoke_loader(
            data_dir,
            n_images=n_images,
            batch_size=batch_size,
            n_h=n_h,
            n_w=n_h,
            seed=loader_seed,
        )
    else:
        return get_pretrain_loader(
            data_dir,
            batch_size=batch_size,
            n_h=n_h,
            n_w=n_h,
            num_workers=num_workers,
            seed=loader_seed,
            drop_last=True,
        )


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train VisionJEPA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config data.seed)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging (dry run)")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    seed = args.seed

    # -- determinism -------------------------------------------------------
    if seed is not None:
        torch.manual_seed(seed)

    # -- device ------------------------------------------------------------
    device = _pick_device()
    print(f"[train] device={device}  config={args.config}  seed={seed}")

    # -- model -------------------------------------------------------------
    model_cfg = _build_model_cfg(cfg)
    model = VisionJEPA(model_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] params  trainable={trainable:,}  total={total:,}")

    # -- data --------------------------------------------------------------
    loader = _build_loader(cfg, seed)
    print(f"[train] {len(loader)} batches/epoch")

    # -- resume state ------------------------------------------------------
    start_epoch = 0
    start_step = 0
    optimizer_state = None
    scheduler_state = None
    resume_wandb_id = None

    if args.resume:
        print(f"[train] loading checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, model=model)
        start_epoch = ckpt["epoch"] + 1   # resume from next epoch
        start_step = ckpt["step"]
        optimizer_state = ckpt["optimizer_state"]
        scheduler_state = ckpt["scheduler_state"]
        resume_wandb_id = ckpt["extra"].get("wandb_run_id")
        print(f"[train] resuming from epoch {start_epoch}, step {start_step}"
              + (f", W&B run {resume_wandb_id}" if resume_wandb_id else ""))

    # -- W&B ---------------------------------------------------------------
    run = None
    if not args.no_wandb:
        try:
            import wandb
            wc = cfg.get("wandb", {})
            tags = list(wc.get("tags", []))
            if seed is not None:
                tags.append(f"seed-{seed}")
            run = wandb.init(
                id=resume_wandb_id,          # None → fresh run; str → stitch
                project=wc.get("project", "jepa-vision"),
                entity=wc.get("entity", "entropy_chess"),
                config={
                    "model": cfg["model"],
                    "train": cfg["train"],
                    "data": cfg["data"],
                    "seed": seed,
                    "device": device,
                },
                tags=tags,
                resume="allow" if resume_wandb_id else None,
            )
            print(f"[train] W&B run: {run.url}")
        except ImportError:
            print("[train] wandb not installed, running without logging")

    # -- run dir -----------------------------------------------------------
    if run is not None:
        run_dir = Path("runs") / run.id
    else:
        run_dir = Path("runs") / f"dry-{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] run_dir={run_dir}")

    # -- train -------------------------------------------------------------
    train_cfg = _build_train_cfg(cfg)
    best_path, last_path = train(
        model=model,
        train_loader=loader,
        cfg=train_cfg,
        device=device,
        run=run,
        run_dir=run_dir,
        start_epoch=start_epoch,
        start_step=start_step,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
    )

    print(f"\n[train] done.  best={best_path}  last={last_path}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
