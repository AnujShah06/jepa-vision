"""
timing_run.py -- Step 0.2 / 1.3 compute reality check.

Instantiates the production Phase 1 VisionJEPA config, runs exactly one
epoch on the full STL-10 unlabeled split (100k images, batch 256, AMP),
and reports sec/epoch, peak unified memory, and extrapolated GPU-hours.

This is a measurement run only.  Nothing is logged to W&B, no checkpoints
are saved, and no hyperparameters are tuned.

Usage:
    uv run python scripts/timing_run.py
"""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.stl10 import get_pretrain_loader
from src.loop import _ema_momentum
from src.models.jepa import VisionJEPA, VisionJEPAConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROD_CFG = VisionJEPAConfig(
    img_size=96,
    patch_size=8,         # 144 tokens
    d_model=192,
    enc_layers=6,
    enc_heads=3,          # 192/3 = 64 per head
    pred_layers=3,
    pred_width=96,        # predictor at half width
    pred_heads=3,         # 96/3 = 32 per head
    sigreg_weight=1.0,
    sigreg_projections=64,
    ema_decay=0.996,
    use_ema=True,
    dropout=0.0,
)

BATCH_SIZE   = 256
N_WARMUP     = 3      # batches to run before starting the clock
TARGET_EPOCHS = 150
N_SEEDS       = 3


# ---------------------------------------------------------------------------
# Memory helpers (MPS unified memory / CUDA VRAM)
# ---------------------------------------------------------------------------

def _reset_peak(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: str) -> float:
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1024 ** 2
    elif device.startswith("cuda"):
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return 0.0


def _current_mb(device: str) -> float:
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1024 ** 2
    elif device.startswith("cuda"):
        return torch.cuda.memory_allocated(device) / 1024 ** 2
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # -- device ------------------------------------------------------------
    if torch.backends.mps.is_available():
        device = "mps"
        amp_dtype = torch.bfloat16
        amp_device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        amp_dtype = torch.float16
        amp_device = "cuda"
    else:
        device = "cpu"
        amp_dtype = torch.bfloat16
        amp_device = "cpu"

    print(f"[timing] device: {device}  AMP dtype: {amp_dtype}")

    # -- model -------------------------------------------------------------
    model = VisionJEPA(PROD_CFG).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[timing] params  trainable: {trainable:,}  total (incl. frozen target): {total:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.04,
    )

    # -- AMP context -------------------------------------------------------
    use_amp = (device != "cpu")
    try:
        amp_ctx = torch.autocast(device_type=amp_device, dtype=amp_dtype,
                                 enabled=use_amp)
        # quick smoke-check
        with amp_ctx:
            _ = torch.zeros(1, device=device)
    except Exception as e:
        print(f"[timing] AMP unavailable ({e}), running fp32")
        amp_ctx = contextlib.nullcontext()
        use_amp = False

    # GradScaler only needed for float16 (not bfloat16)
    scaler = (torch.cuda.amp.GradScaler()
              if (device.startswith("cuda") and amp_dtype == torch.float16)
              else None)

    # -- data --------------------------------------------------------------
    data_dir = Path(__file__).parent.parent / "data"
    n_h = PROD_CFG.img_size // PROD_CFG.patch_size
    print(f"[timing] building full STL-10 unlabeled loader  batch={BATCH_SIZE} ...")
    loader = get_pretrain_loader(
        data_dir,
        batch_size=BATCH_SIZE,
        n_h=n_h,
        n_w=n_h,
        num_workers=4,
        seed=0,
        drop_last=True,
    )
    n_batches = len(loader)
    print(f"[timing] {n_batches} batches / epoch  "
          f"({n_batches * BATCH_SIZE:,} images with drop_last)")

    # -- warmup (not timed) ------------------------------------------------
    print(f"[timing] warming up ({N_WARMUP} batches) ...")
    model.train()
    _reset_peak(device)
    warmup_iter = iter(loader)
    for _ in range(N_WARMUP):
        try:
            batch = next(warmup_iter)
        except StopIteration:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        with amp_ctx:
            out = model(batch)
            loss = out["loss"]
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        model.ema_update()

    # Synchronise before starting clock
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()

    _reset_peak(device)
    mem_after_warmup_mb = _current_mb(device)

    # -- timed epoch -------------------------------------------------------
    print(f"[timing] starting timed epoch ({n_batches} batches) ...")
    t0 = time.perf_counter()

    # We already consumed N_WARMUP batches from the iterator; restart the
    # loader for a clean full-epoch measurement.
    total_steps_epoch = n_batches
    model.train()
    peak_mem_mb = _current_mb(device)

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        with amp_ctx:
            out = model(batch)
            loss = out["loss"]

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.ema_update()

        # sample peak memory every 50 batches
        if batch_idx % 50 == 0:
            cur = _current_mb(device)
            if cur > peak_mem_mb:
                peak_mem_mb = cur

        if batch_idx % 50 == 0 or batch_idx == n_batches - 1:
            elapsed = time.perf_counter() - t0
            rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_batches - batch_idx - 1) / rate if rate > 0 else 0
            print(f"  batch {batch_idx+1:4d}/{n_batches}  "
                  f"loss={loss.item():.4f}  "
                  f"mem={_current_mb(device):.0f}MB  "
                  f"eta={eta:.0f}s", end="\r", flush=True)

    # Final sync before stopping clock
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed_s = time.perf_counter() - t0

    # driver-level peak (includes fragmentation; conservative upper bound)
    driver_peak_mb = _peak_mb(device)

    # -- Results -----------------------------------------------------------
    print()  # clear the \r line
    print()
    print("=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)
    print(f"  Device               : {device}")
    print(f"  AMP                  : {'bfloat16' if use_amp else 'fp32'}")
    print(f"  Batch size           : {BATCH_SIZE}")
    print(f"  Batches / epoch      : {n_batches}")
    print(f"  Images / epoch       : {n_batches * BATCH_SIZE:,}")
    print(f"  Trainable params     : {trainable:,}")
    print(f"  Total params         : {total:,}")
    print()
    print(f"  Epoch wall time      : {elapsed_s:.1f} s  ({elapsed_s/60:.2f} min)")
    print(f"  Throughput           : {n_batches * BATCH_SIZE / elapsed_s:.0f} images/s")
    print(f"  Peak mem (allocated) : {peak_mem_mb:.0f} MB")
    print(f"  Peak mem (driver)    : {driver_peak_mb:.0f} MB")
    print()

    h_per_epoch = elapsed_s / 3600
    h_150       = h_per_epoch * TARGET_EPOCHS
    h_3seeds    = h_150 * N_SEEDS

    print(f"  Extrapolated  150 epochs        : {h_150:.1f} h  ({h_150/24:.1f} days)")
    print(f"  Extrapolated  150 × {N_SEEDS} seeds   : {h_3seeds:.1f} h  ({h_3seeds/24:.1f} days)")
    print()

    if h_150 > 8:
        print("  ⚠  >8 h for a single run -- cloud GPU recommended for Phase 1.")
        print("     Options (PLAYBOOK §0.2): Colab Pro A100, Lightning AI, Kaggle T4.")
        if h_150 > 40:
            print("     Consider shrinking to d=128 or img_size=64 if cloud is unavailable.")
    else:
        print("  ✓ Fits comfortably on this machine.")
    print("=" * 60)


if __name__ == "__main__":
    main()
