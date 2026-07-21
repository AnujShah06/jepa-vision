"""
Regression test for DECISIONS.md [2.0] checkpoint saver fix.

Asserts that train() does NOT produce best.ckpt after running.
Uses a 2-epoch smoke run (tiny model, 4 random images) so the test is fast.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.loop import TrainConfig, train
from src.models.jepa import VisionJEPA, VisionJEPAConfig

# Tiny grid: 32px / 8px = 4×4 = 16 tokens
_IMG_SIZE = 32
_PATCH = 8
_N_H = _N_W = _IMG_SIZE // _PATCH   # 4
_N_TOK = _N_H * _N_W                # 16


def _make_tiny_model() -> VisionJEPA:
    cfg = VisionJEPAConfig(
        img_size=_IMG_SIZE,
        patch_size=_PATCH,
        d_model=64,
        enc_layers=1,
        enc_heads=4,        # 64 / 4 = 16 head_dim
        pred_layers=1,
        pred_width=32,
        pred_heads=4,       # 32 / 4 = 8 head_dim
    )
    return VisionJEPA(cfg)


def _make_batch() -> dict:
    """Return a minimal JEPA batch dict matching VisionJEPA.forward expectations."""
    B = 2
    N_tgt = 4   # 25% of 16 tokens
    N_ctx = 12  # remaining
    tgt_idx = torch.arange(N_tgt)
    ctx_idx = torch.arange(N_tgt, N_tgt + N_ctx)
    return {
        "imgs": torch.randn(B, 3, _IMG_SIZE, _IMG_SIZE),
        "target_idx": tgt_idx,
        "context_idx": ctx_idx,
    }


class _FixedBatchLoader:
    """Minimal loader shim that yields two fixed batches per epoch."""
    def __iter__(self):
        yield _make_batch()
        yield _make_batch()

    def __len__(self):
        return 2


def test_no_best_ckpt_produced() -> None:
    model = _make_tiny_model()
    loader = _FixedBatchLoader()

    cfg = TrainConfig(
        epochs=2,
        ckpt_every=10,     # no periodic save in 2 epochs → final-epoch path fires
        diag_every=1,
        use_amp=False,
    )

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        canon_path = train(
            model=model,
            train_loader=loader,
            cfg=cfg,
            device="cpu",
            run=None,
            run_dir=run_dir,
        )

        # Primary assertion: best.ckpt must NOT exist
        best = run_dir / "best.ckpt"
        assert not best.exists(), (
            f"best.ckpt was created at {best} — saver fix not applied"
        )

        # Secondary: returned path must exist and be the final-epoch file
        assert canon_path.exists(), f"canonical path {canon_path} does not exist"
        assert canon_path.name == "epoch_0002.ckpt", (
            f"expected epoch_0002.ckpt, got {canon_path.name}"
        )
