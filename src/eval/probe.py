"""
probe.py — Step 1.5d: label-scarce transfer harness.

Two evaluations on the formal val split at n ∈ {40, 200, 400, 4000}:

  (A) Frozen-encoder linear probe:
        Pass all N=144 patches through the trained JEPA context encoder
        (no masking), mean-pool → single d_model vector, then train a
        nn.Linear(d_model, n_classes) head on the frozen features.

  (B) From-scratch comparator:
        Identical ViT-Tiny backbone + linear head trained end-to-end on the
        same n images.  LR swept over {1e-3, 3e-4, 1e-4}; best chosen on
        the formal val split.  Same weight_decay as probe.

Pooling: mean over all N=144 patch tokens.  VisionJEPA has no CLS token;
see DECISIONS.md [1.5d] for the rationale.

Public API:
    stratified_sample    — sample n_per_class indices per class
    get_probe_pool       — probe pool indices + labels from disk
    extract_features     — run encoder on a loader, return (feats, labels)
    train_probe          — train linear head on frozen features
    ScratchClassifier    — ViT-Tiny + linear head
    train_scratch        — lr-sweep + epoch training for scratch model
"""

from __future__ import annotations

import contextlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

VAL_SPLIT_FILE = Path(__file__).parent.parent.parent / "data" / "splits" / "stl10_val_idx.json"


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    labels:       list[int] | torch.Tensor,
    n_per_class:  int,
    n_classes:    int = 10,
    seed:         int = 0,
) -> list[int]:
    """
    Return a sorted list of indices (into *labels*) with exactly n_per_class
    from each class.

    Args:
        labels:      sequence of integer class labels (0-indexed).
        n_per_class: how many examples to draw from each class.
        n_classes:   number of distinct classes.
        seed:        RNG seed for reproducibility.

    Raises ValueError if any class has fewer than n_per_class examples.
    """
    rng = random.Random(seed)
    by_class: list[list[int]] = [[] for _ in range(n_classes)]
    for idx, lbl in enumerate(labels):
        lbl = int(lbl)
        if 0 <= lbl < n_classes:
            by_class[lbl].append(idx)

    result: list[int] = []
    for cls, cls_idx in enumerate(by_class):
        if len(cls_idx) < n_per_class:
            raise ValueError(
                f"Class {cls}: only {len(cls_idx)} samples, need {n_per_class}"
            )
        result.extend(rng.sample(cls_idx, n_per_class))
    return sorted(result)


def get_probe_pool(
    data_dir:   str | Path,
    split_file: str | Path | None = None,
) -> tuple[list[int], list[int]]:
    """
    Return (probe_indices, probe_labels) for the 4 000-image probe pool.

    probe_indices: indices into the STL-10 labeled train split (0-4999).
    probe_labels:  corresponding class labels (0-9).
    """
    from torchvision.datasets import STL10

    sf = Path(split_file) if split_file else VAL_SPLIT_FILE
    with open(sf) as f:
        meta = json.load(f)

    val_set = set(meta["indices"])
    probe_indices = [i for i in range(meta["n_train_total"]) if i not in val_set]

    ds = STL10(root=str(Path(data_dir)), split="train", download=False)
    probe_labels = [int(ds.labels[i]) for i in probe_indices]

    return probe_indices, probe_labels


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model:  object,           # VisionJEPA (uses .patch_embed, .pos_embed, .context_encoder)
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean-pooled context-encoder features for all images in *loader*.

    Runs all N=144 patches through the context encoder (no masking) and
    mean-pools to a d_model-dimensional vector per image.

    Returns:
        features : [N, d_model] float32 on CPU
        labels   : [N] int64 on CPU
    """
    model.eval()
    feats:  list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for batch in loader:
        imgs = batch[0].to(device)
        lbl  = batch[1]

        tokens = model.patch_embed(imgs) + model.pos_embed   # [B, N, d]
        out    = model.context_encoder(tokens)                # [B, N, d]
        feat   = out.mean(1).cpu()                            # [B, d]
        feats.append(feat)
        labels.append(lbl.cpu())

    return torch.cat(feats), torch.cat(labels).long()


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def train_probe(
    train_feats:  torch.Tensor,
    train_labels: torch.Tensor,
    val_feats:    torch.Tensor,
    val_labels:   torch.Tensor,
    epochs:       int   = 100,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size:   int   = 256,
    device:       str   = "cpu",
) -> tuple[nn.Linear, float]:
    """
    Train a linear classification head on frozen features.

    Model selection: the checkpoint with the highest val accuracy is returned.

    Returns:
        head         : nn.Linear on CPU (best val checkpoint)
        best_val_acc : float in [0, 1]
    """
    d = train_feats.shape[1]
    n_classes = int(train_labels.max().item()) + 1
    head = nn.Linear(d, n_classes).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    train_f = train_feats.to(device)
    train_l = train_labels.to(device)
    val_f   = val_feats.to(device)
    val_l   = val_labels.to(device)
    N       = train_f.shape[0]

    best_acc:   float = -1.0
    best_state: dict  = {}

    for _ in range(epochs):
        head.train()
        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            idx    = perm[start : start + batch_size]
            loss   = F.cross_entropy(head(train_f[idx]), train_l[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            acc = (head(val_f).argmax(1) == val_l).float().mean().item()
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    return head.cpu(), best_acc


# ---------------------------------------------------------------------------
# From-scratch comparator
# ---------------------------------------------------------------------------

class ScratchClassifier(nn.Module):
    """
    ViT-Tiny backbone + linear head, trained end-to-end from random init.

    Identical encoder capacity to VisionJEPA (d=192, 6 layers, 3 heads).
    Mean-pools all N=144 patch tokens before the linear head.
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()
        from src.models.jepa import (
            PatchEmbed, ViTEncoder,
            build_2d_sincos_pos_embed, VisionJEPAConfig,
        )
        cfg = VisionJEPAConfig()
        n_h = cfg.img_size // cfg.patch_size

        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.d_model)
        self.register_buffer(
            "pos_embed",
            build_2d_sincos_pos_embed(n_h, n_h, cfg.d_model),
        )
        self.encoder = ViTEncoder(cfg.d_model, cfg.enc_layers, cfg.enc_heads, cfg.dropout)
        self.head    = nn.Linear(cfg.d_model, n_classes)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(imgs) + self.pos_embed
        feat   = self.encoder(tokens).mean(1)
        return self.head(feat)


@torch.no_grad()
def _eval_loader_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        imgs   = batch[0].to(device)
        labels = batch[1].to(device)
        preds  = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.shape[0]
    return correct / max(1, total)


def train_scratch(
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    lr_list:       list[float] | tuple[float, ...] = (1e-3, 3e-4, 1e-4),
    epochs:        int   = 200,
    weight_decay:  float = 0.05,
    warmup_epochs: int   = 10,
    device:        str   = "cpu",
    seed:          int   = 0,
    use_amp:       bool  = False,
) -> tuple[float, float]:
    """
    Train a ScratchClassifier end-to-end for each lr in lr_list.

    Cosine lr schedule with warmup; best val accuracy across all lrs returned.

    Returns:
        best_val_acc : float in [0, 1]
        best_lr      : float — lr that produced best_val_acc
    """
    n_batches    = len(train_loader)
    total_steps  = epochs * n_batches
    warmup_steps = warmup_epochs * n_batches

    # AMP context: bfloat16 on MPS/CUDA, no-op otherwise
    if use_amp:
        amp_dtype  = torch.bfloat16
        amp_device = "mps" if device == "mps" else ("cuda" if "cuda" in device else "cpu")
        try:
            _test_ctx = torch.autocast(device_type=amp_device, dtype=amp_dtype)
            with _test_ctx:
                pass
            amp_ctx: contextlib.AbstractContextManager = torch.autocast(
                device_type=amp_device, dtype=amp_dtype
            )
        except Exception:
            amp_ctx = contextlib.nullcontext()
    else:
        amp_ctx = contextlib.nullcontext()

    best_acc = -1.0
    best_lr  = float(lr_list[0])

    for lr in lr_list:
        torch.manual_seed(seed)
        model = ScratchClassifier().to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        def _schedule(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _schedule)
        run_best  = -1.0
        step      = 0

        for _ in range(epochs):
            model.train()
            for batch in train_loader:
                imgs   = batch[0].to(device)
                labels = batch[1].to(device)
                with amp_ctx:
                    loss = F.cross_entropy(model(imgs), labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()
                step += 1

            acc = _eval_loader_acc(model, val_loader, device)
            if acc > run_best:
                run_best = acc

        if run_best > best_acc:
            best_acc = run_best
            best_lr  = lr

    return best_acc, best_lr
