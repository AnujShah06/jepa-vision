"""
stl10.py -- STL-10 data loading for JEPA pretraining.

get_smoke_loader() wraps 100 unlabeled images with the I-JEPA block-masking
collator from src/data/masking.py.  Step 1.1 (full pipeline) adds:
  - full 100k unlabeled pretraining loader
  - probe train/val/test split carve (1k val from 5k labeled train)
  - proper RandomResizedCrop + horizontal flip augmentation for pretraining
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import STL10

from src.data.masking import IJEPAMaskCollator

VAL_SPLIT_FILE = Path(__file__).parent.parent.parent / "data" / "splits" / "stl10_val_idx.json"


def get_pretrain_loader(
    data_dir: str | Path,
    batch_size: int = 256,
    n_h: int = 12,
    n_w: int = 12,
    num_workers: int = 4,
    seed: int | None = None,
    drop_last: bool = True,
    mask_kwargs: dict | None = None,
) -> DataLoader:
    """
    Full 100k STL-10 unlabeled pretraining loader with proper augmentation.

    Transforms per PLAYBOOK §1.1:
      RandomResizedCrop(96, scale 0.3-1.0) + RandomHorizontalFlip + Normalize.
      No colour jitter -- JEPA-family methods deliberately avoid heavy augmentation.

    mask_kwargs: optional dict of IJEPAMaskCollator kwargs (n_targets,
      target_scale, target_aspect, context_scale, context_aspect). If None,
      the collator's defaults (4 targets, target 0.15-0.20, ctx 0.85-1.00) apply.

    Batch format: same as get_smoke_loader.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.RandomResizedCrop(96, scale=(0.3, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = STL10(root=str(data_dir), split="unlabeled",
               transform=transform, download=True)

    collator = IJEPAMaskCollator(n_h=n_h, n_w=n_w, seed=seed,
                                 **(mask_kwargs or {}))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,   # MPS does not use pinned memory
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )


def get_smoke_loader(
    data_dir: str | Path,
    n_images: int = 100,
    batch_size: int = 16,
    n_h: int = 12,
    n_w: int = 12,
    seed: int | None = None,
    mask_kwargs: dict | None = None,
) -> DataLoader:
    """
    DataLoader over `n_images` from the STL-10 unlabeled split.
    Downloads to data_dir on first call (~2.6 GB, one-time).

    Batch format:
      imgs        FloatTensor [B, 3, 96, 96]  normalised
      target_idx  LongTensor  [N_tgt]          I-JEPA target patch indices
      context_idx LongTensor  [N_ctx]          I-JEPA context patch indices

    Masking: I-JEPA block masking (4 target blocks + 1 context block)
    via IJEPAMaskCollator. One mask per batch, refreshed each batch.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = STL10(root=str(data_dir), split="unlabeled",
               transform=transform, download=True)
    ds = Subset(ds, list(range(min(n_images, len(ds)))))

    collator = IJEPAMaskCollator(n_h=n_h, n_w=n_w, seed=seed,
                                 **(mask_kwargs or {}))

    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=collator, num_workers=0)


def get_val_loader(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = False,
    split_file: str | Path | None = None,
) -> DataLoader:
    """
    Formal 1,000-image validation loader.

    Indices are loaded from data/splits/stl10_val_idx.json (committed;
    100 images per class, seed=0 stratified sample from STL-10 labeled
    train).  Probe training must EXCLUDE these indices.

    Returns (imgs [B, 3, 96, 96], labels [B]) batches.
    """
    sf = Path(split_file) if split_file else VAL_SPLIT_FILE
    with open(sf) as f:
        meta = json.load(f)
    indices = meta["indices"]

    return get_eval_loader(
        data_dir=data_dir,
        split="train",
        indices=indices,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )


def get_probe_train_loader(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    n_images: int | None = None,
    split_file: str | Path | None = None,
) -> DataLoader:
    """
    STL-10 labeled train images with val indices excluded.

    Returns (imgs [B, 3, 96, 96], labels [B]) batches.
    n_images: optional cap (useful for low-label-fraction experiments).
    """
    sf = Path(split_file) if split_file else VAL_SPLIT_FILE
    with open(sf) as f:
        meta = json.load(f)
    val_set = set(meta["indices"])

    probe_indices = [i for i in range(meta["n_train_total"]) if i not in val_set]
    if n_images is not None:
        probe_indices = probe_indices[:n_images]

    return get_eval_loader(
        data_dir=data_dir,
        split="train",
        indices=probe_indices,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )


def get_eval_loader(
    data_dir: str | Path,
    split: str = "train",
    n_images: int | None = None,
    indices: list[int] | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """
    Plain image loader for evaluation — no masking collator.

    Returns (imgs [B, 3, 96, 96], labels [B]) batches from the STL-10
    labeled split (train = 5k, test = 8k).

    indices:  explicit list of dataset indices (takes priority over n_images).
    n_images: if indices is None, take the first n_images (None = all).

    Prefer get_val_loader() for the formal validation set and
    get_probe_train_loader() for the probe training set.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize(96, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(96),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = STL10(root=str(data_dir), split=split,
               transform=transform, download=True)

    if indices is not None:
        ds = Subset(ds, indices)
    elif n_images is not None:
        ds = Subset(ds, list(range(min(n_images, len(ds)))))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
