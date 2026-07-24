"""
resisc45.py -- RESISC45 data loading for Phase-2 JEPA pretraining.

Reads from the pre-committed split JSONs (data/splits/resisc45_*.json).
Images are 256×256 on disk; resized to 96×96 for harness compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from src.data.masking import IJEPAMaskCollator

RESISC_ROOT = Path(__file__).parent.parent.parent / "data" / "resisc45"
SPLITS_DIR  = Path(__file__).parent.parent.parent / "data" / "splits"

IMG_SIZE = 96   # resize target, matching Phase-1 / harness convention
N_H = N_W = IMG_SIZE // 8  # 12×12 = 144 tokens (patch_size=8)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class RESISC45Dataset(Dataset):
    """
    Minimal dataset over a committed split JSON.

    Each JSON entry is a relative path under data/resisc45/, e.g.
    "NWPU-RESISC45/airport/airport_001.jpg".  The dataset returns
    (image_tensor, class_name_str) — the collator uses only s[0].
    """

    def __init__(self, split_json: Path, transform: Callable | None = None) -> None:
        with open(split_json) as f:
            self.paths = json.load(f)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        rel = self.paths[idx]
        img = Image.open(RESISC_ROOT / rel).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # class name is the parent directory of the file
        cls = Path(rel).parent.name
        return img, cls


def _pretrain_transform(img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_resisc45_pretrain_loader(
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int | None = None,
    drop_last: bool = True,
    img_size: int = IMG_SIZE,
    mask_kwargs: dict | None = None,
) -> DataLoader:
    """
    RESISC45 40-class train split loader with JEPA block-masking collator.
    Uses resisc45_train_idx.json (22,400 images, quarantine excluded).
    """
    split_json = SPLITS_DIR / "resisc45_train_idx.json"
    ds = RESISC45Dataset(split_json, transform=_pretrain_transform(img_size))
    n_h = n_w = img_size // 8
    collator = IJEPAMaskCollator(n_h=n_h, n_w=n_w, seed=seed, **(mask_kwargs or {}))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )


def get_resisc45_val_loader(
    batch_size: int = 256,
    num_workers: int = 4,
    img_size: int = IMG_SIZE,
) -> DataLoader:
    """Plain val loader (no masking, no shuffle) for probe and energy eval."""
    split_json = SPLITS_DIR / "resisc45_val_idx.json"
    transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    ds = RESISC45Dataset(split_json, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)


def get_resisc45_quarantine_loader(
    batch_size: int = 256,
    num_workers: int = 4,
    img_size: int = IMG_SIZE,
) -> DataLoader:
    """Quarantine class loader for OOD / unseen-anomaly evaluation."""
    split_json = SPLITS_DIR / "resisc45_quarantine.json"
    transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    ds = RESISC45Dataset(split_json, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)
