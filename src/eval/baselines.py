"""
baselines.py -- Step 1.5 energy baselines.

Four baselines for the Gate 1B comparison table:

  pixel_stats_energy   -- pixel std (trivial floor; cheap, no model)
  random_init_energy   -- same image_energy pipeline on an untrained model
                          (proves training carries the signal, not architecture)
  fit_mahalanobis /
  mahalanobis_energy   -- Mahalanobis distance on frozen trained features
                          (standard OOD baseline)
  mae_energy           -- pixel-space MAE reconstruction error K-averaged
                          (the "why latent, not pixels" comparison)

All functions return [B] float tensors so auroc() from src.eval.evaluate
can be called directly.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.eval.energy import image_energy

if TYPE_CHECKING:
    from src.models.jepa import VisionJEPA
    from src.models.mae import PixelMAE


# ---------------------------------------------------------------------------
# 1. Pixel statistics energy
# ---------------------------------------------------------------------------

def pixel_stats_energy(images: torch.Tensor) -> torch.Tensor:
    """
    Per-image pixel standard deviation in the normalised tensor space.

    This is the trivial floor: gaussian noise directly inflates pixel std,
    so AUROC will be high for noise corruptions and near-chance for subtle
    semantic corruptions. That's the honest comparison point — it shows our
    latent energy carries signal beyond pixel-level statistics.

    Args:
        images: [B, 3, H, W] ImageNet-normalised float tensor.

    Returns:
        [B] float tensor of per-image pixel std values.
    """
    flat = images.reshape(images.shape[0], -1)   # [B, 3*H*W]
    return flat.std(dim=-1)


# ---------------------------------------------------------------------------
# 2. Random-init encoder energy
# ---------------------------------------------------------------------------

@torch.no_grad()
def random_init_energy(
    images: torch.Tensor,
    K: int = 8,
    seed: int | None = 0,
    device: str | None = None,
) -> torch.Tensor:
    """
    Energy from a randomly initialised VisionJEPA with the production config.

    Proves that *training*, not architecture, is responsible for AUROC signal.
    Expected result: AUROC ≈ 0.50 (random energy, no signal).

    A fresh model is constructed each call so there is no state leak between
    calls with different seeds.
    """
    from src.models.jepa import VisionJEPA, VisionJEPAConfig

    model = VisionJEPA(VisionJEPAConfig()).eval()
    if device is not None:
        model = model.to(device)
    result = image_energy(model, images, K=K, seed=seed, device=device)
    return result["energy"].cpu()


# ---------------------------------------------------------------------------
# 3. Mahalanobis distance on frozen trained features
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_encoder_features(
    model: "VisionJEPA",
    loader: DataLoader,
    device: str,
    n_samples: int = 2048,
) -> torch.Tensor:
    """
    Extract mean-pooled context-encoder features for n_samples images.

    All N=144 patch tokens are passed to the context encoder (no masking)
    and averaged into a single d_model-dimensional feature vector per image.

    Args:
        model:     trained VisionJEPA (context_encoder used, no EMA).
        loader:    plain image loader (yields (imgs, labels) or {"imgs": ...}).
        device:    computation device.
        n_samples: stop after collecting this many feature vectors.

    Returns:
        [n_samples, d_model] float tensor on CPU.
    """
    model.eval()
    feats: list[torch.Tensor] = []
    collected = 0

    for batch in loader:
        if collected >= n_samples:
            break
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch["imgs"]
        imgs = imgs.to(device)

        tokens = model.patch_embed(imgs) + model.pos_embed   # [B, N, d]
        out    = model.context_encoder(tokens)                # [B, N, d]
        feat   = out.mean(1).cpu()                            # [B, d]
        feats.append(feat)
        collected += feat.shape[0]

    return torch.cat(feats)[:n_samples]


def fit_mahalanobis(
    features: torch.Tensor,
    reg: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a multivariate Gaussian to `features` and return (mean, precision).

    Args:
        features: [N, d] float tensor of in-distribution feature vectors.
        reg:      diagonal regularisation added to the covariance matrix.

    Returns:
        mean:      [d] float tensor.
        precision: [d, d] float tensor (inverse of regularised covariance).
    """
    features = features.float()
    mean     = features.mean(0)                    # [d]
    centered = features - mean                     # [N, d]
    cov      = (centered.T @ centered) / max(1, len(features) - 1)   # [d, d]
    cov      = (cov + cov.T) / 2   # enforce symmetry before inversion
    cov_reg   = cov + reg * torch.eye(cov.shape[0], dtype=cov.dtype)
    precision = torch.linalg.inv(cov_reg)          # [d, d]
    precision = (precision + precision.T) / 2      # enforce symmetry post-inversion
    return mean, precision


@torch.no_grad()
def mahalanobis_energy(
    images: torch.Tensor,
    model: "VisionJEPA",
    mean: torch.Tensor,
    precision: torch.Tensor,
    device: str | None = None,
) -> torch.Tensor:
    """
    Mahalanobis distance from the in-distribution Gaussian.

    Higher distance = more anomalous = higher energy.

    Returns [B] float tensor of distances.
    """
    if device is None:
        device = next(model.parameters()).device

    images    = images.to(device)
    mean_dev  = mean.to(device)
    prec_dev  = precision.to(device)

    tokens = model.patch_embed(images) + model.pos_embed   # [B, N, d]
    out    = model.context_encoder(tokens)                  # [B, N, d]
    feat   = out.mean(1).float()                            # [B, d]

    diff = feat - mean_dev                                  # [B, d]
    # M_i = sqrt( diff_i^T @ Precision @ diff_i )
    mahal_sq = (diff @ prec_dev * diff).sum(-1).clamp(min=0.0)   # [B]
    return mahal_sq.sqrt().cpu()


# ---------------------------------------------------------------------------
# 4. MAE reconstruction energy
# ---------------------------------------------------------------------------

@torch.no_grad()
def mae_energy(
    mae_model: "PixelMAE",
    images: torch.Tensor,
    K: int = 8,
    seed: int | None = 0,
    device: str | None = None,
) -> torch.Tensor:
    """
    Mean MAE reconstruction error per image, averaged over K random masks.

    Higher error = harder to reconstruct = higher energy.
    With the untrained model this is a random baseline; with a trained MAE
    it becomes the pixel-level energy comparison.

    Returns [B] float tensor.
    """
    if device is None:
        device = next(mae_model.parameters()).device

    images = images.to(device)
    B = images.shape[0]

    rng = random.Random(seed)
    per_k: list[torch.Tensor] = []

    for _ in range(K):
        k_seed = rng.randint(0, 2**31)
        out = mae_model(images, seed=k_seed)
        per_k.append(out["loss_per_image"].detach().cpu())   # [B]

    return torch.stack(per_k).mean(0)   # [B]
