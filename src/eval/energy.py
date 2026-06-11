"""
energy.py -- multi-mask latent prediction energy for VisionJEPA.

Energy(image) = mean latent prediction error over target patches,
averaged over K independent mask samples drawn at eval time.

Two views of the same computation:
  - Scalar energy: the K-averaged mean-patch error, used for AUROC.
  - Patch energy map: per-patch mean error accumulated across K masks,
    reshaped to the 12×12 grid and upsampled for spatial heatmaps.

Mask sampling is eval-side (sample_block_mask called directly with a
seeded random.Random); the training collator is never touched.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.masking import sample_block_mask

if TYPE_CHECKING:
    from src.models.jepa import VisionJEPA

# ImageNet normalisation constants (used for heatmap denormalisation only)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# Core energy function
# ---------------------------------------------------------------------------

@torch.no_grad()
def image_energy(
    model: "VisionJEPA",
    images: torch.Tensor,
    K: int = 8,
    seed: int | None = 0,
    device: str | None = None,
) -> dict:
    """
    Compute per-image energy and per-patch energy map.

    For each of K independent mask samples:
      1. Run context encoder on context patches.
      2. Run target encoder (frozen) on all patches, take target positions.
      3. Run predictor; compute smooth-L1 error per patch.
    Accumulate patch errors, then average over K samples.

    Args:
        model:  trained VisionJEPA (eval mode recommended but not required).
        images: [B, 3, H, W] normalised image tensor.
        K:      number of independent mask samples per image.
        seed:   seed for the mask sampler; None = non-deterministic.
        device: device for computation; defaults to model parameter device.

    Returns dict:
        "energy":       [B] float tensor — scalar energy per image.
        "patch_energy": [B, N] float tensor — per-patch mean error
                        (N = n_h * n_w; positions never sampled as target
                        keep value 0.0 and are excluded from the scalar).
    """
    if device is None:
        device = next(model.parameters()).device

    images = images.to(device)
    B = images.shape[0]

    cfg = model.cfg
    n_h = cfg.img_size // cfg.patch_size
    n_w = cfg.img_size // cfg.patch_size
    N   = n_h * n_w

    rng = random.Random(seed)

    # Patchify + pos embed once (shared across K mask samples)
    tokens = model.patch_embed(images) + model.pos_embed   # [B, N, d]

    # Target encoder runs once — it is frozen and mask-independent
    full_emb = model.target_encoder(tokens)                # [B, N, d]

    # Accumulators for per-patch error and visit counts
    patch_sum = torch.zeros(B, N, device=device)
    patch_cnt = torch.zeros(B, N, device=device)

    for _ in range(K):
        mask = sample_block_mask(n_h=n_h, n_w=n_w, rng=rng)
        tgt_idx = torch.tensor(mask.target_patches,  dtype=torch.long, device=device)
        ctx_idx = torch.tensor(mask.context_patches, dtype=torch.long, device=device)
        N_ctx   = ctx_idx.shape[0]

        # Context encoder
        ctx_tokens = tokens[:, ctx_idx, :]           # [B, N_ctx, d]
        ctx_emb    = model.context_encoder(ctx_tokens)  # [B, N_ctx, d]

        # Target latents at target positions
        target_latent = full_emb[:, tgt_idx, :].detach()  # [B, N_tgt, d]

        # Predictor: concat context embeddings + positional mask tokens
        tgt_pos     = model.pos_embed[:, tgt_idx, :]                           # [1, N_tgt, d]
        mask_tokens = model.mask_token.expand(B, tgt_idx.shape[0], -1) + tgt_pos
        pred_input  = torch.cat([ctx_emb, mask_tokens], dim=1)                 # [B, N_ctx+N_tgt, d]
        pred_input  = model.pred_proj_in(pred_input)
        pred_out    = model.predictor(pred_input)
        predicted   = model.pred_proj_out(pred_out[:, N_ctx:, :])              # [B, N_tgt, d]

        # Per-patch smooth-L1 error (mean over embedding dimension) → [B, N_tgt]
        patch_err = F.smooth_l1_loss(
            predicted, target_latent, reduction="none"
        ).mean(-1)

        # Scatter into accumulators
        patch_sum[:, tgt_idx] += patch_err
        patch_cnt[:, tgt_idx] += 1.0

    # Average over K samples; unvisited positions remain 0
    safe_cnt     = patch_cnt.clamp(min=1.0)
    patch_energy = patch_sum / safe_cnt              # [B, N]

    # Scalar energy: mean over visited positions only
    visited      = (patch_cnt > 0).float()           # [B, N]
    n_visited    = visited.sum(-1).clamp(min=1.0)    # [B]
    energy       = (patch_energy * visited).sum(-1) / n_visited   # [B]

    return {"energy": energy, "patch_energy": patch_energy}


# ---------------------------------------------------------------------------
# Loader-level helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def energy_over_loader(
    model: "VisionJEPA",
    loader: DataLoader,
    K: int = 8,
    seed: int | None = 0,
    device: str | None = None,
) -> torch.Tensor:
    """
    Compute scalar energy for every image delivered by *loader*.

    The loader must yield plain image tensors [B, 3, H, W] or
    (image, label) tuples — labels are ignored.

    Returns [N] float tensor of energies in loader order.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_energies: list[torch.Tensor] = []

    # Use a reproducible but per-image-shifted seed across batches so
    # different batches don't all draw the same K masks.
    batch_seed = seed

    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch["imgs"]
        result = image_energy(model, imgs, K=K, seed=batch_seed, device=device)
        all_energies.append(result["energy"].cpu())
        if batch_seed is not None:
            batch_seed += 1   # advance seed so successive batches differ

    return torch.cat(all_energies)


# ---------------------------------------------------------------------------
# Spatial heatmap
# ---------------------------------------------------------------------------

def energy_heatmap(
    patch_energy: torch.Tensor,
    src_image: torch.Tensor,
    n_h: int = 12,
    n_w: int = 12,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Produce a heatmap overlay of per-patch energy on the source image.

    Args:
        patch_energy: [N] or [n_h, n_w] energy per patch.
        src_image:    [3, H, W] ImageNet-normalised image tensor.
        n_h, n_w:     patch grid dimensions.
        alpha:        heatmap opacity (0 = src only, 1 = heatmap only).

    Returns:
        [H, W, 3] uint8 numpy array — ready for matplotlib imshow / PIL.
    """
    import matplotlib.cm as cm  # imported here to keep top-level import light

    N  = n_h * n_w
    pe = patch_energy.detach().cpu().float()

    if pe.shape != (n_h, n_w):
        pe = pe.reshape(n_h, n_w)      # [n_h, n_w]

    H, W = src_image.shape[-2], src_image.shape[-1]

    # Upsample patch grid to image resolution
    heat = pe.unsqueeze(0).unsqueeze(0)                       # [1, 1, n_h, n_w]
    heat = F.interpolate(heat, size=(H, W), mode="bilinear",
                         align_corners=False).squeeze()       # [H, W]

    # Normalise to [0, 1] for colormap
    lo, hi = heat.min(), heat.max()
    if hi > lo:
        heat_norm = ((heat - lo) / (hi - lo)).numpy()
    else:
        heat_norm = np.zeros((H, W), dtype=np.float32)

    # Apply 'hot' colormap → [H, W, 4] RGBA in [0, 1]
    heatmap_rgba = cm.hot(heat_norm)[:, :, :3].astype(np.float32)  # [H, W, 3]

    # Denormalise source image to [0, 1]
    img = src_image.detach().cpu().float()                    # [3, H, W]
    mean = _IMAGENET_MEAN.view(3, 1, 1)
    std  = _IMAGENET_STD.view(3, 1, 1)
    img  = (img * std + mean).clamp(0.0, 1.0)
    img_np = img.permute(1, 2, 0).numpy()                     # [H, W, 3]

    # Alpha blend
    overlay = (1.0 - alpha) * img_np + alpha * heatmap_rgba
    overlay = (overlay.clip(0.0, 1.0) * 255).astype(np.uint8)
    return overlay
