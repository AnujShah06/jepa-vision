"""
mae.py -- Pixel-space MAE baseline for Step 1.5.

Same ViT-Tiny encoder budget as VisionJEPA (d=192, 6 layers, 3 heads).
Lightweight decoder: 2-layer transformer at dec_width=128 + linear pixel
projection.

Standard random-mask MAE (75% masked), target = per-patch pixel values
optionally normalised per-patch (norm_pix_loss=True, as in He et al. 2022).

Architecture intentionally identical to VisionJEPA on the encoder side so
"trained JEPA vs trained MAE" is a fair comparison of the latent-prediction
objective against pixel-reconstruction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.jepa import PatchEmbed, ViTEncoder, build_2d_sincos_pos_embed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PixelMAEConfig:
    """Architecture hyperparameters for the pixel-MAE baseline."""
    img_size: int    = 96
    patch_size: int  = 8        # → 12×12 = 144 patches
    # encoder: identical budget to VisionJEPA
    d_model: int     = 192
    enc_layers: int  = 6
    enc_heads: int   = 3
    # decoder: lightweight — roughly 1/3 encoder cost
    dec_width: int   = 128
    dec_layers: int  = 2
    dec_heads: int   = 2        # dec_width / dec_heads = 64
    mask_ratio: float = 0.75   # fraction of patches masked (standard MAE)
    dropout: float   = 0.0
    norm_pix_loss: bool = True  # normalise pixel targets per patch


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PixelMAE(nn.Module):
    """
    Masked Auto-Encoder with a ViT-Tiny encoder and lightweight decoder.

    forward(imgs, seed) returns:
        loss           : scalar MSE reconstruction loss
        loss_per_image : [B] per-image MSE at masked patches
        mask_idx       : [N_mask] LongTensor — patch indices that were masked
        ctx_idx        : [N_ctx]  LongTensor — visible patch indices
    """

    def __init__(self, cfg: PixelMAEConfig | None = None):
        super().__init__()
        self.cfg = cfg or PixelMAEConfig()
        c = self.cfg
        n_h = c.img_size // c.patch_size
        n_w = c.img_size // c.patch_size

        self.patch_embed = PatchEmbed(c.img_size, c.patch_size, c.d_model)

        pos_emb = build_2d_sincos_pos_embed(n_h, n_w, c.d_model)
        self.register_buffer("pos_embed", pos_emb)   # [1, N, d_model]

        self.encoder = ViTEncoder(c.d_model, c.enc_layers, c.enc_heads, c.dropout)

        # Projection from encoder space to decoder space
        self.enc_to_dec = nn.Linear(c.d_model, c.dec_width, bias=True)

        # Decoder mask token in decoder space (+ sincos pos embed projected)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, c.dec_width))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Project sincos pos embed to dec_width for mask tokens
        self.pos_proj = nn.Linear(c.d_model, c.dec_width, bias=False)

        self.decoder = ViTEncoder(c.dec_width, c.dec_layers, c.dec_heads, c.dropout)

        # Project decoder output to raw pixel values at each masked patch
        self.pixel_proj = nn.Linear(
            c.dec_width, c.patch_size * c.patch_size * 3, bias=True
        )

    # -- helpers -----------------------------------------------------------

    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] → [B, N, patch_size²×3] in row-major patch order."""
        p = self.cfg.patch_size
        x = F.unfold(imgs, kernel_size=p, stride=p)  # [B, 3*p*p, N]
        return x.transpose(1, 2)                      # [B, N, 3*p*p]

    def _sample_mask(self, N: int, seed: int | None = None) -> tuple[list[int], list[int]]:
        """Return (mask_idx, ctx_idx) — random masking of N patches."""
        n_mask = int(N * self.cfg.mask_ratio)
        rng = random.Random(seed)
        all_idx = list(range(N))
        rng.shuffle(all_idx)
        return sorted(all_idx[:n_mask]), sorted(all_idx[n_mask:])

    # -- forward -----------------------------------------------------------

    def forward(self, imgs: torch.Tensor, seed: int | None = None) -> dict:
        B = imgs.shape[0]
        device = imgs.device

        # Tokenise
        tokens = self.patch_embed(imgs) + self.pos_embed   # [B, N, d_model]
        N = tokens.shape[1]

        mask_idx, ctx_idx = self._sample_mask(N, seed)
        mask_t = torch.tensor(mask_idx, dtype=torch.long, device=device)
        ctx_t  = torch.tensor(ctx_idx,  dtype=torch.long, device=device)

        # -- encode visible tokens ----------------------------------------
        ctx_tokens = tokens[:, ctx_t, :]        # [B, N_ctx, d_model]
        ctx_emb    = self.encoder(ctx_tokens)   # [B, N_ctx, d_model]
        ctx_proj   = self.enc_to_dec(ctx_emb)   # [B, N_ctx, dec_width]

        # -- build full decoder input (all N positions) -------------------
        # Start with projected positional embeddings at dec_width
        pos_dec = self.pos_proj(self.pos_embed)          # [1, N, dec_width]
        mask_toks = self.mask_token + pos_dec.expand(B, -1, -1)  # [B, N, dec_width]

        # Overwrite visible positions with encoded (projected) features.
        # Clone to avoid in-place autograd issues; align dtype because under
        # AMP the encoder/projection emit bfloat16 while pos_proj+mask_token
        # may stay float32.
        dec_input = mask_toks.clone().to(ctx_proj.dtype)
        dec_input[:, ctx_t, :] = ctx_proj

        # -- decode -------------------------------------------------------
        dec_out = self.decoder(dec_input)                           # [B, N, dec_width]
        pred    = self.pixel_proj(dec_out[:, mask_t, :])            # [B, N_mask, p²*3]

        # -- target pixels ------------------------------------------------
        patches = self._patchify(imgs)                              # [B, N, p²*3]
        target  = patches[:, mask_t, :].detach()                   # [B, N_mask, p²*3]

        if self.cfg.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # -- loss ----------------------------------------------------------
        sq_err = (pred - target).pow(2).mean(-1)    # [B, N_mask] — mean over pixels
        loss_per_image = sq_err.mean(-1)             # [B]
        loss = loss_per_image.mean()

        return {
            "loss":           loss,
            "loss_per_image": loss_per_image,
            "mask_idx":       mask_t,
            "ctx_idx":        ctx_t,
        }
