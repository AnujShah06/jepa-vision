"""
jepa.py -- VisionJEPA: patch-based JEPA for 96x96 images.

Architecture follows I-JEPA (Assran et al. 2023) adapted from the cocktail
repo's recipe-slot design:
  - PatchEmbed: 2D conv to flatten image into N patch tokens
  - 2D sin-cos positional embeddings (images have spatial structure)
  - ViTEncoder: stack of standard transformer blocks (context + EMA target)
  - Predictor: same-width transformer; input = context tokens + mask tokens
    at target positions; output at mask positions = predicted target latents
  - EMA update: ported verbatim from cocktail_jepa CocktailJEPA.ema_update

Smoke-test (Gate 0) uses a tiny override: d_model=64, 2 enc + 2 pred layers.
Full production architecture (d=192, 6 layers, 3 heads) is Step 1.2, where
proper block masking, the separate pred_width, and per-patch heatmaps land.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.loss import jepa_loss


# ---------------------------------------------------------------------------
# Positional embedding
# ---------------------------------------------------------------------------

def build_2d_sincos_pos_embed(n_h: int, n_w: int, d_model: int) -> torch.Tensor:
    """
    2D sin-cos positional embedding. Returns [1, n_h*n_w, d_model].

    Splits d_model into four equal quarters: sin_y, cos_y, sin_x, cos_x.
    Requires d_model divisible by 4.
    """
    assert d_model % 4 == 0, f"d_model ({d_model}) must be divisible by 4"
    q = d_model // 4

    freq = 1.0 / (10000 ** (torch.arange(q).float() / q))  # [q]

    gy = torch.arange(n_h).float()
    gx = torch.arange(n_w).float()

    y_emb = torch.outer(gy, freq)  # [n_h, q]
    x_emb = torch.outer(gx, freq)  # [n_w, q]

    # broadcast to [n_h, n_w, q]
    sin_y = torch.sin(y_emb).unsqueeze(1).expand(-1, n_w, -1)
    cos_y = torch.cos(y_emb).unsqueeze(1).expand(-1, n_w, -1)
    sin_x = torch.sin(x_emb).unsqueeze(0).expand(n_h, -1, -1)
    cos_x = torch.cos(x_emb).unsqueeze(0).expand(n_h, -1, -1)

    emb = torch.cat([sin_y, cos_y, sin_x, cos_x], dim=-1)  # [n_h, n_w, d_model]
    return emb.reshape(1, n_h * n_w, d_model)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Conv2d-based patch tokeniser. [B, 3, H, W] -> [B, N, d_model]."""

    def __init__(self, img_size: int, patch_size: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, d]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Standard ViT encoder: stack of TransformerBlocks + final LayerNorm."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VisionJEPAConfig:
    """Architecture + loss hyperparameters for VisionJEPA."""
    # image / patch
    img_size: int = 96
    patch_size: int = 8           # 96/8 -> 12x12 = 144 tokens
    # context / target encoder (ViT-Tiny for production)
    d_model: int = 192
    enc_layers: int = 6
    enc_heads: int = 3            # 192/3 = 64 per head
    # predictor (lower capacity -- Step 1.2 adds separate pred_width projection)
    pred_layers: int = 3
    pred_heads: int = 3           # must divide d_model
    # regularizer
    sigreg_weight: float = 1.0
    sigreg_projections: int = 64
    # EMA
    ema_decay: float = 0.996      # starting point; loop ramps toward 1.0
    use_ema: bool = True
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# VisionJEPA
# ---------------------------------------------------------------------------

class VisionJEPA(nn.Module):
    """
    Vision JEPA: patch tokeniser + 2D sincos pos-embed + ViT context/target
    encoders + transformer predictor + EMA update.

    forward() batch keys:
      imgs        [B, 3, H, W]
      target_idx  [N_tgt]  patch indices the predictor must predict
      context_idx [N_ctx]  patch indices the context encoder sees

    forward() returns a loss dict: {"loss", "pred_loss", "sigreg_term",
    "context_emb"} where context_emb [B, N_ctx, d] is used for diagnostics.
    """

    def __init__(self, cfg: VisionJEPAConfig | None = None):
        super().__init__()
        self.cfg = cfg or VisionJEPAConfig()
        c = self.cfg
        n_h = c.img_size // c.patch_size
        n_w = c.img_size // c.patch_size

        self.patch_embed = PatchEmbed(c.img_size, c.patch_size, c.d_model)

        pos_emb = build_2d_sincos_pos_embed(n_h, n_w, c.d_model)
        self.register_buffer("pos_embed", pos_emb)  # [1, N, d], not trained

        self.context_encoder = ViTEncoder(
            c.d_model, c.enc_layers, c.enc_heads, c.dropout
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # learnable mask token; added to target positional embeddings
        self.mask_token = nn.Parameter(torch.zeros(1, 1, c.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # predictor runs at same width as encoder (Step 1.2 adds proj for pred_width)
        self.predictor = ViTEncoder(
            c.d_model, c.pred_layers, c.pred_heads, c.dropout
        )

    # -- forward -----------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        imgs = batch["imgs"]            # [B, 3, H, W]
        target_idx = batch["target_idx"]   # [N_tgt]
        context_idx = batch["context_idx"] # [N_ctx]

        B = imgs.shape[0]
        N_ctx = context_idx.shape[0]

        # 1. Patchify + positional embedding: [B, N, d]
        tokens = self.patch_embed(imgs) + self.pos_embed

        # 2. Context encoder: sees only context patches
        ctx_tokens = tokens[:, context_idx, :]          # [B, N_ctx, d]
        ctx_emb = self.context_encoder(ctx_tokens)       # [B, N_ctx, d]

        # 3. Target encoder (EMA, no grad): sees all patches, take target pos
        with torch.no_grad():
            full_emb = self.target_encoder(tokens)       # [B, N, d]
        target_latent = full_emb[:, target_idx, :].detach()  # [B, N_tgt, d]

        # 4. Predictor: context embeddings + mask tokens at target positions
        tgt_pos = self.pos_embed[:, target_idx, :]       # [1, N_tgt, d]
        mask_tokens = self.mask_token.expand(B, target_idx.shape[0], -1) + tgt_pos
        pred_input = torch.cat([ctx_emb, mask_tokens], dim=1)  # [B, N_ctx+N_tgt, d]
        pred_out = self.predictor(pred_input)
        predicted = pred_out[:, N_ctx:, :]               # [B, N_tgt, d]

        # 5. Loss: smooth-L1 prediction loss + SIGReg on context embeddings
        predicted_flat = predicted.reshape(-1, self.cfg.d_model)
        target_flat = target_latent.reshape(-1, self.cfg.d_model)
        encoder_emb = ctx_emb.reshape(-1, self.cfg.d_model)  # [B*N_ctx, d]

        loss_dict = jepa_loss(
            predicted=predicted_flat,
            target=target_flat,
            encoder_embeddings=encoder_emb,
            sigreg_weight=self.cfg.sigreg_weight,
            sigreg_projections=self.cfg.sigreg_projections,
        )
        loss_dict["context_emb"] = ctx_emb  # [B, N_ctx, d] -- for diagnostics
        return loss_dict

    # -- EMA update (ported verbatim from CocktailJEPA.ema_update) ----------

    @torch.no_grad()
    def ema_update(self, decay: float | None = None) -> None:
        """
        target <- decay * target + (1 - decay) * context

        Called after each optimizer step. The decay value for the current
        step is computed by loop.py's _ema_momentum() and passed in; falls
        back to cfg.ema_decay when omitted.
        """
        if not self.cfg.use_ema:
            return
        d = self.cfg.ema_decay if decay is None else decay
        for tgt, src in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            tgt.mul_(d).add_(src, alpha=1.0 - d)
        # buffers (e.g. LayerNorm running stats) copied outright
        for tgt, src in zip(self.target_encoder.buffers(),
                            self.context_encoder.buffers()):
            tgt.copy_(src)
