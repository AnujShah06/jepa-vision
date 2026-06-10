"""
jepa.py -- VisionJEPA model stub.

Architecture (ViT-Tiny context/target encoders + low-capacity predictor +
block-masking module) is built in Step 1.2. The ema_update() method is
ported here now because loop.py calls it from day one and the decay-schedule
handling belongs next to the EMA implementation.

Ported from: cocktail_jepa/src/cocktail_jepa/model/jepa.py (CocktailJEPA)
Changes: removed recipe-specific components (TokenEncoder, proportion head,
set-transformer encoders); kept ema_update verbatim including buffer copying
and use_ema no-op path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class VisionJEPAConfig:
    """Architecture + loss hyperparameters for VisionJEPA."""
    # encoder
    img_size: int = 96
    patch_size: int = 8           # 96/8 = 12x12 = 144 tokens
    d_model: int = 192
    enc_layers: int = 6
    enc_heads: int = 3
    # predictor
    pred_layers: int = 3
    pred_width: int = 96
    pred_heads: int = 3
    # regularizer
    sigreg_weight: float = 1.0
    sigreg_projections: int = 64
    # EMA
    ema_decay: float = 0.996      # starting value; loop ramps toward 1.0
    use_ema: bool = True
    dropout: float = 0.0
    # TODO Step 1.2: masking hyperparameters (n_targets, target_scale, etc.)


class VisionJEPA(nn.Module):
    """
    Vision JEPA: ViT-Tiny context/target encoders + predictor + block masking.

    Full architecture body is TODO Step 1.2. ema_update() is functional now.
    """

    def __init__(self, cfg: VisionJEPAConfig | None = None):
        super().__init__()
        self.cfg = cfg or VisionJEPAConfig()
        # TODO Step 1.2: construct context_encoder, target_encoder, predictor,
        #   positional embeddings, mask-token embedding, patch embedding layer.
        self.context_encoder: nn.Module | None = None
        self.target_encoder: nn.Module | None = None

    def forward(self, batch: dict) -> dict:
        # TODO Step 1.2: implement the masked patch-prediction forward pass.
        raise NotImplementedError("Step 1.2: build VisionJEPA architecture.")

    # -- EMA update (ported from CocktailJEPA.ema_update, cocktail-JEPA repo) -
    @torch.no_grad()
    def ema_update(self, decay: float | None = None) -> None:
        """Move the target encoder toward the context encoder by EMA.

        target <- decay * target + (1 - decay) * context

        Called by the training loop AFTER each optimizer step. No-op when
        use_ema is False (ablation) or when the architecture stub has not
        yet been built. The decay value for the current step is computed by
        loop.py's _ema_momentum() and passed in; falls back to cfg.ema_decay
        when not supplied.
        """
        if not self.cfg.use_ema:
            return
        if self.target_encoder is None or self.context_encoder is None:
            return  # no-op until Step 1.2 wires up the encoders
        d = self.cfg.ema_decay if decay is None else decay
        for tgt, src in zip(self.target_encoder.parameters(),
                            self.context_encoder.parameters()):
            tgt.mul_(d).add_(src, alpha=1.0 - d)
        # buffers (e.g. LayerNorm running stats) are copied outright
        for tgt, src in zip(self.target_encoder.buffers(),
                            self.context_encoder.buffers()):
            tgt.copy_(src)
