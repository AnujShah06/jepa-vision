# PROJECT_STATE.md

## Session rules (read at start of every session)

1. One experiment or one component per session. No "while we're at it" scope creep.
2. The LLM proposes; the training run decides. No conclusion enters this file without a W&B link.
3. Any number destined for the resume or report needs ≥3 seeds.
4. Test sets are touched only by the evaluation harness, never during development. Model selection uses validation only.

---

## Current phase / step

**Phase 0 — Foundation**
**Gate 0 PASSED** — end-to-end smoke test passing on MPS

---

## Last completed run

**Gate 0 smoke test**
- W&B run: https://wandb.ai/entropy_chess/jepa-vision/runs/95gqj0np
- Config: d_model=64, enc_layers=2, enc_heads=4, pred_layers=2, pred_heads=4, sigreg_weight=0.1
- Data: 100 STL-10 unlabeled images, 7 batches/epoch, batch_size=16
- Device: MPS (Apple Silicon)
- Trainable params: 212,608

Results (2 epochs):

| epoch | loss   | pred_loss | sigreg | effective_rank | spread  |
|-------|--------|-----------|--------|----------------|---------|
| 0     | 0.4269 | 0.3571    | 0.6980 | 38.20          | 9.7952  |
| 1     | 0.3347 | 0.2916    | 0.4313 | 36.80          | 10.5630 |

Loss decreased, effective_rank healthy (38 >> 1), spread positive — plumbing is green.

---

## Open decisions

- Compute platform not yet chosen (Step 0.2). STL-10 unlabeled (~2.6 GB) now cached in data/.
- Three seeds chosen over five for vision multi-seed protocol (see DECISIONS.md).
- Random patch masking used for smoke test; proper block masking (4 target blocks, 85-100% context) is Step 1.1.

---

## Next action

**Phase 1, Step 1.1** — proper data pipeline:
- Block-masking collator (4 target blocks, 15-20% each, aspect 0.75-1.5; context 85-100% minus target patches)
- Visual unit test: render 10 sampled masks to PNG and eyeball
- Download OOD sets (SVHN test, CIFAR-10 test) now while building data code
- Upgrade `src/data/stl10.py`: full 100k unlabeled pretraining loader + probe train/val/test split

---

## Waived gates + justification

None. Gate 0 passed on hardware (MPS), W&B link above.
