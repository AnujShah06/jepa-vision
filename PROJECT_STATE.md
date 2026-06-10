# PROJECT_STATE.md

## Session rules (read at start of every session)

1. One experiment or one component per session. No "while we're at it" scope creep.
2. The LLM proposes; the training run decides. No conclusion enters this file without a W&B link.
3. Any number destined for the resume or report needs ≥3 seeds.
4. Test sets are touched only by the evaluation harness, never during development. Model selection uses validation only.

---

## Current phase / step

**Phase 1 — I-JEPA-mini on STL-10**
**Step 1.2 complete** — I-JEPA block masking + visual unit test

---

## Last completed run

**Gate 0 re-run with block masking collator**
- W&B run: https://wandb.ai/entropy_chess/jepa-vision/runs/emfmwch0
- Config: same tiny d_model=64, 2 layers, 4 heads; new I-JEPA block collator
- Data: 100 STL-10 unlabeled images, 7 batches/epoch, batch_size=16
- Device: MPS (Apple Silicon)

Results (2 epochs):

| epoch | loss   | pred_loss | sigreg | effective_rank | spread   |
|-------|--------|-----------|--------|----------------|----------|
| 0     | 0.4159 | 0.3600    | 0.5591 | 39.81          | 9.8466   |
| 1     | 0.4013 | 0.3450    | 0.5630 | 37.00          | 10.5424  |

Loss decreasing, effective_rank healthy, plumbing confirmed with real block masks.

Previous Gate 0 run (random-partition collator):
- W&B run: https://wandb.ai/entropy_chess/jepa-vision/runs/95gqj0np

---

## Open decisions

- Compute platform not yet chosen (Step 0.2). STL-10 unlabeled cached in data/.
- Three seeds chosen over five (see DECISIONS.md).
- Block masking currently uses ONE mask per batch (all images share same target/context
  split). Per-image masking (each image independently sampled) is deferred to Step 1.3
  and requires variable-length handling or padding in the model forward pass.
- Context block area lower bound ~75% (instead of nominal 85%) due to integer
  rounding + grid clamping at extreme aspect ratios. Documented in test_masking.py.

---

## Next action

**Phase 1, Step 1.1 / 1.3 — full data pipeline + production training run:**
- Upgrade `src/data/stl10.py`: full 100k unlabeled pretraining loader with
  RandomResizedCrop(96, scale=0.3–1.0) + horizontal flip augmentation
- Probe train/val/test split (carve 1k val from 5k labeled train, quarantine test)
- Timing run (Step 0.2): measure 1 epoch wall time to set the training schedule
- Then: production VisionJEPA config (d=192, 6 layers, 3 heads), 150-epoch reference run
  with W&B collapse dashboard (effective_rank must stay above ~50% of d throughout)

---

## Waived gates + justification

None. Gate 0 passed (original run + re-run with block masking).
