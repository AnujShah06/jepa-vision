# PROJECT_STATE.md

## Session rules (read at start of every session)

1. One experiment or one component per session. No "while we're at it" scope creep.
2. The LLM proposes; the training run decides. No conclusion enters this file without a W&B link.
3. Any number destined for the resume or report needs ≥3 seeds.
4. Test sets are touched only by the evaluation harness, never during development. Model selection uses validation only.

---

## Current phase / step

**Phase 0 — Foundation**
**Step 0.3 complete** — scaffold, ported components, environment, 15/15 tests green

---

## Last completed run

None yet. (Gate 0 smoke test is next.)

---

## Open decisions

- Compute platform not yet chosen (Step 0.2). Options: Colab Pro, Lightning AI, Kaggle, Purdue cluster.
- Three seeds chosen over five for vision multi-seed protocol (see DECISIONS.md).

---

## Next action

**Gate 0 smoke test** — tiny model (d=64, 2 layers), 100 images, 2 epochs, diagnostics logging to W&B.

Concrete next steps:
1. Build `src/data/` STL-10 loader with block-masking collator (Step 1.1)
2. Build `src/models/vit.py` + `predictor.py`, fill in `VisionJEPA.forward()` (Step 1.2)
3. Write `scripts/smoke_test.py`: tiny config, 100 images, 2 epochs, W&B run
4. Confirm W&B run appears with `loss` + `effective_rank` curves
5. Record W&B link here before declaring Gate 0 passed

---

## Waived gates + justification

None.
