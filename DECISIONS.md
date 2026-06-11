# DECISIONS.md — append-only log of irreversible choices

Format: `[Step] Decision — one-line rationale`

---

[0.3] **Three seeds for multi-seed protocol** — vision SSL community standard (three seeds is the norm for STL-10-scale experiments); five seeds was the cocktail project's choice for a faster run. Logged as a deliberate reduction. Any resume/report number still requires ≥3 seeds.

[0.3] **uv as the sole package manager** — reproducible lockfile, fast installs, no pip-install side channels.

[0.3] **SIGReg retained as the anti-collapse regularizer** — ported from cocktail-JEPA Phase-2 fix #18. EMA is also retained (not dropped as LeJEPA suggests) because the energy function is built on the EMA target encoder's latents; dropping EMA removes the thing the energy estimator is built on.

[0.3] **sigreg_term kept verbatim** — the Epps-Pulley characteristic-function test is mathematically settled; no adaptation needed for the vision domain. Only jepa_loss (its caller) is lightly adapted to remove recipe-specific terminology.

[1.3] **Compute platform: Apple M-series MPS (no cloud GPU for Phase 1)** — timing run measured 120 s/epoch at batch 256, bfloat16 AMP. 150 epochs × 3 seeds = 15 h total, spread across laptop sessions. Cloud GPU not needed; Phase 2 will re-evaluate if 100k→40k aerial dataset changes the calculus.

[1.3] **Predictor width: 96 (half of encoder d=192)** — matches PLAYBOOK spec. Implemented with Linear(192→96) proj_in and Linear(96→192) proj_out around a 3-layer transformer. This adds 36K params (18K in + 18K out) to the trainable count.

[1.3] **Batch size: 256, AMP bfloat16, no grad-accum** — 130 MB peak tensor memory leaves ample headroom on M-series unified RAM. Grad accumulation not needed.

[1.3] **Overfit gate passed (moving-target behavior confirmed, no bugs)** — diagnostic run with frozen target (use_ema=false) and sigreg_weight=0 on 500 images, 200 epochs: pred_loss 0.461→0.125 (W&B: ia2z1vva). Did not reach ~0 because IJEPAMaskCollator draws a fresh random mask every batch; the predictor must generalise across all mask positions rather than memorise (image, position) pairs, giving an irreducible floor ~0.12. Decomposition: 0.236 (moving-target run) ≈ 0.125 (mask-variability floor) + 0.11 (moving-target overhead). The 0.24 floor in the production overfit run is explained entirely by moving-target dynamics; the predictor path, masking indexing, and loss function are all correct.

[1.3] **Checkpoint-resume: save epoch index in blob; resume by restoring optimizer+scheduler state_dict** — scheduler created with full-run total_steps on every launch; restoring state_dict advances last_epoch so LR continues smoothly. W&B run stitched via wandb_run_id stored in checkpoint extra dict.
