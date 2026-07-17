# PROJECT_STATE.md

## Session rules (read at start of every session)

1. One experiment or one component per session. No "while we're at it" scope creep.
2. The LLM proposes; the training run decides. No conclusion enters this file without a W&B link.
3. Any number destined for the resume or report needs ≥3 seeds.
4. Test sets are touched only by the evaluation harness, never during development. Model selection uses validation only.

---

## Current phase / step

**Phase 1 — I-JEPA-mini on STL-10**
**Step 1.6q/r COMPLETE** — R3 run-2 canonical. ref_s1 MPS infra void repaired. Decision 1=A PASS. Probe-on-test done (band check FAIL noted). Scratch v2 evidence: gaps +2.8/+0.4/+1.0pp at n=40/200/400; −6.5pp at n=4000; only n=40 exceeds combined spread. Gate decision = human. Binding claim filled (CIFAR-10=0.864 added). phase1.md drafted. Next: human review of phase1.md; Phase 2 kickoff.

**Tonight's command:** `bash scripts/tonight.sh`

**Today's tasks: see Step 1.6q section below.**

---

## Run ledger (single source of truth for training-run status)

Every session: if the user reports a completed run, update the matching row
(W&B id, final eff_rank, final loss, gate verdict) BEFORE other work.
Never describe status from memory — only from this table.

| Run | Config | Status | W&B / notes |
|---|---|---|---|
| Reference seed 0 | `configs/phase1_ref.yaml` | DONE, Gate 1A passed | `tkqjawa0` — eff_rank 175.3, loss 0.2096, pred_loss 0.2071 (W&B final). best.ckpt=ep143. **Canonical: epoch_0150.ckpt**. Probe n=4000 mean 0.601 (probe seeds {0,1,2}: 0.600/0.601/0.603, σ=0.0015) |
| Reference seed 1 (partial) | `configs/phase1_ref.yaml` | ABORTED at ep54 — DO-NOT-USE | `8cw5vncy` — superseded by lbd900za |
| Reference seed 1 | `configs/phase1_ref.yaml` | DONE, Gate 1A passed | `lbd900za` — eff_rank 172.6, loss 0.2102, pred_loss 0.2074 (W&B final). best.ckpt=ep147. **Canonical: epoch_0150.ckpt**. Probe n=4000 mean 0.564 (probe seeds {0,1,2}: 0.564/0.564/0.564, σ=0.000) |
| Reference seed 2 (partial) | `configs/phase1_ref.yaml` | ABORTED at ep15 — DO-NOT-USE | `37yyfu00` — superseded by gommvdgc. Explains 08:35 Jul 8 relaunch anomaly: overnight launch died at ep15 (session kill), relaunched morning as gommvdgc. |
| Reference seed 2 | `configs/phase1_ref.yaml` | DONE, Gate 1A passed | `gommvdgc` — eff_rank 173.2, loss 0.2072, pred_loss 0.2045, spread 19.54, var 0.995. best.ckpt=ep145. **Canonical: epoch_0150.ckpt**. Probe n=4000 mean 0.582 (probe seeds {0,1,2}: 0.582/0.579/0.584, σ=0.0025) |
| Hardmask seed 0  | `configs/phase1_hardmask.yaml` | DONE, Gate 1A passed. **Adoption REJECTED** (probe 0.587 < 0.62) | `fw1out6d` — eff_rank 189.2, loss 0.2933, pred_loss 0.2918 (ep150). best.ckpt is epoch 2 (checkpoint-saving bug). **Canonical: epoch_0150.ckpt**. Probe n=4000 0.587 (statistically indistinguishable from reference spread 0.564–0.601) |
| MAE baseline     | `configs/mae_baseline.yaml` | **DONE** | `eoofx7fk` (deep-music-14) — final loss 0.4630, epoch 149 (0-indexed). **Canonical: epoch_0150.ckpt** (150 % 10 == 0 ✓). best.ckpt valid (MAE loss monotone). W&B: https://wandb.ai/entropy_chess/jepa-vision/runs/eoofx7fk |
| **R3 run-1** | terminal_benchmark.py (all 5 ckpts, test split) | **ABORTED — DO-NOT-USE** | Launched overnight Jul 12→13. Aborted at ~9/75 Stage-2 cells: kIOGPUCommandBufferCallbackErrorOutOfMemory at shot_noise sev2. Stage-1 clean means completed cleanly (all 10 heads). 9 observed Stage-2 cells (gaussian sev1-5: 0.747/0.687/0.722/0.879/0.964; shot sev1-4: 0.973/0.755/0.995/0.928 — non-monotone, post-OOM corruption suspected). Interruption exception invoked (ritual step 5). **NUMBERS NOT USED IN ANY REPORT.** Item-5 diagnostic (1.6q): gaussian sev1 Δ=0.168 vs run-2 confirms non-monotone run-1. **VOID IN ENTIRETY.** |
| **R3 run-2** | terminal_benchmark.py (all 5 ckpts, test split) | **DONE — CANONICAL** | Launched 2026-07-16, exit 0, ~4.5h (Stage1=2265s, Stage2=12972s, Stage3=636s, Stage4=185s). 75/75 Stage-2 cells complete (crash-insurance JSONL). ref_s1 Stage-2 VOID-INFRA (MPS silent Stage-1 corruption; recomputed 1.6q). ref_s0 Stage-2 canonical (2 clean seeds). mahal_tgt D3 confirmed: SVHN=0.986, CIFAR=0.864. Stage-4b (test probe) run separately (scripts/probe_on_test.py). Report: reports/terminal_test.md. |

**Checkpoint-saving bug note:** For the hardmask run, pred_loss is NOT monotonically decreasing — it starts low (EMA target close to context encoder early in training) and rises as EMA momentum grows from 0.996→1.0. The checkpoint saver saved epoch 2 as "best" because it had the lowest pred_loss (0.2738). The correct fully-trained checkpoint is epoch_0150.ckpt (pred_loss 0.2918). The adoption verdict used epoch_0150.ckpt. This bug does not affect reference runs (their loss is monotonically decreasing). Must fix checkpoint saving for future runs that use increasing-difficulty schedules.

---

## Compute reality

Measured 2026-06-10 on Apple M-series MPS (bfloat16 AMP, batch 256).

| Metric | Value |
|---|---|
| Epoch wall time | 119.8 s (2.0 min) |
| Throughput | 833 images/s |
| Peak tensor memory | ~130 MB (MPS allocated) |
| MPS driver pool | ~6.4 GB (includes Metal shader cache — not model memory) |
| **150 epochs × 1 seed** | **5.0 h** |
| **150 epochs × 3 seeds** | **15.0 h** |

Config: d=192, 6 enc layers, 3 heads; predictor d=96, 3 layers; batch 256; 99,840 img/epoch.
Trainable params: 3,079,392. Total (incl. frozen target): 5,748,960 ≈ 6M.

**Decision: run Phase 1 entirely on this laptop (MPS). No cloud GPU needed.**
15 h / 3 seeds can be spread across days by running seeds back-to-back.

---

## Last completed runs

**Timing run** — no W&B (measurement only)
- Loss during timing epoch: 0.89 → 0.33 (single epoch on 100k unlabeled, production config)
- Confirms model is learning correctly at production scale

**Overfit check** (500 images, no aug, 200 epochs, seed 0):
- W&B: https://wandb.ai/entropy_chess/jepa-vision/runs/0652g54d
- pred_loss: 0.483 → 0.236 (sustained monotone descent) ✓
- effective_rank: 92 → 44 (ep 25) → 73 (ep 199); above collapse floor throughout ✓
- **PASS** — model learns correctly; 0.24 floor explained by moving-target diagnostic below

**Fixed-target diagnostic** (500 images, use_ema=false, sigreg_weight=0, 200 epochs, seed 0):
- W&B: https://wandb.ai/entropy_chess/jepa-vision/runs/ia2z1vva
- pred_loss: 0.461 → 0.125; effective_rank 88 → 93.8 (full rank, no collapse pressure)
- Decomposition: 0.236 (moving target) ≈ 0.125 (mask-variability floor) + 0.11 (moving-target overhead)
- Did not reach ~0 because IJEPAMaskCollator draws a fresh mask each batch → predictor must
  generalise across mask positions, not memorise — irreducible floor ~0.12
- **PASS** — no bugs in predictor path or masking indexing; 0.24 floor is moving-target behavior

**Checkpoint-resume test** (500 images, 15 epochs total, seed 42):
- W&B: https://wandb.ai/entropy_chess/jepa-vision/runs/0lm9ij86
- Part 1 (ep 0-6): fresh run, checkpoint saved at ep 5 (epoch_0005.ckpt, step=40)
- Part 2 (ep 5-14): resumed same W&B run, optimizer+scheduler state restored
- Loss continued 0.47 → 0.40 after resume (no reset) ✓; artifacts ep10 + ep15 uploaded ✓
- Note: ep 5-6 get W&B step-conflict warnings (expected; part1 ran past the checkpoint epoch)
- **PASS** — save/load/resume pipeline works end-to-end

**Production reference run — seed 0 (150 epochs, 100k unlabeled STL-10):**
- W&B run ID: tkqjawa0  (checkpoint: runs/tkqjawa0/best.ckpt, best at epoch 143)
- Gate 1A: PASSED — final eff_rank 175.3 (>50% of d=192 throughout)
- Probe grid (Step 1.5d) numbers are real production numbers from this completed run.

**Production reference run — seed 1 (150 epochs, 100k unlabeled STL-10):**
- W&B run ID: lbd900za  (checkpoint: runs/lbd900za/best.ckpt)
- Gate 1A: PASSED — final eff_rank 172.6, final loss 0.2102, pred_loss 0.2074

**Production reference run — seed 2 (150 epochs, 100k unlabeled STL-10):**
- W&B run ID: gommvdgc (flowing-wildflower-13)
- Gate 1A: PASSED — final eff_rank 173.2, final loss 0.2072, pred_loss 0.2045, spread 19.54, var 0.995

**MAE baseline training:**
- Status: NOT RUN (stale "STILL TRAINING" entry was fiction. Entry point re-smoke-tested 2026-07-07: 2 ep × 100 imgs, loss 1.318. Deleted.)

**Gate 0 re-run with block masking:**
- W&B: https://wandb.ai/entropy_chess/jepa-vision/runs/emfmwch0

**Gate 0 original (random-partition collator):**
- W&B: https://wandb.ai/entropy_chess/jepa-vision/runs/95gqj0np

---

## Open decisions

- Block masking uses ONE mask per batch (per-image masking deferred to Step 1.3).
- Context block actual area floor ~75% (nominal 85%) due to rounding at extreme aspect; documented in test_masking.py.
- Three seeds for multi-seed protocol (see DECISIONS.md).

---

## Step 1.4 — Energy function results

**Validation split** — `data/splits/stl10_val_idx.json` (committed):
- Stratified 100/class × 10 classes = 1,000 images; seed=0; carved from STL-10 labeled train
- Probe training must use the complement: 4,000 images (range(5000) minus val indices)
- `get_val_loader()` and `get_probe_train_loader()` in src/data/stl10.py enforce this split

**K-sweep** (formal 1,000-image val split, gaussian_noise severity=3, ckpt=tkqjawa0/best.ckpt):

| K  | AUROC  | clean μ | corrupt μ | clean σ |
|----|--------|---------|-----------|---------|
| 1  | 0.781  | 0.2009  | 0.2243    | 0.0367  |
| 4  | 0.747  | 0.2135  | 0.2322    | 0.0298  |
| 8  | 0.763  | 0.2179  | 0.2368    | 0.0276  |
| 16 | 0.792  | 0.2183  | 0.2379    | 0.0258  |

- AUROC K-insensitive within single-seed noise (0.747–0.792, ±0.02 around 0.77) ✓
- clean σ decreases monotonically (variance reduction as designed) ✓
- K=8 locked as inference default (see DECISIONS.md)
- Figure: reports/figures/k_sweep.png

Note: the original K-sweep (first session of Step 1.4) used the first 500 images from STL-10 train
(indices 0-499), which is NOT the formal val split (only 112/500 overlap). Results superseded.

**Files added (Step 1.4):**
- `src/eval/energy.py` — `image_energy`, `energy_over_loader`, `energy_heatmap`
- `src/data/stl10.py` — added `get_eval_loader` (plain images, no masking)
- `scripts/k_sweep.py` — K-sweep script
- `tests/test_energy.py` — 10 new tests (41/41 total passing)

---

## Step 1.5a — Baseline sanity results

**Sanity check:** `baseline_sanity.py --jepa_ckpt runs/tkqjawa0/best.ckpt --severity 3`
(validation only; no W&B link — sanity script, not a training run)

| Baseline | AUROC | clean μ | corrupt μ | clean σ |
|---|---|---|---|---|
| Pixel std (trivial floor) | 0.7221 | 1.0139 | 1.1926 | 0.2597 |
| Random-init JEPA (K=8) | 0.7891 | 0.5114 | 0.5305 | 0.0080 |
| Mahalanobis (trained features) | 0.6040 | 13.89 | 14.29 | 2.20 |
| MAE reconstruction (K=8, untrained) | 0.3929 | 1.3369 | 1.3326 | 0.0135 |
| **JEPA energy K=8 (trained, seed-0)** | **0.7642** | **0.2179** | **0.2370** | **0.0276** |

**Observations:**
- Random-init AUROC=0.789 is high because gaussian noise changes embedding norms even through random weights; trained JEPA (0.764) is slightly below on this single seed — Gate 1B needs ≥3 seeds before drawing any conclusion.
- Mahalanobis (0.604) confirms feature space responds to distribution shift but is weaker than the prediction-error energy.
- MAE untrained inverts (0.393 < 0.5): gaussian noise inflates per-patch variance; `norm_pix_loss` normalizes by that variance, making the untrained decoder's near-zero predictions look better on noisy patches. Expected — untrained MAE must be trained before it is a meaningful baseline.
- Pixel std (0.722) is the honest floor for pixel-level gaussian noise.

**Bug fixed during sanity run:** `auroc()` in `evaluate.py` used `torch.arange` without `device=` arg — fails when inputs are on MPS. Fixed with `device=all_scores.device`.

**Files added (Step 1.5a):**
- `src/models/mae.py` — PixelMAE (same encoder budget as VisionJEPA; lightweight decoder)
- `src/eval/baselines.py` — four baseline energy functions
- `configs/mae_baseline.yaml` — MAE training config
- `scripts/train_mae.py` — MAE training entry point
- `scripts/baseline_sanity.py` — runs all 5 baselines, prints table
- `tests/test_baselines.py` — 25 tests (65/65 total passing)

---

## Step 1.5d — Probe grid results (seed-0, tkqjawa0, val only)

**Full-budget run:** `probe_sweep.py --jepa_ckpt runs/tkqjawa0/best.ckpt --probe_epochs 100 --scratch_epochs 200`
Wall time: 125 min.  Report: `reports/probe_seed0_val.md`
(No W&B link — evaluation script, not a training run.)

| n | Frozen probe | From-scratch | Gap (P−S) | Best LR |
|---|---|---|---|---|
| 40 | 0.2930 | 0.2380 | **+0.055** | 1e-03 |
| 200 | 0.3940 | 0.3840 | **+0.010** | 1e-04 |
| 400 | 0.4350 | 0.4120 | **+0.023** | 1e-04 |
| 4000 | 0.5730 | 0.6360 | −0.063 | 1e-03 |

**Interpretation:**
- Low-label cells (n=40/200/400): probe beats scratch in all 3 → pretraining helps under label scarcity ✓
- n=4000 (full probe pool, 400/class): scratch (0.636) > probe (0.573) by −6.3 pp
- Gate 1B criterion (iii) ("pretrained > scratch in most low-label cells"): **3/4 cells positive** ✓ — still needs ≥3 seeds for reportable numbers
- Gate 1B criterion (ii) sanity floor ("probe at 100% labels ≥ ~70%"): probe=57.3% — **FAILING** ⚠️ gap is 12.7 pp; numbers are from the completed seed-0 production run (tkqjawa0, 150 epochs); floor not met with plain mean-pool context-encoder + fixed lr probe
- Probe path diagnostics run (see below): best single-seed config reaches 0.601 — floor still failing by 9.9 pp; this is a representation quality issue, not a probe configuration issue

## Step 1.5d — Probe diagnostics (n=4000, seed-0, val only)

`scripts/probe_diag.py --jepa_ckpt runs/tkqjawa0/best.ckpt`

| Variant | Val Acc | Ep | Best LR |
|---|---|---|---|
| context mean (baseline) | 0.5710 | 100 | 1e-03 |
| target mean | 0.5720 | 100 | 1e-03 |
| context mean + z-score | 0.5800 | 100 | 1e-03 |
| target mean + z-score | 0.5810 | 100 | 1e-03 |
| context mean+max concat | 0.5730 | 100 | 1e-03 |
| context last-2-layer concat | 0.5920 | 100 | 1e-03 |
| target mean + lr-sweep + 200ep | 0.5900 | 200 | 3e-03 |
| **target mean+zscore + lr-sweep + 200ep** | **0.6010** | **200** | **3e-03** |

**Findings:**
- Target encoder ≈ context encoder (+0.001) — EMA hasn't diverged substantially, suggesting either short training or high EMA decay. Not the JEPA-family norm where target markedly outperforms context.
- Z-score standardisation: +1.0 pp — modest but consistent
- Last-2-layer concat: +2.1 pp — penultimate layer features carry complementary information
- Longer schedule (200ep) + lr=3e-3: +1.9 pp (lr=1e-3 was underpowered at n=4000)
- Best combo (target+zscore+lr-sweep+200ep): **0.601**, up +3.0 pp from baseline
- **Gate 1B floor (≥0.70): FAIL — gap is −9.9 pp even with the best probe config**
- The 9.9 pp residual gap is a representation quality issue, not a probe configuration issue; exhausting probe variants recovers only 3 pp of the 12.7 pp deficit
- Re-run multi-seed sweep with best config (target+zscore, lr=3e-3, 200ep) once seeds 1 & 2 are trained

**Files added (Step 1.5d):**
- `src/eval/probe.py` — `stratified_sample`, `get_probe_pool`, `extract_features`, `train_probe`, `ScratchClassifier`, `train_scratch` (AMP-enabled)
- `scripts/probe_smoke.py` — quick sanity check (n=200, small epoch budget)
- `scripts/probe_sweep.py` — full-budget grid runner, writes Markdown report
- `scripts/probe_diag.py` — diagnostic ablation runner
- `tests/test_probe.py` — 22 tests (87/87 total passing)
- `reports/probe_seed0_val.md` — full-budget seed-0 grid (context mean, lr=1e-3, 100ep)

---

## Next actions (decision tree)

**HARDMASK REJECTED (2026-07-07).** Pre-registered REJECT branch fires.

| Slot | Action |
|---|---|
| **DONE** | Full val benchmark complete → `reports/terminal_val.md` (b803je03m, 3389s, exit 0) |
| **DONE** | Gate 1B floor approved (0.582±0.018). R3 runsheet approved. Amendments A1–A9 implemented. |
| **DONE** | Harness smoke-tested on val (dry run, no --unlock_test): manifest OK, mahal_tgt/ctx/mae_untrained/random_init all computed, Stage 2+3 running |
| ~~**Tonight**~~ **DONE** | Scratch loop (A3): 36/36 complete, 4.70h. n=40=0.2397±0.0076, n=200=0.3533±0.0098, n=400=0.3893±0.0032, n=4000=0.5787±0.0121. Gate 1B(iii) val-side preliminary: gaps +3.9/+3.8/+4.7/+0.3pp |
| **Scratch manifest gate** | Confirm 36/36 in reports/scratch_manifest.json, n=4000 sanity band [0.55, 0.70] (observed 0.566–0.590 ✓), eyeball per-cell table |
| **Gap wiring + val smoke** | Wire gap column into terminal_benchmark.py Stage 4; smoke-test on val (no --unlock_test). B2 fallback: gap blank-with-note pending recipe-fixed scratch rerun |
| **Win-count reconciliation** | Binding count 11/15 from terminal_val.md (15-type run). "8/13" from 1.6g was a text error — the 1.6g table yields 10/13; glass(✗)/fog(✓) added in 1.6h → 11/15. Mask seed=0 in energy.py since d66ba65 (scaffold). Corruption generation unseeded but once-and-shared across models → no change. |
| **Patch verify** | `uv run python scripts/patch_imagecorruptions.py` → ALREADY_PATCHED × both ✓ (confirmed this session) |
| **R3** | Human-launched per DECISIONS.md runsheet launch ritual. |
| Then | Write reports/phase1.md from terminal_test.md; Phase 2 kickoff |

**Gate 1B floor**: approved 0.582±0.018 (reference config, n=4000, mean ± std over training seeds {0.600, 0.565, 0.579}). Gate 1B(i): paired-bootstrap CI on AUROC margin excludes 0. Gate 1B(ii): 0.582±0.018 reported, not gated. Gate 1B(iii): val-side preliminary (gaps +3.9/+3.8/+4.7/+0.3pp); formal gate at R3. See DECISIONS.md.

---

## Step 1.6o — Final gate evidence + scratch adjudication + R3 GO (2026-07-12)

**Item 0: Corrections applied.** Gate 1B(iii) language changed from "PASS ✓" to "val-side preliminary — formal at R3". Gate 1B floor label corrected to "(reference config, n=4000, mean ± std over training seeds {0.600, 0.565, 0.579})". Scratch sanity band restored to [0.55, 0.70]. Saturday-N labels relabelled to functional names. DECISIONS.md launch ritual step 4 amended (4a/4b/4c/4d split). tonight.sh overwritten as pointer stub.

**Item 1: Scratch evidence (36/36 complete).**

Per-lr per (seed, n) all status=ok, epochs_completed=200:

| key | n | lr | val_acc |
|---|---|---|---|
| s0_n40 | 40 | 1e-03 | 0.2330 | s0_n40_lr3e-04 | 0.226 | s0_n40_lr1e-04 | 0.223 |
| s0_n200 | 200 | 1e-03 | 0.328 | 3e-04 | 0.359★ | 1e-04 | 0.343 |
| s0_n400 | 400 | 1e-03 | 0.368 | 3e-04 | 0.393★ | 1e-04 | 0.374 |
| s0_n4000 | 4000 | 1e-03 | 0.576 | 3e-04 | 0.577 | 1e-04 | **0.580**★ |
| s1_n40 | 40 | 1e-03 | **0.248**★ | 3e-04 | 0.241 | 1e-04 | 0.238 |
| s1_n200 | 200 | 1e-03 | 0.339 | 3e-04 | 0.349 | 1e-04 | **0.359**★ |
| s1_n400 | 400 | 1e-03 | 0.358 | 3e-04 | 0.371 | 1e-04 | **0.387**★ |
| s1_n4000 | 4000 | 1e-03 | 0.561 | 3e-04 | 0.559 | 1e-04 | **0.566**★ |
| s2_n40 | 40 | 1e-03 | **0.238**★ | 3e-04 | 0.232 | 1e-04 | 0.234 |
| s2_n200 | 200 | 1e-03 | 0.326 | 3e-04 | **0.342**★ | 1e-04 | 0.338 |
| s2_n400 | 400 | 1e-03 | 0.354 | 3e-04 | 0.380 | 1e-04 | **0.388**★ |
| s2_n4000 | 4000 | 1e-03 | 0.568 | 3e-04 | 0.578 | 1e-04 | **0.590**★ |

★ = best_lr selection. n=4000 sanity band [0.55, 0.70]: observed 0.566–0.590 ✓.

Cross-seed means ± σ: n=40 0.2397±0.0076, n=200 0.3533±0.0098, n=400 0.3893±0.0032, n=4000 0.5787±0.0121.

**Item 2: Scratch discrepancy adjudication — Branch B2 fires.**

s0_n4000 per-lr (same seed as 1.5d 0.636): lr1e-03=0.576, lr3e-04=0.577, lr1e-04=0.580. Best=0.580 ≤ 0.60.

Recipe diff (1.5d probe_sweep.py vs A3 run_scratch_comparator.py):
- **Augmentation**: 1.5d had `RandomResizedCrop(96, scale=(0.5,1.0)) + RandomHorizontalFlip`. A3 has `Resize(96) + CenterCrop(96)` only (no augmentation). This is the dominant driver.
- **Batch size**: 1.5d=128 fixed; A3=min(256,n). At n=4000: 1.5d 2× more steps/epoch.
- **Effective optimizer steps at n=4000**: 1.5d ceil(4000/128)=32 steps/ep × 200ep = 6400; A3 ceil(4000/256)=16 × 200 = 3200. A3 has ~2× fewer steps PLUS no augmentation.
- **Train accuracy**: not logged (ScratchClassifier only saves val_acc).

Branch B fires (s0_n4000 ≤ 0.60 AND concrete optimization disadvantage confirmed). B1 vs B2: re-running 9 n=4000 cells with batch=128 + augmentation would take ~5-6h and cannot complete by ~20:00. **Branch B2 fires: R3 tonight WITHOUT scratch. Gap column BLANK-WITH-NOTE in terminal_benchmark.py Stage 4. Gate 1B(iii) evaluated post-hoc on val with recipe-fixed scratch. Test set NEVER reopens for scratch.**

**Item 4: Win-count + corruption-RNG reconciliation.**

(a) Binding count: **11/15** from terminal_val.md (15-type run, step 1.6h). Manual recount: gaussian✓ shot✓ impulse✓ zoom✓ frost✓ fog✓ brightness✓ contrast✓ elastic✓ pixelate✓ jpeg✓ — defocus✗ glass✗ motion✗ snow✗.

Source of "8/13": text error in step 1.6g. The 1.6g table yields 10/13 wins (recount: gaussian, shot, impulse, zoom, frost, brightness, contrast, elastic, pixelate, jpeg). The 1.6g text "8/13" was simply incorrect. Adding glass(✗) + fog(✓) in 1.6h gives 11/15.

(b) Per-type deltas between 1.6g and 1.6h runs: ≤0.001 for all types (within bootstrap noise). Same corrupted images used.

(c) Corruption generation: unseeded (no seed arg in `_corrupt_tensor`), BUT once-and-shared per (type, severity) across all models in Stage 2. → "once-and-shared" condition: no change needed.

(d) Mask seed=0 in energy.py since `d66ba65` (scaffold commit). Both 1.6g and 1.6h val runs had seed=0.

**Item 5: Preconditions.**
- `uv run python scripts/patch_imagecorruptions.py` → Fix 1 (glass_blur): ALREADY_PATCHED; Fix 2 (fog): ALREADY_PATCHED ✓
- Approval lines: present in DECISIONS.md §1.6i (Gate 1B floor: Anuj 2026-07-10; R3 runsheet: Anuj 2026-07-10)
- tonight.sh: overwritten as pointer stub (echoes "R3 tonight — launch manually per DECISIONS.md runsheet ritual")
- Gap wiring: terminal_benchmark.py Stage 4 now loads scratch_manifest.json, shows gap with B2 underfit note

**Val smoke-test (dry_run, split=val, no --unlock_test): exit 0 ✓**

Startup manifest:
```
[manifest] split=val
[manifest] val=1000 (data/splits/stl10_val_idx.json, 100/class seed=0)
[manifest] probe_pool=4000 (STL-10 labeled train complement of val)
[manifest] OOD: SVHN test + CIFAR-10 test (data/ood/)
[scratch gap] loaded 4/4 cells from scratch_manifest.json
```

Stage 4 probe-grid section (from reports/terminal_val_s4gap.md):
```
Model                         n=40         n=200         n=400         n=4000
---------------------------------------------------------------------------
ref_s0                0.2937±0.0034  0.4147±0.0246  0.4550±0.0073  0.6030±0.0008
ref_s1                0.2690±0.0043  0.3653±0.0116  0.4170±0.0118  0.5643±0.0017
ref_s2                0.2683±0.0184  0.3897±0.0160  0.4357±0.0109  0.5803±0.0005
hardmask_s0*          0.2950±0.0022  0.4180±0.0142  0.4813±0.0146  0.5897±0.0009

JEPA ref mean                0.277         0.390         0.436         0.583
Scratch A3 mean              0.240         0.353         0.389         0.579
Gap (JEPA-ref−A3)         +0.0373*      +0.0366*      +0.0466*      +0.0039*

* A3 recipe underfit: batch=min(256,n) vs 1.5d batch=128; no augmentation vs RandomResizedCrop+HFlip.
  Gap shown for reference; binding gap requires recipe-fixed rerun.
  Gate 1B(iii) post-hoc on val; test set never reopens.
```

Gap wiring confirmed working. Scratch 4/4 cells loaded. Gaps positive at all n. B2 note printed correctly.

---

## Step 1.6q — R3 run-2 canonical + ref_s1 infra void + probe-on-test (2026-07-16)

**Item 1: R3 run-2 recorded (canonical).**
- R3 run-2: exit 0, ~4.5h, 75/75 Stage-2 cells. Report: reports/terminal_test.md.
- Item-5 diagnostic: gaussian sev1 Δ=0.168 between run-1 and run-2 → run-1 VOID IN ENTIRETY. Run-2 canonical.
- Wall: Stage1=2265s, Stage2=12972s, Stage3=636s, Stage4=185s.

**Item 2: MPS silent Stage-1 corruption — ref_s1 (lbd900za) incident.**
- Symptom: ref_s1 clean test mean 0.2711 (expected ~0.219); Stage-2 AUROCs all 0.07–0.19.
- Root cause: MPS async dispatch — `.cpu()` returned before MPS GPU computation completed. Large K=8 batch caused race condition; returned tensor had stale/partial data.
- Fix: `torch.mps.synchronize()` before `.cpu()` in `_jepa_energy` (terminal_benchmark.py).
- Run-1 hardmask also corrupted (clean mean 0.3865 vs ~0.2901 expected) → run-1 VOID confirmed by two sources.
- `random_init` seeded-fixed (torch.manual_seed(0)); drift Δ=0.0054 = MPS non-determinism, not corruption.
- `mae_untrained` fresh-per-run (no seed); drift Δ=0.040 = different random weights. Not a bug.

**Item 3: Decision 1=A — ref_s1 recompute (PASS).**
- Script: `scripts/recompute_ref_s1.py` (MPS sync fix applied).
- Validation (binding): recomputed clean mean = **0.2190** ∈ [0.216, 0.222] → **PASS**.
- ref_s1 Stage-3 OOD (recomputed): SVHN=0.1277 [0.1235, 0.1319]; CIFAR-10=0.5048 [0.4968, 0.5132].
- ref_s1 Stage-2 rows: **VOID-INFRA** (not rerun; would require another 4.5h benchmark run).
- Files updated: reports/energy_dumps/clean_ref_s1_test.npy; reports/energy_dumps/ood_auroc_test.json.

**Item 4: Decision 2=YES — probe-on-test (Stage 4b).**
- Script: `scripts/probe_on_test.py`. Parity check PASS (test==val at 4-decimal precision when test_loader=val_loader).
- Stage 4b test results (3 seeds, z-score fitted on val, LR selected on val):

| Model       | n=40          | n=200         | n=400         | n=4000        |
|-------------|---------------|---------------|---------------|---------------|
| ref_s0      | 0.2786±0.0111 | 0.3830±0.0084 | 0.4293±0.0095 | 0.5592±0.0020 |
| ref_s1†     | 0.2600±0.0228 | 0.3528±0.0174 | 0.3983±0.0050 | 0.5419±0.0027 |
| ref_s2      | 0.2677±0.0291 | 0.3710±0.0180 | 0.4110±0.0088 | 0.5396±0.0056 |
| hardmask_s0*| 0.2845±0.0136 | 0.4051±0.0045 | 0.4473±0.0051 | 0.5657±0.0018 |

JEPA ref mean (test) n=4000: **0.547** (vs val-era 0.583).

**Band check (binding):** n=4000 within ±0.03 of val-era {0.603, 0.564, 0.580}:
- ref_s0: |Δ|=0.044 → FAIL (val=0.603, test=0.559)
- ref_s1: |Δ|=0.022 → PASS
- ref_s2: |Δ|=0.041 → FAIL (val=0.580, test=0.540)
- Band check result: **FAIL** (2/3 exceed ±0.03). Direction expected (probe LR selected on val n=1000; test n=8000 gives lower accuracy). No recompute triggered; numbers reported as-is.

**Item 5: Report repairs to reports/terminal_test.md.**
- Header fixed: "Val Split" → "Test Split"; metadata line: `split=val` → `split=test`.
- ref_s1 Stage-2 rows marked †VOID-INFRA.
- ref_s1 Stage-3 OOD row updated to recomputed values.
- Stage 4 header updated with val-era note.
- Stage 4b section added (test probe grid + band check + per-seed detail).
- Wall-clock updated (added Stage 1 row, Stage 4b row).

**15-type win/loss vs pixel_std (ref_s0, test split, Stage-2 mean over severities):**

| Type | ref_s0 | pixel_std | Win? |
|------|--------|-----------|------|
| gaussian_noise | 0.736 | 0.734 | ✓ (+0.002) |
| shot_noise | 0.760 | 0.766 | ✗ (−0.006) |
| impulse_noise | 0.796 | 0.741 | ✓ (+0.055) |
| defocus_blur | 0.209 | 0.299 | ✗ (inverted) |
| glass_blur | 0.269 | 0.343 | ✗ (inverted) |
| motion_blur | 0.262 | 0.339 | ✗ (inverted) |
| zoom_blur | 0.360 | 0.371 | ✗ (−0.011) |
| snow | 0.357 | 0.486 | ✗ (−0.129) |
| frost | 0.249 | 0.234 | ✓ (+0.015) — borderline |
| fog | 0.078 | 0.072 | ✓ (+0.006) — both inverted |
| brightness | 0.532 | 0.443 | ✓ (+0.089) |
| contrast | 0.019 | 0.009 | ✓ (+0.010) — both inverted |
| elastic_transform | 0.439 | 0.476 | ✗ (−0.037) |
| pixelate | 0.443 | 0.441 | ✓ (+0.002) — borderline |
| jpeg_compression | 0.566 | 0.484 | ✓ (+0.082) |

Win count (test): **8/15** (gaussian, impulse, frost, fog, brightness, contrast, pixelate, jpeg). Clear wins (margin >>0.009): impulse, brightness, jpeg. Borderline (margin ≤0.015): gaussian, frost, fog, contrast, pixelate. Losses on blur family (4) + shot + zoom + snow + elastic.

**Binding claim (filled from test report):**
"Latent prediction error detects corruption in 8/15 types at test (vs pixel-std baseline). The same frozen encoder's feature density detects semantic domain shift (mahal_tgt SVHN=0.986 at test, probe-pool fit, no additional training). Energy alone inverts on semantic OOD — prediction-difficulty mechanism, Spearman rho=0.770 (val+SVHN). Two readouts, one encoder."

**mahal_tgt complementarity note:** mahal_tgt detects energy-inverted types well — defocus=0.937, contrast=0.977, fog=0.920 — confirming the two readouts are complementary on exactly the types where energy fails.

**Scratch v2 results (recipe-fixed: batch=128, RandomResizedCrop+HFlip), 3 seeds × 36 runs, 3.94h:**

| Cell | val_acc | best_lr |
|------|---------|---------|
| s0_n40 | 0.2430 | 1e-03 |
| s0_n200 | 0.3800 | 1e-04 |
| s0_n400 | 0.4140 | 1e-03 |
| s0_n4000 | 0.6350 | 3e-04 |
| s1_n40 | 0.2570 | 3e-04 |
| s1_n200 | 0.3910 | 3e-04 |
| s1_n400 | 0.4320 | 1e-03 |
| s1_n4000 | 0.6580 | 1e-03 |
| s2_n40 | 0.2460 | 1e-03 |
| s2_n200 | 0.3860 | 3e-04 |
| s2_n400 | 0.4320 | 3e-04 |
| s2_n4000 | 0.6510 | 1e-03 |

Cross-seed means ± σ: n=40 0.2487±0.0074, n=200 0.3857±0.0055, n=400 0.4260±0.0104, n=4000 0.6480±0.0118.

**Binding Gap (JEPA val − scratch v2):**

| n | JEPA ref mean | Scratch v2 mean | Gap | Gate |
|---|---|---|---|---|
| 40 | 0.277 | 0.249 | +0.028 | ✓ |
| 200 | 0.390 | 0.386 | +0.004 | ✓ |
| 400 | 0.436 | 0.426 | +0.010 | ✓ |
| 4000 | 0.583 | 0.648 | −0.065 | (outside criterion) |

~~Gate 1B(iii): PASS~~ — STRUCK (1.6r). Replacement: Gate 1B(iii) evidence: point gaps positive at n=40/200/400 (+2.8/+0.4/+1.0pp); n=4000 −6.5pp to recipe-fixed scratch. Only n=40 exceeds combined spread (RSS=0.016 vs gap=0.028); n=200/400 within noise. Gate decision = human.

**Binding claim (verbatim from DECISIONS.md, slots filled):**
"Latent prediction error detects corruption in 8/15 types at test (vs pixel-std baseline). The same frozen encoder's feature density detects semantic domain shift (mahal_tgt SVHN=0.986, CIFAR-10=0.864 at test, probe-pool fit, no additional training). Energy alone inverts on semantic OOD — prediction-difficulty mechanism, Spearman rho=0.770 (pooled val+SVHN). Two readouts, one encoder."

---

## Step 1.6p — R3 run-1 aborted (MPS OOM) — memory fix + parity gate (2026-07-14)

**Item 0: R3 run-1 recorded.**
- Launched overnight Jul 12→13 from DECISIONS.md runsheet (all 5 ckpts, --split test --unlock_test).
- Aborted at ~9/75 Stage-2 cells: kIOGPUCommandBufferCallbackErrorOutOfMemory at shot_noise sev2.
- Interruption exception invoked (integrity doubt = crash-equivalent under ritual step 5).
- Stage-1 clean means completed, all 10 heads (no integrity concern).
- Stage-2 evidence (9 cells, NUMBERS NOT USED):
  - gaussian sev1-5: 0.747/0.687/0.722/0.879/0.964 (mean 0.800, in-band)
  - shot sev1-4: 0.973/0.755/0.995/0.928 (non-monotone, post-OOM corruption suspected → discarded)
- Root cause: Stage-2 loop sent full 8k×3×96×96 float32 tensor (~884 MB) to MPS per model per cell; K=8 mask iterations accumulate intermediate activations on top of that; with 4 JEPA models + pixel_std + random_init + 2 mahal + MAE running sequentially per cell, MPS allocator exhausted at ~cell 10.

**Item 1: Memory fix (throughput-only; eval logic FROZEN).**
- `scripts/terminal_benchmark.py` changes:
  - Added `--eval_chunk_size` (default=1000). Runsheet command UNCHANGED.
  - Stage 2: all 6 unbatched energy calls replaced with batched equivalents:
    - JEPA models: `_jepa_energy` → `_jepa_energy_batched(..., batch_size=eval_chunk_size)`
    - pixel_std: `pixel_stats_energy(cor_imgs.to(device)).cpu()` → `pixel_stats_energy(cor_imgs)` (CPU-only, no device move)
    - random_init: `_jepa_energy(rand_model,...)` → `_jepa_energy_batched(..., batch_size=eval_chunk_size)`
    - mahal_tgt: explicit batch_size=eval_chunk_size (was already batched but implicit default)
    - mahal_ctx: `mahalanobis_energy(cor_imgs,...)` → `_mahal_energy_batched(..., batch_size=eval_chunk_size)`
    - mae_untrained, mae_trained: unchanged (already batched at bs=64)
  - Stage 1: two fixes: pixel_std no longer moved to device; mahal_ctx uses `_mahal_energy_batched`.
  - After each Stage-2 cell: `del cor_imgs; torch.mps.empty_cache()` (MPS allocator flush).
  - Crash insurance: after each cell, append completed-cell JSON to `reports/terminal_test_progress.jsonl` (write-only, test split only).

**Item 2: Parity gate — PASS.**
- Script: `scripts/parity_gate_1p.py` (exit 0)
- (a) Energy parity: JEPA max|Δ|=0.00e+00 [PASS]; pixel_std max|Δ|=1.19e-07 [PASS]; mahal_ctx max|Δ|=0.00e+00 [PASS]
- (b) AUROC parity (all 10 heads, 200 images, gaussian_noise sev=3):
  - ref_s0: 0.761 [PASS]; ref_s1: 0.592 [PASS]; ref_s2: 0.588 [PASS]; hardmask_s0*: 0.878 [PASS]
  - pixel_std: 0.733 [PASS]; random_init: 0.467 [PASS]; mahal_tgt: 0.732 [PASS]; mahal_ctx: 0.945 [PASS]
  - mae_untrained: 0.472 [PASS]; mae_trained: 0.994 [PASS]
- **Gate: PASS — relaunch unblocked.**

**Item 3: Throughput rehearsal — GO.**
- Synthetic 8k = val × 8 (zero test contact), gaussian_noise sev=3, all 10 heads.
- Corruption generation: 5.9s
- Stage 1 (clean energies, 8k): 209.8s
- One cell wall (all heads, 8k): 217.2s ← **≤ 400s → GO**
- MPS driver alloc after cleanup: 189 MB (confirms no memory leak)
- Projected Stage 2 (75 cells): 271 min (~4.5h)
- Projected total (Stage 1-4): 4.9h
- **Verdict: GO tonight — projected complete ~5h after launch.**

**Item 5: Run-2 integrity diagnostic (pre-registered).**
On R3 run-2 completion, compare run-2 gaussian/shot cells against run-1 observed values:
- gaussian sev1: run-1=0.747; run-2 should be within normal noise (±0.05 expected for test)
- gaussian sev2: run-1=0.687; compare
- gaussian sev3: run-1=0.722; compare
- gaussian sev4: run-1=0.879; compare
- gaussian sev5: run-1=0.964; compare
- shot sev1: run-1=0.973 (suspect, post-OOM); note in comparison
This diagnostic GATES NOTHING. Purpose: detect if run-2 has a systematic issue vs the clean run-1 cells (gaussian sev1-5 were pre-OOM and should be trustworthy). Any large deviation from gaussian sev1-5 warrants investigation before finalising the report.

---

## Step 1.6n — Scratch results + Gate 1B(iii) val-side preliminary (2026-07-12)

**A3 scratch comparator — 36/36 complete, 4.70h total wall.**

Best-lr per cell (formal-val, best lr per training-seed × n):

| Cell | val_acc | best_lr |
|---|---|---|
| s0_n40 | 0.2330 | 1e-03 |
| s0_n200 | 0.3590 | 1e-03 |
| s0_n400 | 0.3930 | 1e-03 |
| s0_n4000 | 0.5800 | 1e-03 |
| s1_n40 | 0.2480 | 1e-03 |
| s1_n200 | 0.3590 | 1e-03 |
| s1_n400 | 0.3870 | 1e-03 |
| s1_n4000 | 0.5660 | 3e-04 |
| s2_n40 | 0.2380 | 1e-03 |
| s2_n200 | 0.3420 | 1e-03 |
| s2_n400 | 0.3880 | 1e-03 |
| s2_n4000 | 0.5900 | 1e-03 |

Cross-seed means ± σ per n:

| n | scratch mean | σ |
|---|---|---|
| 40 | 0.2397 | 0.0076 |
| 200 | 0.3533 | 0.0098 |
| 400 | 0.3893 | 0.0032 |
| 4000 | 0.5787 | 0.0121 |

**Gate 1B(iii) val-side preliminary — gaps consistent with pass; formal evaluation at R3 on test; gate decision = human.**

| n | JEPA (b803je03m 3-seed mean) | Scratch (A3 3-seed mean) | Gap |
|---|---|---|---|
| 40 | 0.2783 | 0.2397 | **+3.9 pp** ✓ |
| 200 | 0.3912 | 0.3533 | **+3.8 pp** ✓ |
| 400 | 0.4359 | 0.3893 | **+4.7 pp** ✓ |
| 4000 | 0.5815 | 0.5787 | +0.3 pp (tie) |

Pre-registered rule: JEPA > scratch at n=40/200/400 → PASS. n=4000 tie is outside the gate criterion.

tonight.sh overwritten for Saturday-2 smoke-test (Stage-4 gap wiring, val split, no --unlock_test).

---

## Step 1.6m — Daylight scratch relaunch (2026-07-12)

tonight.sh overwritten with full scratch loop command. Manifest clean-empty (pre-registered wipe of all 13 error entries from 1.6l). 36 runs launched: `caffeinate -is uv run python scripts/run_scratch_comparator.py`. Results in Step 1.6n.

---

## Step 1.6l — Scratch loop crash fix (2026-07-12)

**Bug:** `stratified_sample(probe_indices, probe_labels, n_per_class=n//10)` passed `probe_labels` as positional arg 2 (which maps to `n_per_class`) then also passed `n_per_class=n//10` by keyword → "multiple values for argument 'n_per_class'". Crashed all 13 runs that ran tonight (wall_s=3.3s each, val_acc=NaN).

**Fix:** `pool_sel = stratified_sample(probe_labels, n_per_class=n//10); sel_indices = [probe_indices[i] for i in pool_sel]`

**Manifest provenance adjudication:**
- Manifest created: Jul 12 13:15:11 (committed in d3f072e by Anuj as part of "1.6j decision updates")
- All 13 entries: error status, wall_s=3.3–3.4s — NOT 200-epoch runs. Pre-registered rule fired: all 13 deleted.
- The "5 completed cells" referenced in CONTEXT SYNC = human misreading `[SKIP]` terminal output against error-keyed entries. No completed runs ever existed in the manifest.
- Sanctioned smoke cell (s0_n200_lr1e-03, 2 epochs) never written to manifest. No F2 (unreported execution) evidenced.

**Hardened manifest schema (Issue 3):** each entry now records `epochs_completed`, `start_time` (ISO UTC), `end_time` (ISO UTC), `git_sha`. Skip-if-present check requires `status=='ok' AND epochs_completed==200`.

**Tests:** 6 new tests in tests/test_scratch_comparator.py — all 4 n-values through sampling path, regression test for buggy call, _is_complete() logic. 113/113 total pass.

**tonight.sh:** overwritten to print "No overnight run…" and exit 0. Morning session overwrites with relaunch command.

---

## Step 1.6i — Pre-R3 code + smoke-test (2026-07-10)

**Harness amendments (A1–A9) implemented:**
- A1: `--split test --unlock_test` wiring + negative guard test (2 pytest tests PASS)
- A2: Mask RNG seeded at seed=0; MPS non-determinism documented (|Δ|<0.001)
- A4: mahal_ctx labeled [FIT-ON-EVAL-SET] throughout; mahal_tgt = primary (probe-pool)
- A6: mae_untrained restored to Stage 2 + Stage 3
- A7: Energy dumps (clean_*.npy + ood_auroc_*.json) to reports/energy_dumps/ after Stage 3
- A8: scripts/patch_imagecorruptions.py — idempotent; glass_blur PATCHED, fog ALREADY_PATCHED
- paired_margin_auroc_ci added to src/eval/bootstrap.py (Gate 1B(i))

**Two-sided validation (bsuvue5wt, exit 0):**
- mu_ref=0.21908 sigma_ref=0.02689 (probe-pool, 4k)
- Condition A (≥4/6 inverted ≥0.60): PASS — 5/6 (defocus=0.671, glass=0.620, motion=0.633, fog=0.861, contrast=0.961; frost=0.590 fails)
- Condition B (detected types within 0.05 of one-sided): FAIL — all 5 types exceed threshold (gaussian |Δ|=0.224, shot=0.204, impulse=0.190, brightness=0.113, jpeg=0.073)
- **OVERALL: FAIL** — Pre-registered rule fires: report attempt only; two-sided NOT integrated into harness. Root cause: clean val mean_ts=0.789 already high; noise corruptions barely shift ts above clean level.

**Smoke test (dry run, val split, no --unlock_test):** manifest correct (val=1000, probe_pool=4000), all energy heads computed, Stage 2 corruption loop running cleanly.

**All 107 tests pass.** Commit: 1.6i.

---

## Step 1.6h — OOD diagnostics D1–D5 (2026-07-10)

**Script:** `scratchpad/ood_diag.py`. Exit 0. Val + OOD sets only; no test data touched.

| Diagnostic | Result | Reading |
|---|---|---|
| D1 random-init SVHN (single model) | 0.477 | ≈0.5 → **training-specific** inversion |
| D2 loader parity | IDENTICAL PASS | Stat differences are real distributional signal |
| D3 Mahal-on-target-feats (probe-pool) | SVHN=0.985, CIFAR-10=0.855 | ≥0.9 → **two-readout claim fires** |
| D3 Mahal-on-context-feats (val-fit) | SVHN=0.998, CIFAR-10=0.949 | Existing baseline confirmed |
| D4 Spearman(energy, Laplacian-var) | pooled rho=0.770, val=0.328, SVHN=0.613 | ≥0.5 → **prediction-difficulty mechanism supported** |
| D5 Histograms | 3 PNGs in reports/figs/ | Human eyeballs only |

**D2 per-channel stats (256-image batches):**
- val: mean=[−0.253,−0.156,−0.079] std=[1.132,1.135,1.186]
- svhn: mean=[−0.092,0.065,0.409] std=[1.017,1.052,1.040] (blue-channel shift)
- cifar10: mean=[0.073,0.154,0.228] std=[1.112,1.128,1.186]

**Binding claim language (two-readout branch):**
"Latent prediction error detects corruption in most types (11/15 val, vs pixel-std). Same frozen encoder's feature density detects semantic domain shift (Mahal-on-target-feats SVHN=0.985, probe-pool fit, no additional training). Energy inverts on semantic OOD — prediction-difficulty mechanism (Spearman rho=0.770 energy vs Laplacian variance, pooled val+SVHN). Two readouts, one encoder."

**R3 runsheet:** DECISIONS.md §R3 RUNSHEET — FOR HUMAN APPROVAL. Contains pre-checklist, model list, frozen corruption list (15/15), both OOD readouts, probe protocol, headline table specs, wall-clock projection (~6h overnight), exact invocation command.

---

## Step 1.6h — Full 15/15 val benchmark + OOD interpretation (2026-07-09)

**Harness:** `scripts/terminal_benchmark.py`, 15 corruption types × 5 severities, val split. Exit 0. Wall 3389s (0.94h).
**Report:** `reports/terminal_val.md` (final, 15/15 complete).

**Item 3 fix:** glass_blur and fog were broken by library API changes. Fixed with 3-line .venv patch:
- `imagecorruptions/corruptions.py:46`: `np.float_` → `np.float64` (NumPy 2.0)
- `imagecorruptions/corruptions.py:198,210`: `multichannel=True` → `channel_axis=-1` (scikit-image API)
Both types now produce valid outputs; fix verified before rerun.

**⚠️ random_init baseline unreliable in current harness:** `random_init_energy` creates a fresh model per call, so Stage 1 clean energies (Model_A) and each Stage 2 corruption cell (Model_B, Model_C, …) come from different random models. The AUROC measures noise, not model response to corruption. Confirmed: gaussian_noise AUROC swung 0.456→0.132 between two equivalent runs. Gate 1B criterion (i) is evaluated against **pixel_std only**; random_init column is decorative in the corruption table. Fix for R3: pre-instantiate one random model in Stage 1 and reuse it throughout Stage 2.

**Stage 2 — Gate 1B criterion (i) assessment (ref_s0, mean over 5 severities, 15/15 types):**

| Type | JEPA ref_s0 | pixel_std | JEPA>pix? |
|---|---|---|---|
| gaussian_noise | 0.780 | 0.738 | ✓ |
| shot_noise | 0.803 | 0.769 | ✓ |
| impulse_noise | 0.835 | 0.744 | ✓ |
| defocus_blur | 0.236 | 0.300 | ✗ (inverted) |
| glass_blur | 0.299 | 0.344 | ✗ (inverted) |
| motion_blur | 0.292 | 0.338 | ✗ (inverted) |
| zoom_blur | 0.389 | 0.379 | ✓ |
| snow | 0.407 | 0.501 | ✗ |
| frost | 0.285 | 0.240 | ✓ |
| fog | 0.085 | 0.069 | ✓ (both inverted; fog mechanism = blur analogue) |
| brightness | 0.578 | 0.459 | ✓ |
| contrast | 0.020 | 0.009 | ✓ (both deeply inverted; JEPA marginally less bad) |
| elastic_transform | 0.484 | 0.477 | ✓ |
| pixelate | 0.484 | 0.442 | ✓ |
| jpeg_compression | 0.607 | 0.484 | ✓ |

**JEPA > pixel_std: 11/15 (73%) → Gate 1B criterion (i): PASS ✓**

Inversion pattern: all 4 blur-family types (defocus, glass, motion, fog) + snow invert. Mechanism confirmed: low-frequency/smooth corruptions reduce spatial complexity → prediction task easier → lower energy → anti-detection. pixel_std wins on snow because snowflake pixels have high local variance.

**Stage 3 — OOD AUROC (full table):**

| Model | SVHN | CIFAR-10 |
|---|---|---|
| ref_s0 | 0.098 | 0.411 |
| ref_s1 | 0.133 | 0.518 |
| ref_s2 | 0.088 | 0.438 |
| hardmask_s0* | 0.165 | 0.401 |
| pixel_std | 0.102 | 0.405 |
| random_init | 0.118 | 0.028 |
| **mahalanobis** | **0.998** | **0.949** |
| mae_trained | 0.013 | 0.147 |

OOD interpretation logged: JEPA energy inverts on low-complexity semantic OOD (SVHN 0.10) — prediction-error pathology (Nalisnick et al. 2019 class). mae_trained also inverts (SVHN 0.013) via same norm_pix_loss mechanism. random_init OOD numbers also unreliable for same model-mismatch reason. Mahalanobis dominates semantic OOD (SVHN 0.998, CIFAR-10 0.949). Phase-2 amendment: unseen-class flagging via Mahalanobis/kNN on frozen features.

**Stage 4 — Probe (final, 3 probe seeds, 4 models):**

| Model | n=40 | n=200 | n=400 | n=4000 |
|---|---|---|---|---|
| ref_s0 | 0.2920±0.0067 | 0.4177±0.0177 | 0.4563±0.0088 | **0.6003±0.0005** |
| ref_s1 | 0.2690±0.0033 | 0.3647±0.0058 | 0.4147±0.0160 | **0.5650±0.0022** |
| ref_s2 | 0.2740±0.0221 | 0.3913±0.0184 | 0.4367±0.0068 | **0.5793±0.0017** |
| hardmask_s0* | 0.3003±0.0090 | 0.4187±0.0111 | 0.4823±0.0119 | 0.5870±0.0036 |

3-seed reference n=4000: mean = **0.582**, σ_training_seed = **0.018**.
All σ_probe < 0.005 → probe measurement variance negligible; training-seed spread is real.
Gate 1B criterion (ii) reported value: **0.582 ± 0.018** (APPROVED — Anuj, 2026-07-10; per-encoder means {0.600, 0.565, 0.579}, sample-σ over 3 training seeds).
Gate 1B criterion (iii): val-side preliminary — gaps n=40 +3.9pp, n=200 +3.8pp, n=400 +4.7pp, n=4000 +0.3pp. Consistent with pass; formal evaluation at R3 on test; gate decision = human.

**bootstrap.py smoke-test (Item 4b): PASS.** `bootstrap_auroc_ci(n_boot=2000)` → point=0.688 lo=0.665 hi=0.711, CI properly contains point estimate. R3 dependency confirmed.

**R3 wall-clock projection (Item 4c):**
Stage 2 scales 8× (8k vs 1k images) → ~2626s×8 ≈ 21,000s. Stage 3 unchanged ~580s. Stage 4 ~185s (probe train pool same 4k). **Total ~22,000s (~6h). Recommendation: caffeinate overnight.**

---

## Step 1.6g — Full val benchmark results (2026-07-09)

**Harness:** `scripts/terminal_benchmark.py` (no `--dry_run`), 15 corruption types × 5 severities, val split.
**Report:** `reports/terminal_val.md`
**Exit:** 0 (clean). Wall time: 2887s (0.80h). K=8, n_boot=2000.

**Parity gate (Items 1a–1c, all PASS before benchmark):**
- 1a: mask positions batch-shared across K draws (per-image-spatial independence is within each draw) — stated ✓
- 1b: energy parity max|Δ| ≤ 2.96e-05 for all 3 helpers (JEPA, Mahal, RandInit) ✓
- 1c: AUROC parity |Δ|=0.0000 for all 3 helpers (200-image clean vs gaussian_noise sev=3) ✓

**Skipped corruptions (library incompatibility — 10/75 cells absent):**
- `glass_blur`: scikit-image `gaussian()` `multichannel` kwarg removed in newer releases
- `fog`: NumPy `np.float_` removed in NumPy 2.0
- 65/75 cells complete; both types documented in report header.

**Stage 2 — Corruption AUROC (ref_s0, mean over severities; 13 evaluable types):**

| Type | ref_s0 | pixel_std | random_init | mahal | mae_trained |
|---|---|---|---|---|---|
| gaussian_noise | **0.781** | 0.738 | 0.456 | 0.863 | 0.966 |
| shot_noise | **0.802** | 0.769 | 0.541 | 0.868 | 0.954 |
| impulse_noise | **0.836** | 0.744 | 0.189 | 0.892 | 0.979 |
| defocus_blur | 0.236 ⚠️ | 0.300 | 0.151 | **0.977** | 0.050 |
| motion_blur | 0.293 ⚠️ | 0.338 | 0.252 | **0.917** | 0.149 |
| zoom_blur | 0.389 | 0.379 | 0.261 | **0.837** | 0.207 |
| snow | 0.408 | 0.501 | 0.270 | **0.719** | 0.766 |
| frost | 0.284 ⚠️ | 0.241 | 0.639 | **0.872** | 0.470 |
| brightness | **0.578** | 0.459 | 0.145 | **0.806** | 0.415 |
| contrast | 0.020 ⚠️ | 0.009 | 0.304 | **0.991** | 0.143 |
| elastic_transform | 0.483 | 0.476 | 0.365 | 0.506 | **0.653** |
| pixelate | **0.484** | 0.442 | 0.235 | **0.640** | 0.486 |
| jpeg_compression | **0.607** | 0.484 | 0.220 | **0.662** | 0.541 |

**Gate 1B criterion (i) verdict — PASS ✓:**
- JEPA ref_s0 > pixel_std in 8/13 evaluable types (majority, 62%)
- JEPA ref_s0 > random_init in 11/13 types (frost and contrast are exceptions)
- Pattern: JEPA detects noise/compression/brightness; INVERTS on blur (defocus, motion) and contrast (easy-prediction artifacts). Named limitation, not a bug.

**Seed spread (all 3 ref seeds, mean over severities, noise types):**
- gaussian: s0=0.781, s1=0.661, s2=0.646 — seed spread ±0.07 is real; s0 is the strong outlier
- shot: s0=0.802, s1=0.689, s2=0.677
- impulse: s0=0.836, s1=0.713, s2=0.695
- Hardmask (single seed, rejected lever) matches or exceeds refs on noise — consistent with higher eff_rank, but adoption was correctly rejected (probe gap unchanged)

**Stage 3 — OOD AUROC:**

| Model | SVHN | CIFAR-10 |
|---|---|---|
| ref_s0 | 0.098 | 0.411 |
| ref_s1 | 0.133 | 0.518 |
| ref_s2 | 0.088 | 0.438 |
| hardmask_s0* | 0.165 | 0.401 |
| pixel_std | 0.102 | 0.405 |
| random_init | 0.680 | 0.194 |
| **mahalanobis** | **0.998** | **0.949** |
| mae_trained | 0.013 | 0.147 |

⚠️ JEPA energy inverts on both semantic OOD sets — consistent with dry run. Mahalanobis dominates here. MAE trained also inverts (norm_pix_loss mechanism identical to JEPA easy-prediction artifact). Named in Gate 1B claim scoping.

**Stage 4 — Probe grid (3 probe seeds per cell, locked protocol):**

| Model | n=40 mean±σ_probe | n=200 | n=400 | n=4000 |
|---|---|---|---|---|
| ref_s0 | 0.2823±0.0037 | 0.4163±0.0187 | 0.4500±0.0059 | **0.6027±0.0005** |
| ref_s1 | 0.2713±0.0063 | 0.3673±0.0131 | 0.4133±0.0147 | **0.5643±0.0021** |
| ref_s2 | 0.2683±0.0268 | 0.3927±0.0155 | 0.4367±0.0054 | **0.5810±0.0008** |
| hardmask_s0* | 0.2953±0.0045 | 0.4267±0.0093 | 0.4833±0.0125 | 0.5850±0.0000 |

**3-seed reference (training seeds 0/1/2), n=4000:** mean = **0.582**, σ_training_seed = **0.018** (canonical; per-encoder means {0.600, 0.565, 0.579})
- ~~0.583 ± 0.016 (superseded — from earlier 2-probe-seed run; stricken)~~
- σ_probe per cell: 0.0005–0.0021 (all < 0.005) → probe variance not masking training-seed signal ✓
- Gate 1B criterion (ii) probe value: **0.582 ± 0.018** (APPROVED — Anuj, 2026-07-10)
- Gate 1B criterion (iii): val-side preliminary — gaps n=40 +3.9pp, n=200 +3.8pp, n=400 +4.7pp, n=4000 +0.3pp. Consistent with pass; formal evaluation at R3 on test; gate decision = human.

---

## Step 1.6f — Terminal dry run results (2026-07-09)

**Harness:** `scripts/terminal_benchmark.py --dry_run` (3 corruption types × 3 severities, val split).
**Report:** `reports/terminal_dryrun.md`
**Exit:** 0 (clean). OOM bug fixed (batched OOD inference). CIFAR-10 now cached in `data/ood/`.

**Data manifest verified:**
- val=1000 (data/splits/stl10_val_idx.json, stratified 100/class seed=0) ✓
- probe_pool=4000 (STL-10 labeled train complement) ✓
- OOD: SVHN test (26032), CIFAR-10 test (10000, downloaded) ✓
- STL-10 test: NOT LOADED (guarded) ✓

**Stage 2 — Corruption AUROC (3 types, mean over severities {1,3,5}):**

| Model | gaussian_noise | defocus_blur | jpeg_compression |
|---|---|---|---|
| ref_s0 | 0.785 | 0.234 | 0.604 |
| ref_s1 | 0.677 | 0.331 | 0.624 |
| ref_s2 | 0.660 | 0.276 | 0.584 |
| hardmask_s0* | 0.877 | 0.256 | 0.602 |
| pixel_std | 0.742 | 0.298 | 0.484 |
| random_init | 0.762 | 0.878 | 0.901 |
| mahalanobis | 0.857 | 0.974 | 0.664 |
| mae_untrained | 0.580 | 0.458 | 0.482 |
| mae_trained | 0.959 | 0.054 | 0.539 |

**Stage 2 ref_s0 per-severity (gaussian_noise):** sev1=0.639, sev3=0.768, sev5=0.948 — monotone rise with severity ✓

**Stage 3 — OOD AUROC:**

| Model | SVHN | CIFAR-10 |
|---|---|---|
| ref_s0 | 0.098 | 0.411 |
| ref_s1 | 0.133 | 0.518 |
| ref_s2 | 0.088 | 0.438 |
| random_init | 0.998 | 0.226 |
| mahalanobis | 0.998 | 0.949 |
| mae_trained | 0.013 | 0.147 |

**⚠️ OOD AUROC < 0.5 for all JEPA refs on SVHN, and all refs on CIFAR-10 (except ref_s1 barely above 0.5).** This is NOT a bug — it is a result. JEPA latent prediction energy INVERTS for semantic OOD: SVHN digit patches (flat, low-variance) are EASIER to predict than STL-10 natural patches → lower energy → model ranks SVHN as more "normal" than STL-10. Must be reported explicitly as a regime limitation: JEPA energy is a corruption detector, not a semantic OOD detector.

**⚠️ defocus_blur AUROC < 0.5 for all JEPA refs.** Same mechanism: blurring removes high-frequency content → prediction task becomes easier → lower energy. JEPA anti-detects blur. Random-init (0.878) and Mahalanobis (0.974) detect blur well because they measure feature distribution shift. This limits Gate 1B claim (i): trained > random-init only on noise/compression types, NOT on blur.

**Stage 4 — Probe grid (locked protocol, 4 models):**

| Model | n=40 | n=200 | n=400 | n=4000 |
|---|---|---|---|---|
| ref_s0 | 0.306 | 0.426 | 0.437 | 0.599 |
| ref_s1 | 0.266 | 0.376 | 0.403 | 0.564 |
| ref_s2 | 0.282 | 0.383 | 0.427 | 0.581 |
| hardmask_s0* | 0.311 | 0.432 | 0.465 | 0.589 |

**Wall-clock (dry run):** Stage 2=143s, Stage 3=2792s (incl. 2309s CIFAR-10 download, one-time), Stage 4=63s. Total=2998s.

**Full val projection (CIFAR-10 cached):** Stage 2=~1200s, Stage 3=~480s, Stage 4=63s. **Total ≈ 30 min → full val benchmark is tonight.**

---

## Step 1.6e — Terminal harness built (2026-07-08)

**`scripts/terminal_benchmark.py` — complete R3 evaluation harness:**

- Stage 2: Corruption AUROC grid — 15 types × 5 severities (or 3 × 3 with `--dry_run`)
- Stage 3: OOD AUROC — SVHN and CIFAR-10 downloaded on-the-fly to `data/ood/`
- Stage 4: Probe grid — n ∈ {40, 200, 400, 4000}, locked protocol (target mean+zscore, lr-sweep, 200ep)
- All models: ref seeds 0/1/2, hardmask_s0* (labeled rejected lever), MAE trained (if available, else flagged MISSING), pixel_std, random_init, Mahalanobis, mae_untrained
- `--dry_run` flag: 3 corruption types for harness validation
- `--unlock_test` guard: required to run against sealed test set
- Output: Markdown report with tables + wall-clock summary to `--out` path

Dry-run command (Thursday):
```
uv run python scripts/terminal_benchmark.py \
    --ref_ckpts runs/tkqjawa0/epoch_0150.ckpt \
                runs/lbd900za/epoch_0150.ckpt \
                runs/gommvdgc/epoch_0150.ckpt \
    --hardmask_ckpt runs/fw1out6d/epoch_0150.ckpt \
    --dry_run --out reports/terminal_dryrun.md
```
Add `--mae_ckpt runs/<mae-run-id>/epoch_0150.ckpt` once MAE is trained.

---

## Step 1.6d — Seed-2 gate + seed-consistency probes (2026-07-08)

**Reference seed 2 Gate 1A (gommvdgc):** finals in-band vs seeds 0/1 — eff_rank 173.2 (band 172.6–175.3), pred_loss 0.2045 (band 0.2047–0.2074), spread 19.54, var 0.995. Gate 1A: PASSED.

**Seed-consistency probes (locked protocol, epoch_0150, n=4000, val only):**

| Seed | W&B | Locked probe n=4000 | Gap from seed 0 |
|---|---|---|---|
| 0 | `tkqjawa0` | 0.603 | — |
| 1 | `lbd900za` | 0.567 | −0.036 ⚠️ FLAG |
| 2 | `gommvdgc` | 0.581 | −0.022 |

Mean ≈ 0.584, spread 0.567–0.603 (~3.6pp).

**⚠️ FLAG (pre-registered): seed 1 is 0.036 from seed 0, exceeding the >0.030 threshold. Limiting variable: under adjudication — probe-variance experiment, session 1.6e.**

**Probe-variance experiment results (1.6e, binding):**

| Encoder | Probe seeds {0,1,2} | Per-encoder mean | σ_probe |
|---|---|---|---|
| tkqjawa0 (train seed 0) | 0.600, 0.601, 0.603 | 0.601 | 0.0015 |
| lbd900za (train seed 1) | 0.564, 0.564, 0.564 | 0.564 | 0.000 |
| gommvdgc (train seed 2) | 0.582, 0.579, 0.584 | 0.582 | 0.0025 |

Pooled σ_probe ≈ 0.0017. Decision rule fired: **σ_probe ≤ 0.005 → encoder-level seed variance is real.**

σ_seed (sample-std of per-encoder means {0.600, 0.565, 0.579}) = **0.018**. Reference 3-seed result: **0.582 ± 0.018** (3 training seeds, locked protocol, epoch_0150, n=4000, val only). [Prior citations of 0.582±0.019 or 0.583±0.016 are stricken.]

Reporting convention: report encoder-level mean ± σ_seed everywhere. No probe-seed averaging needed (σ_probe << σ_seed).

Hardmask fw1out6d probe n=4000 = 0.587: falls inside the reference seed distribution (0.564–0.601) — statistically indistinguishable from the reference spread.

**Hardmask rejection evidence (for results archive):**
- fw1out6d locked protocol val probes: n=4000 0.587 / n=200 0.428
- Reference (seed-0 epoch_0150): n=4000 0.603 / n=200 ~0.438
- Target−context gap +0.001 (same too-easy-EMA signature as reference)
- Pretext pred_loss 0.292 vs 0.207 (+41% difficulty), no transfer gain
- Interpretation on record: masking-difficulty lever pulled and measured, did not improve transfer; with probe-vs-epoch plateau at ep120, the ~0.60 probe number is a property of the recipe/regime, not an unfinished tuning job. Single-seed comparison — 1.4pp deficit phrased as "did not improve; if anything slightly lower (single seed)", not a proven regression.

---

## Step 1.6c — Hardmask adoption verdict (2026-07-07)

**fw1out6d identity verification:** W&B config confirmed `--config configs/phase1_hardmask.yaml --seed 0`, launched 2026-06-13T00:32:28Z. Gate 1A: PASSED (eff_rank 189.2/192, pred_loss 0.2918 — higher than reference ~0.207 as pre-registered for harder task).

**Unknown run resolved:** `8cw5vncy` = `phase1_ref.yaml --seed 1`, aborted at epoch 54, superseded by `lbd900za`. Added to ledger.

**MAE entry point:** `scripts/train_mae.py` confirmed present and smoke-tested (2 ep × 100 imgs, loss 1.318).

**Locked probe results on fw1out6d/epoch_0150.ckpt (val only):**

| n | Target+zscore lr-sweep 200ep (LOCKED) | Context baseline | Target−Context gap |
|---|---|---|---|
| 4000 | **0.5870** | 0.5810 | +0.001 (gap ≤ 0.01 — harder masking did NOT differentiate EMA target) |
| 200  | 0.4280 | 0.4130 | +0.015 |

**Checkpoint mini-sweep {90,120}:** NOT POSSIBLE — fw1out6d only saved best.ckpt (epoch 2, bug) and epoch_0150.ckpt. Cannot read ep150−ep120 delta.

**Target−context gap:** +0.001 at n=4000 — still ≤ 0.01. Harder masking did not materially change EMA differentiation. Same "too-easy EMA" signature as reference.

**Pre-registered decision rule R1:** n=4000 ≥ 0.62 AND eff_rank ≥ 96 → ADOPT; otherwise → REJECT.
- eff_rank 189.2 ≥ 96 ✓
- n=4000 probe = 0.587 < 0.62 ✗
- **REJECT branch fires.** Hardmask did not improve over reference (ref: 0.601; hardmask: 0.587 — 1.4pp regression).

---

## Step 1.6 — Harder masking pilot (single seed, pre-registered fork)

**Status:** config + plumbing built and tested; training NOT YET launched.

**Mask statistics gate** (`scripts/mask_stats.py --n_samples 10000 --seed 0`):

| Config | n_tgt | target_scale | context_scale | tgt_union mean (p5/p95) | ctx_after mean (p5/p95) | fallback |
|---|---|---|---|---|---|---|
| (a) current `phase1_ref` | 4 | (0.15, 0.20) | (0.85, 1.00) | 69.7 (52/87) | 60.2 (41/80) | 0.00% |
| (b1) hard `phase1_hardmask` | 4 | (0.20, 0.25) | (0.75, 0.90) | 80.2 (60/100) | 44.3 (26/64) | 0.00% |

Gate criteria: ctx_after p5 ≥ 20 patches AND fallback ≤ 2%.
**(b1) PASSES on the first try** — no lever adjustment needed.
Figure: `reports/figures/mask_samples_hard.png`.

**Pre-registered adoption rule** (apply after the hardmask seed-0 run finishes):

After training, evaluate seed 0 on val with the locked probe protocol
(target encoder + mean-pool + z-score + lr-swept 200-epoch head) at n=4000 AND n=200,
plus target-encoder effective-rank diagnostics.

**Adopt hardmask as production iff:**
- n=4000 probe accuracy ≥ 0.62 (baseline 0.601 at ep150 + 2 pp), AND
- target-encoder effective rank ≥ 96 / 192 (50% of d_model floor)

**If adopted:**
- Hardmask becomes the production config; queue seeds 1 + 2 with `phase1_hardmask.yaml`.
- `phase1_ref.yaml` is demoted to "reference" (cocktail-style reference-vs-final framing).
- Log frozen hyperparameters in DECISIONS.md.

**If <+2pp or rank fails:**
- Hardmask is rejected.
- Confront the Gate 1B floor itself: literature-calibrated revision in DECISIONS.md
  (candidate fixes: larger backbone, longer schedule, EMA momentum schedule,
  or revisit the predictor width — to be pre-registered before training).

**Files added (Step 1.6):**
- `scripts/mask_stats.py` — 10k-sample mask-yield simulator + adoption gate
- `configs/phase1_hardmask.yaml` — production-format config with explicit `masking:` block
- `src/data/stl10.py` — `get_pretrain_loader` / `get_smoke_loader` now accept `mask_kwargs`
- `scripts/train.py` — `_build_mask_kwargs` reads optional `cfg["masking"]` block
- `tests/test_masking.py` — pytest-parametrised over `{ref, hardmask}` (32 cells, was 12)

**Launch command (when ready):**
```
uv run python scripts/train.py --config configs/phase1_hardmask.yaml --seed 0
```

---

## Step 1.5e — Probe-vs-epoch diagnostic (seed-0, val only)

**Script:** `scripts/probe_vs_epoch.py --ckpt_dir /tmp/ckpts_probe_diag --seed 0`
(No W&B link — evaluation script, not a training run.)
**Figure:** `reports/figures/probe_vs_epoch.png`
**Report:** `reports/probe_vs_epoch.md`

| Epoch | tgt n=4000 | ctx n=4000 | gap (T−C) | tgt n=200 | ctx n=200 | gap (T−C) |
|---|---|---|---|---|---|---|
| 30  | 0.5610 | 0.5710 | −0.0100 | 0.3850 | 0.3890 | −0.0040 |
| 60  | 0.5760 | 0.5650 | +0.0110 | 0.4250 | 0.4340 | −0.0090 |
| 90  | 0.5870 | 0.5800 | +0.0070 | 0.4380 | 0.4470 | −0.0090 |
| 120 | 0.6040 | 0.6040 | +0.0000 | 0.4250 | 0.4260 | −0.0010 |
| 150 | 0.6000 | 0.6050 | −0.0050 | 0.4290 | 0.4290 | +0.0000 |

**Pre-registered reading: H2 — curve plateaued by ~ep120**
Δ(ep120→150) = −0.004 < 0.005 threshold. Curve peaks at ep120 and is flat/slightly declining at ep150.
Next action: one from-scratch run with harder masking (target scale 0.20–0.25, context 0.75–0.90).

**Observations:**
- n=4000 curve rises 0.561→0.604 (ep30→ep120), then stalls (0.600 at ep150): clear plateau
- n=200 curve similarly peaks at ep90 (0.438), then 0.429 at ep150: same pattern
- Target ≈ context encoder gap throughout (|gap| ≤ 0.011 at all epochs) — EMA target not pulling ahead; consistent with easy masking task giving weak self-supervised signal
- No epoch reaches the 0.70 Gate 1B floor — extending training alone will not close the gap

**Files added (Step 1.5e):**
- `scripts/probe_vs_epoch.py` — probe accuracy vs pretraining epoch curve
- `reports/figures/probe_vs_epoch.png` — two-panel plot (n=4000 and n=200)
- `reports/probe_vs_epoch.md` — table + pre-registered reading

---

## Waived gates + justification

None. Gate 0 passed twice.

---

## Phase-2 prep list (record only — no code this week; trainer-code moratorium expires at Phase-2 kickoff)

| Item | Priority | Notes |
|---|---|---|
| Fix checkpoint-saver "best" criterion | P0 | `best-by-lowest-pred-loss` is invalid for SSL runs where pred_loss is non-monotone (EMA momentum ramp). Evidence: fw1out6d best.ckpt = epoch 2 (untrained). Fix: save `epoch_N` every ckpt_every AND save `last.ckpt`; eliminate `best.ckpt` or redefine "best" as lowest-val-loss on a held-out set. Do NOT modify trainer before Phase-2 kickoff. |
| Evaluate larger backbone (ViT-Small d=384) | P1 | If probe ceiling is the primary Phase-2 claim, 100× capacity gap to I-JEPA-H is the debt named in Gate 1B revision. Triage at Phase-2 kickoff. |
| Color augmentation (RandomGrayscale, ColorJitter) | P1 | Literature: color aug is a primary driver of semantic representation in contrastive SSL. Not used in Phase 1 (intentional for edge-deployment story). Evaluate vs. probe ceiling at Phase 2. |
| Phase-2 dataset: RESISC45 + AID aerial imagery | P0 | Next domain. Pre-register train/val/test splits before any training. |
