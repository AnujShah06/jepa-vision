# PROJECT_STATE.md

## Session rules (read at start of every session)

1. One experiment or one component per session. No "while we're at it" scope creep.
2. The LLM proposes; the training run decides. No conclusion enters this file without a W&B link.
3. Any number destined for the resume or report needs ≥3 seeds.
4. Test sets are touched only by the evaluation harness, never during development. Model selection uses validation only.

---

## Current phase / step

**Phase 1 — I-JEPA-mini on STL-10**
**Step 1.5d complete** — probe + from-scratch harness built, smoke-tested at n=200

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
- W&B run ID: tkqjawa0  (checkpoint: runs/tkqjawa0/best.ckpt)
- This IS the completed seed-0 production run; probe grid numbers in Step 1.5d are real.

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

## Next action

**Phase 1, Step 1.5b — production reference run (3 seeds) + Gate 1A/1B:**

Run three seeds back-to-back (seed 1 in progress):

```
uv run python scripts/train.py --config configs/phase1_ref.yaml --seed 0
uv run python scripts/train.py --config configs/phase1_ref.yaml --seed 1   ← in progress
uv run python scripts/train.py --config configs/phase1_ref.yaml --seed 2
```

After each run, record the W&B link here and check Gate 1A (effective_rank > 96 throughout).

After all 3 seeds are trained (150 epochs each):
1. Re-run `probe_sweep.py` per seed; pool results → 3-seed mean ± std per cell
2. Re-run `baseline_sanity.py` per seed; confirm JEPA energy > random-init with CI
3. Gate 1B checks:
   - (i) corruption AUROC clearly above random-init and pixel baselines (majority of corruption types)
   - (ii) probe at n=4000 ≥ 70% val acc
   - (iii) probe > scratch in most low-label cells (already 3/4 on seed-0 preliminary)
4. Update DECISIONS.md with final frozen hyperparameters

---

## Waived gates + justification

None. Gate 0 passed twice.
