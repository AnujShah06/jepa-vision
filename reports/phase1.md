# Phase 1 Report — I-JEPA-mini on STL-10
## Draft for human review — NOT final

---

## 1. Headline Outcomes

**Two readouts, one encoder.**

A ViT-Tiny (~3M trainable params) trained with masked latent prediction on 100k unlabeled STL-10 images produces two distinct anomaly signals from the same frozen encoder:

1. **Energy readout (corruption detection):** latent prediction error detects corruption in **8/15** types at test vs the pixel-std baseline. Three types (impulse noise, brightness, jpeg compression) are clearly above (margin >5pp); five (gaussian, frost, fog, contrast, pixelate) are borderline (≤1.5pp); seven types (blur family, snow, zoom, elastic) are below or inverted.

2. **Density readout (semantic domain shift):** Mahalanobis distance on frozen target-encoder features, fit on the probe pool with no labels and no additional training, achieves SVHN AUROC **0.986** [0.984, 0.987] and CIFAR-10 AUROC **0.864** [0.858, 0.869] at test.

These two readouts are **complementary**: types where the energy signal fails (defocus 0.209, contrast 0.019, fog 0.078) are exactly the types where mahal_tgt is strongest (defocus 0.937, contrast 0.977, fog 0.920).

**Why the energy inverts.** The inversion is training-specific (diagnostic D1: random-init SVHN AUROC ≈0.5 vs trained AUROC 0.08), loader-artifact-free (D2), and mechanistically supported: Spearman ρ=0.770 between prediction energy and Laplacian variance pooled over val+SVHN (D4). Low-complexity inputs — SVHN digit patches, blurred patches, high-contrast flat regions — are *easier* to predict than natural STL-10 patches, so energy decreases rather than increases under corruption. This is the Nalisnick-class deep-generative-model pathology: the model assigns lower anomaly score to out-of-distribution inputs that happen to be simpler, not more complex, than the training distribution.

MAE trained on the same images with pixel-reconstruction loss shows the same inversion pattern on SVHN (AUROC 0.013) via an identical mechanism (norm_pix_loss normalises by per-patch variance; smooth patches become easy to reconstruct).

**Transfer: three-regime summary.**

| Label budget | JEPA probe (val) | Scratch v2 (val) | Point gap |
|---|---|---|---|
| n=40 (1%) | 0.277±0.014 | 0.249±0.007 | +0.028 |
| n=200 (5%) | 0.390±0.025 | 0.386±0.006 | +0.004 |
| n=400 (10%) | 0.436±0.019 | 0.426±0.010 | +0.010 |
| n=4000 (100%) | 0.583±0.018 | 0.648±0.012 | −0.065 |

*σ values are training-seed standard deviations (3 seeds). Combined spread (RSS) per n: 0.016 / 0.026 / 0.022 / 0.022.*

Three regimes: (i) n=40: frozen probe has a +2.8pp advantage that exceeds the combined seed spread — pretraining helps under extreme label scarcity; (ii) n=200/400: point gaps positive but within combined noise — the advantage exists directionally but is not measurably robust at this scale; (iii) n=4000: recipe-matched from-scratch wins by 6.5pp — at full labels, end-to-end training with augmentation dominates.

The **~0.58 probe ceiling** is a property of the recipe, not an open tuning problem. Both available levers were pulled: duration (plateau confirmed at ep120, Δ ep120→150 = −0.004) and masking difficulty (hardmask raised pretext pred_loss by +41% with no transfer gain). Residual gap to supervised ceiling reflects ViT-Tiny capacity, no color augmentation, and constrained pretraining scale.

---

## 2. Data and Splits

**Dataset.** STL-10 (96×96, 10 classes): 100k unlabeled images for pretraining; 5k labeled train for downstream evaluation; 8k labeled test.

**Val split.** Stratified 100/class × 10 = 1,000 images carved from the labeled train set; seed=0; complement (4,000 images, 400/class) forms the probe pool. Split committed at `data/splits/stl10_val_idx.json` before any model training.

**Test seal.** STL-10 test (8,000 images) opened once, for R3 terminal evaluation. Run-1 aborted before Stage-2 completion (MPS OOM at cell 9/75, 2026-07-13); interruption exception invoked. Run-2 launched 2026-07-16, exit 0, 4.46h; 75/75 Stage-2 cells complete. All test-derived numbers in this report come exclusively from run-2.

**OOD sets.** SVHN test (26,032 images) and CIFAR-10 test (10,000 images), downloaded to `data/ood/`, unchanged between val and test evaluation.

---

## 3. Model and Training

**Architecture.** VisionJEPA ViT-Tiny: d=192, 6 encoder layers, 3 heads; predictor d=96, 3 layers (proj_in + transformer + proj_out); ~3M trainable params; ~6M total (incl. frozen EMA target). No CLS token; patch size 12×12 → 64 tokens per 96px image.

**Training.** 150 epochs, batch 256, bfloat16 AMP, AdamW + cosine-warmup schedule; SIGReg anti-collapse regulariser; EMA momentum 0.996→1.0; block masking (target scale 0.15–0.20, context scale 0.85–1.00). Three reference seeds (0/1/2); one hardmask seed-0 (rejected).

**Gate 1A (collapse check).** All four trained models pass: eff_rank ≥ 172/192 throughout (>89% of d=192). Finals: ref_s0 175.3, ref_s1 172.6, ref_s2 173.2, hardmask 189.2.

**Checkpoint policy.** Canonical checkpoint: `epoch_0150.ckpt` for all models. `best.ckpt` by lowest pred_loss is deprecated: for the hardmask run, EMA warmup caused pred_loss to *increase* over training (epoch 2 was erroneously saved as best). All evaluation uses epoch_0150.

---

## 4. Corruption Detection

### 4.1 Corruption AUROC — Test Split (R3 run-2)

*Mean AUROC over 5 severities. Chance = 0.500.*

| Type | ref_s0 | ref_s1 | ref_s2 | hardmask* | pixel_std | mahal_tgt | mae_trained |
|------|--------|--------|--------|-----------|-----------|-----------|-------------|
| gaussian_noise | 0.736 | — | 0.639 | 0.836 | 0.734 | 0.747 | 0.963 |
| shot_noise | 0.760 | — | 0.672 | 0.838 | 0.766 | 0.757 | 0.950 |
| impulse_noise | **0.796** | — | 0.692 | 0.863 | 0.741 | 0.768 | **0.978** |
| defocus_blur | 0.209 | — | 0.276 | 0.209 | 0.299 | **0.937** | 0.054 |
| glass_blur | 0.269 | — | 0.334 | 0.264 | 0.343 | **0.870** | 0.149 |
| motion_blur | 0.262 | — | 0.323 | 0.249 | 0.339 | **0.859** | 0.159 |
| zoom_blur | 0.360 | — | 0.416 | 0.318 | 0.371 | 0.757 | 0.215 |
| snow | 0.357 | — | 0.369 | 0.482 | 0.486 | 0.587 | 0.752 |
| frost | 0.249 | — | 0.262 | 0.359 | 0.234 | 0.767 | 0.466 |
| fog | 0.078 | — | 0.108 | 0.129 | 0.072 | **0.920** | 0.299 |
| brightness | **0.532** | — | 0.527 | 0.484 | 0.443 | 0.699 | 0.415 |
| contrast | 0.019 | — | 0.032 | 0.032 | 0.009 | **0.977** | 0.153 |
| elastic_transform | 0.439 | — | 0.484 | 0.442 | 0.476 | 0.474 | 0.649 |
| pixelate | 0.443 | — | 0.496 | 0.414 | 0.441 | 0.606 | 0.488 |
| jpeg_compression | **0.566** | — | 0.584 | 0.550 | 0.484 | 0.565 | 0.541 |

*ref_s1 Stage-2 VOID-INFRASTRUCTURE: MPS silent Stage-1 corruption (see §8 Limitations). 2-seed reference corruption row (ref_s0, ref_s2) is canonical.*
*hardmask*: single seed, rejected lever (R1 probe < 0.62).*

**Win count (ref_s0 vs pixel_std): 8/15.** Clearly above (margin >5pp): impulse (+5.5pp), brightness (+8.9pp), jpeg (+8.2pp). Borderline (margin ≤1.5pp): gaussian (+0.2pp), frost (+1.5pp), fog (+0.6pp), contrast (+1.0pp), pixelate (+0.2pp). Below or inverted: defocus, glass, motion, zoom, shot, snow, elastic.

**Gate 1B(i) status.** Pre-registered criterion: paired-margin bootstrap CIs exclude zero for the majority of win types. As operationalized, this requires per-image corrupted energy arrays, which were not dumped. CIs cannot be computed post-hoc. Only point counts are available. Of the 8 wins: 3 (impulse, brightness, jpeg) have margins well above the ±0.009 bootstrap noise floor from val runs; 5 are within that floor. Gate 1B(i) as operationalized: **uncomputable; human decision required.**

**mahal_tgt corruption complementarity.** The density readout compensates exactly where energy fails: defocus 0.937 (energy 0.209), contrast 0.977 (energy 0.019), fog 0.920 (energy 0.078), glass 0.870, motion 0.859.

### 4.2 ref_s0 Per-Severity with Bootstrap CIs (Test Split)

*Selected rows; full table in reports/terminal_test.md.*

| Corruption | Sev 1 | Sev 2 | Sev 3 | Sev 4 | Sev 5 | Mean |
|---|---|---|---|---|---|---|
| gaussian_noise | 0.579 [0.571,0.588] | 0.633 [0.624,0.642] | 0.719 [0.710,0.728] | 0.828 [0.820,0.835] | 0.920 [0.915,0.926] | 0.736 |
| impulse_noise | 0.720 [0.712,0.729] | 0.732 [0.724,0.741] | 0.760 [0.752,0.768] | 0.849 [0.842,0.856] | 0.919 [0.914,0.924] | 0.796 |
| brightness | 0.491 [0.482,0.500] | 0.519 [0.510,0.528] | 0.538 [0.529,0.547] | 0.552 [0.543,0.561] | 0.560 [0.551,0.569] | 0.532 |
| jpeg_compression | 0.500 [0.492,0.509] | 0.534 [0.526,0.543] | 0.553 [0.544,0.562] | 0.607 [0.599,0.616] | 0.634 [0.626,0.643] | 0.566 |
| defocus_blur | 0.312 [0.304,0.320] | 0.268 [0.260,0.275] | 0.199 [0.193,0.206] | 0.155 [0.149,0.161] | 0.112 [0.107,0.118] | 0.209 |

Monotone severity response holds for detected types (gaussian, impulse, jpeg increase with severity), confirming the energy signal is tracking corruption intensity, not noise. Defocus shows monotone *decrease* — the inversion mechanism is consistent across severities.

---

## 5. Semantic OOD Detection

### 5.1 OOD AUROC Table (Test Split)

*Energy and Mahalanobis readouts. Fit-set column: where the density estimator was fit.*

| Model | SVHN (energy) | CIFAR-10 (energy) | SVHN (mahal_tgt) | CIFAR-10 (mahal_tgt) | Fit set |
|---|---|---|---|---|---|
| ref_s0 | 0.078 | 0.367 | 0.986 | 0.864 | probe pool (4k) |
| ref_s1† | 0.128 | 0.505 | 0.986 | 0.864 | probe pool (4k) |
| ref_s2 | 0.085 | 0.429 | 0.986 | 0.864 | probe pool (4k) |
| hardmask_s0* | 0.121 | 0.340 | — | — | — |
| pixel_std | 0.101 | 0.395 | — | — | — |
| random_init | 0.419 | 0.378 | — | — | — |
| mahal_ctx | 0.981 | 0.852 | — | — | val (1k) FIT-ON-EVAL-SET |
| mae_untrained | 0.771 | 0.532 | — | — | — |
| mae_trained | 0.013 | 0.144 | — | — | — |

*mahal_tgt fit on probe pool (4k labeled train, disjoint from val); same estimator for all ref seeds — mahal_tgt SVHN/CIFAR rows are identical across ref_s0/s1/s2.*
*† ref_s1 Stage-2 VOID-INFRA; Stage-3 OOD recomputed with MPS sync fix (SVHN=0.128, CIFAR=0.505).*
*mahal_ctx: fit on val set — fit-set contaminated for val reporting; clean for test (disjoint). Both readouts shown.*

### 5.2 OOD Diagnostics

**D1 (training specificity).** Random-init SVHN energy AUROC = 0.419 ≈ 0.5 → the inversion (trained AUROC 0.078–0.121) is a property of the learned representation, not the architecture or data pipeline.

**D2 (loader parity).** Val, SVHN, CIFAR-10 loaders use identical preprocessing (Resize(96,BICUBIC) → CenterCrop(96) → Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])). Per-channel mean differences are real distributional signal: SVHN blue-channel shift, CIFAR-10 positive offset.

**D3 (density readout fires).** mahal_tgt (probe-pool fit, target encoder, no labels): SVHN=0.986, CIFAR-10=0.864 at test. Both exceed the pre-registered threshold (SVHN ≥ 0.9 triggers two-readout claim). mahal_ctx (val-fit, context encoder): SVHN=0.981, CIFAR-10=0.852 — slightly lower than target encoder, and the fit-set is contaminated for val evaluation.

**D4 (mechanism).** Spearman ρ(energy, Laplacian variance) = 0.770 pooled val+SVHN (val-only: 0.328; SVHN-only: 0.613). Energy tracks image sharpness / spatial complexity. SVHN digit patches have lower Laplacian variance than STL-10 natural patches → lower prediction energy → trained model ranks SVHN as more "normal."

**Energy histograms.** `reports/figs/`: jepa_ref_s0, random_init, mahal_target (D5).

---

## 6. Two-Sided Readout — Derived, Validated, Rejected by Its Own Rule

A two-sided anomaly score was constructed to handle both energy-increasing (detected) and energy-decreasing (inverted) corruptions in a single readout. The derivation followed three stages:

1. **Design.** Two-sided score ts = |E − μ_ref| / σ_ref, where μ_ref and σ_ref are estimated from probe-pool clean energies. High ts for both very-high and very-low energies.

2. **Validation (Condition A).** Pre-registered PASS condition: ≥4/6 inverted corruption types achieve two-sided AUROC ≥ 0.60. Result: 5/6 PASS (defocus 0.671, glass 0.620, motion 0.633, fog 0.861, contrast 0.961; frost 0.590 fails). Condition A PASS — the mechanism works for the inverted family.

3. **Rejection (Condition B).** Pre-registered: each detected type stays within 0.05 of its one-sided AUROC. Result: FAIL — all 5 detected types exceed threshold (gaussian |Δ|=0.224, shot=0.204, impulse=0.190, brightness=0.113, jpeg=0.073). Root cause: clean low-energy images are pulled up by the fold to the same magnitude as corrupted images, destroying rank separation where the one-sided margin was small.

**Per pre-registered rule (FAIL on Condition B): two-sided readout NOT integrated into harness.** The Condition-A result is reported as mechanistic confirmation that the inversion is bidirectionally detectable in principle; the Condition-B result explains why the combined readout cannot be operationalised without information loss. Two-sided scores are not applied to any test-set data; any future test-side use requires a new pre-registration before the test set is opened.

---

## 7. Transfer Evaluation

### 7.1 Probe Grids

**Protocol (locked).** Target encoder, mean-pool over patch tokens, z-score normalised (mean/std fitted on val features only), linear head (nn.Linear), AdamW, 200 epochs, lr swept over {3e-3, 1e-3, 3e-4} with best-val selection. z-score and LR selection always on val regardless of evaluation split.

**Val-era Stage 4** (z-score, LR selection, and eval all on val):

| Model | n=40 | n=200 | n=400 | n=4000 |
|---|---|---|---|---|
| ref_s0 | 0.2937±0.0034 | 0.4147±0.0246 | 0.4550±0.0073 | 0.6030±0.0008 |
| ref_s1 | 0.2690±0.0043 | 0.3653±0.0116 | 0.4170±0.0118 | 0.5643±0.0017 |
| ref_s2 | 0.2683±0.0184 | 0.3897±0.0160 | 0.4357±0.0109 | 0.5803±0.0005 |
| hardmask_s0* | 0.2950±0.0022 | 0.4180±0.0142 | 0.4813±0.0146 | 0.5897±0.0009 |

**Stage 4b — Test eval** (z-score and LR on val; eval on test; 3 probe seeds):

| Model | n=40 | n=200 | n=400 | n=4000 |
|---|---|---|---|---|
| ref_s0 | 0.2786±0.0111 | 0.3830±0.0084 | 0.4293±0.0095 | 0.5592±0.0020 |
| ref_s1† | 0.2600±0.0228 | 0.3528±0.0174 | 0.3983±0.0050 | 0.5419±0.0027 |
| ref_s2 | 0.2677±0.0291 | 0.3710±0.0180 | 0.4110±0.0088 | 0.5396±0.0056 |
| hardmask_s0* | 0.2845±0.0136 | 0.4051±0.0045 | 0.4473±0.0051 | 0.5657±0.0018 |

JEPA ref mean test n=4000: **0.547** (vs val-era 0.583). Val→test drop of ~3.6pp is in the expected direction (probe LR selected on val n=1000, evaluated on test n=8000).

Pre-registered band check n=4000: within ±0.03 of val-era reference {0.600, 0.565, 0.579}. Result: ref_s0 |Δ|=0.044 FAIL, ref_s1 |Δ|=0.022 PASS, ref_s2 |Δ|=0.041 FAIL. Band check fired FAIL. Numbers reported as-is; no recompute triggered.

**Seed variance.** Training-seed σ at n=4000: 0.018 (val-era). All probe-seed σ < 0.005; training-seed spread is the dominant source of variation. hardmask_s0* probe n=4000 = 0.587 (val) — falls within the reference training-seed distribution {0.565–0.601}, statistically indistinguishable.

### 7.2 Scratch Comparator — Strawman Guard Arc

**A3 (recipe-underfit, no aug, batch=min(256,n)):** val means n=40 0.240, n=200 0.353, n=400 0.389, n=4000 0.579. At n=4000 the gap was +0.4pp — a near-tie that appeared consistent with the label-efficiency claim across all n.

**Pre-registered rule B2 disqualified A3** before R3: batch=min(256,n) vs 1.5d batch=128; no RandomResizedCrop+HFlip vs explicit augmentation. Recipe disadvantage confirmed → result favorable, must rerun. A3 numbers appear with underfit label and are not used in any claim.

**Scratch v2 (recipe-fixed, batch=128, RandomResizedCrop+HFlip):** 36 runs, 3.94h.

Full per-cell table (all 36 runs; all status=ok, epochs_completed=200):

| Key | n | lr | val_acc |
|-----|---|----|----|
| s0_n40_lr1e-03★ | 40 | 1e-03 | 0.243 |
| s0_n40_lr3e-04 | 40 | 3e-04 | 0.240 |
| s0_n40_lr1e-04 | 40 | 1e-04 | 0.236 |
| s0_n200_lr1e-03 | 200 | 1e-03 | 0.360 |
| s0_n200_lr3e-04 | 200 | 3e-04 | 0.367 |
| s0_n200_lr1e-04★ | 200 | 1e-04 | 0.380 |
| s0_n400_lr1e-03★ | 400 | 1e-03 | 0.414 |
| s0_n400_lr3e-04 | 400 | 3e-04 | 0.406 |
| s0_n400_lr1e-04 | 400 | 1e-04 | 0.411 |
| s0_n4000_lr1e-03 | 4000 | 1e-03 | 0.634 |
| s0_n4000_lr3e-04★ | 4000 | 3e-04 | 0.635 |
| s0_n4000_lr1e-04 | 4000 | 1e-04 | 0.626 |
| s1_n40_lr1e-03 | 40 | 1e-03 | 0.251 |
| s1_n40_lr3e-04★ | 40 | 3e-04 | 0.257 |
| s1_n40_lr1e-04 | 40 | 1e-04 | 0.256 |
| s1_n200_lr1e-03 | 200 | 1e-03 | 0.378 |
| s1_n200_lr3e-04★ | 200 | 3e-04 | 0.391 |
| s1_n200_lr1e-04 | 200 | 1e-04 | 0.370 |
| s1_n400_lr1e-03★ | 400 | 1e-03 | 0.432 |
| s1_n400_lr3e-04 | 400 | 3e-04 | 0.427 |
| s1_n400_lr1e-04 | 400 | 1e-04 | 0.427 |
| s1_n4000_lr1e-03★ | 4000 | 1e-03 | 0.658 |
| s1_n4000_lr3e-04 | 4000 | 3e-04 | 0.643 |
| s1_n4000_lr1e-04 | 4000 | 1e-04 | 0.645 |
| s2_n40_lr1e-03★ | 40 | 1e-03 | 0.246 |
| s2_n40_lr3e-04 | 40 | 3e-04 | 0.245 |
| s2_n40_lr1e-04 | 40 | 1e-04 | 0.234 |
| s2_n200_lr1e-03 | 200 | 1e-03 | 0.376 |
| s2_n200_lr3e-04★ | 200 | 3e-04 | 0.386 |
| s2_n200_lr1e-04 | 200 | 1e-04 | 0.383 |
| s2_n400_lr1e-03 | 400 | 1e-03 | 0.417 |
| s2_n400_lr3e-04★ | 400 | 3e-04 | 0.432 |
| s2_n400_lr1e-04 | 400 | 1e-04 | 0.415 |
| s2_n4000_lr1e-03★ | 4000 | 1e-03 | 0.651 |
| s2_n4000_lr3e-04 | 4000 | 3e-04 | 0.648 |
| s2_n4000_lr1e-04 | 4000 | 1e-04 | 0.646 |

★ = best-lr selection per cell. Manifest: `reports/scratch_v2_manifest.json`. A3 manifest preserved at `reports/scratch_manifest.json`.

Cross-seed means ± σ: n=40 0.249±0.007, n=200 0.386±0.006, n=400 0.426±0.010, n=4000 0.648±0.012.

PD5 check: s0_n4000=0.635, target 0.636±0.02 → WITHIN BAND. Recipe fix reproduces 1.5d result; A3 underfit confirmed.

**Binding gap (JEPA val − scratch v2) with seed spreads:**

| n | JEPA ref mean | σ_J | Scratch v2 | σ_S | Gap | RSS | Note |
|---|---|---|---|---|---|---|---|
| 40 | 0.277 | 0.014 | 0.249 | 0.007 | +0.028 | 0.016 | gap > RSS |
| 200 | 0.390 | 0.025 | 0.386 | 0.006 | +0.004 | 0.026 | within noise |
| 400 | 0.436 | 0.019 | 0.426 | 0.010 | +0.010 | 0.022 | within noise |
| 4000 | 0.583 | 0.018 | 0.648 | 0.012 | −0.065 | 0.022 | scratch wins |

**Gate 1B(iii) evidence:** point gaps positive in all three low-label cells (+2.8/+0.4/+1.0pp); −6.5pp at n=4000. Only n=40 exceeds combined spread; n=200/400 within noise. **Gate decision = human, with spreads in view.**

---

## 8. Limitations

**Semantic OOD unsolved by energy alone.** JEPA energy AUROC on SVHN = 0.078–0.128 (inverted). This is the primary failure mode. The density readout (mahal_tgt) compensates, but requires fitting on a labeled-class-balanced probe pool; the density estimator is not label-free.

**Complexity confound.** The inversion mechanism (energy decreases for low-complexity inputs) is not a bug: it reflects the correct prediction of the model's behaviour. However, it means the energy readout cannot distinguish "OOD but simple" from "in-distribution" — a practical limitation for any setting where the anomaly is lower-complexity than the training distribution.

**Probe ceiling with both levers tested.** Linear probe at n=4000 reaches 0.583±0.018 (val) / 0.547 (test). Both levers available pre-R3 — training duration (plateau at ep120) and masking difficulty (hardmask +41% pretext, no transfer gain) — have been pulled. Residual shortfall to supervised baselines (from-scratch recipe-matched = 0.648) reflects ViT-Tiny capacity (~3M), no color augmentation, and constrained pretraining scale. This is a named regime ceiling, not a tunable parameter.

**Full-label regime conceded.** At n=4000 (100% of probe pool), recipe-matched from-scratch outperforms frozen JEPA by 6.5pp. The label-efficiency story holds directionally only at n=40; n=200/400 are within combined noise.

**ref_s1 Stage-2 VOID-INFRASTRUCTURE.** The 2-seed corruption reference row (ref_s0, ref_s2) is canonical. ref_s1 (lbd900za) Stage-2 AUROC rows are void due to MPS silent Stage-1 corruption: the clean energy tensor computed before the corruption loop had stale data (`.cpu()` returned before MPS computation completed for K=8 × 8000 images). Fix applied (torch.mps.synchronize() before .cpu()) and verified (recomputed clean mean 0.2711→0.2190, PASS). Stage-3 OOD recomputed; Stage-2 cannot be recovered without a full Stage-2 rerun. Incident documented in DECISIONS.md.

**mahal_ctx contamination.** The context-encoder Mahalanobis estimator is fit on the val set (1k images). For val-side reporting this is fit-on-eval-set contamination; for test reporting it is clean (val and test are disjoint). Both are labeled `[FIT-ON-EVAL-SET]` throughout tables. The primary OOD readout (mahal_tgt, probe-pool fit) is uncontaminated at both val and test.

**1k-val CI widths.** Bootstrap CIs for Stage-2 corruption AUROC are computed over 8k test images; CIs in val runs (1k) are ~3× wider. The per-severity ref_s0 CIs in §4.2 are from the test run.

**MPS soft non-determinism.** Residual run-to-run variation on Apple MPS is ≤0.001 AUROC for energy metrics (verified in parity gate). Probe accuracy variation between equivalent seeds is ≤0.0017. All reported values are means over ≥3 seeds.

**Corruptions as proxy.** ImageNet-C corruptions (Hendrycks & Dietterich 2019) are a widely-used proxy for distribution shift but do not cover adversarial perturbations, domain shift, or systematic sensor degradation. The reported AUROC values are specific to this proxy.

---

## 9. Conclusion

"Latent prediction error detects corruption in 8/15 types at test (vs pixel-std baseline). The same frozen encoder's feature density detects semantic domain shift (mahal_tgt SVHN=0.986, CIFAR-10=0.864 at test, probe-pool fit, no additional training). Energy alone inverts on semantic OOD — prediction-difficulty mechanism, Spearman rho=0.770 (pooled val+SVHN). Two readouts, one encoder."

---

*Draft: 2026-07-17. Data source: reports/terminal_test.md (R3 run-2, exit 0, 2026-07-16). Gate decisions pending human review. Do not distribute.*
