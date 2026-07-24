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

[1.4] **Energy = mean smooth-L1 latent prediction error over target patches, averaged over K independent mask samples** — eval-side sampling via sample_block_mask with seeded random.Random; training collator untouched.

[1.4] **K=8 as inference default** — K-sweep on the formal 1,000-image val split (gaussian_noise sev=3, ckpt tkqjawa0/best.ckpt, stratified 100/class seed=0):

| K  | AUROC  | clean σ |
|----|--------|---------|
| 1  | 0.781  | 0.0367  |
| 4  | 0.747  | 0.0298  |
| 8  | 0.763  | 0.0276  |
| 16 | 0.792  | 0.0258  |

AUROC is K-insensitive within single-seed noise (range 0.747–0.792, spread ≈ ±0.02 around 0.77). K=8 chosen on variance/cost: σ drops 25% vs K=1 while costing 8× the forward passes; K=16 adds only 6% further variance reduction at 2× the cost of K=8. Phase 3 C++ export will batch K masks as one forward pass to amortise the overhead.

[1.4] **Per-patch energy: accumulate smooth-L1 errors per patch across K masks, divide by visit count** — unvisited positions left at 0.0 and excluded from scalar mean. Heatmap: bilinear upsample [n_h×n_w] → [H×W], 'hot' colormap, ImageNet-denormalized source image alpha-blended at α=0.5.

[1.5a] **Baseline suite: pixel-std, random-init, Mahalanobis, PixelMAE (untrained), JEPA (trained)** — sanity check on val gaussian-3 (n=1000, K=8, seed-0 ckpt tkqjawa0/best.ckpt). Results: pixel-std 0.722, random-init 0.789, Mahalanobis 0.604, MAE-untrained 0.393, JEPA-trained 0.764. Random-init AUROC high because gaussian noise shifts embedding norms through any architecture; Gate 1B comparison requires ≥3 seeds + CIs before JEPA-vs-random-init conclusion can be drawn. Untrained MAE inverts (AUROC<0.5) due to norm_pix_loss: gaussian noise inflates per-patch variance, normalization shrinks the reconstruction target toward zero, and the untrained decoder's near-zero outputs look better on noisy patches — training the MAE removes this artifact.

[1.5a] **PixelMAE architecture: same ViT-Tiny encoder as VisionJEPA (d=192, 6 layers, 3 heads); lightweight decoder (enc_to_dec projection + 2-layer transformer at d=128 + pixel_proj); 75% random masking; norm_pix_loss=True** — identical encoder budget ensures MAE-vs-JEPA comparison isolates objective (pixel reconstruction vs latent prediction) rather than capacity.

[1.5a] **auroc() device fix: added device=all_scores.device to torch.arange call** — failure manifested on MPS when energy tensors returned on device; torch.arange defaults to CPU. Fix applied to src/eval/evaluate.py (ported code, not rewritten).

[1.5d] **Pooling for linear probe: mean over all N=144 patch tokens (no CLS token)** — VisionJEPA has no CLS token; adding one would change the pretrained architecture. Mean-pooling is standard for patch-only ViTs (DINOv2, MAE) and matches our Mahalanobis baseline's feature extraction. Logged as frozen for the probe transfer evaluation.

[1.5d] **Probe epoch budget: linear head 100 epochs, from-scratch 200 epochs** — from-scratch needs more steps to learn representations AND classify; giving it more epochs is conservative (avoids strawman). Both use AdamW; from-scratch uses cosine lr schedule with 10-epoch warmup (same schedule shape as JEPA pretraining). From-scratch lr swept over {1e-3, 3e-4, 1e-4} on formal val split; best lr reported per (n, seed) cell.

[1.5d] **Probe label fractions: n ∈ {40, 200, 400, 4000} = {1%, 5%, 10%, 100%} of the 4 000-image probe pool** — 10 classes × {4, 20, 40, 400}/class; stratified sampling seeded per (n, seed) pair. Probe pool = STL-10 labeled train minus formal val split (4 000 images, 400/class).

[1.5d] **Probe protocol LOCKED (primary): target encoder, mean pool, z-score, lr=3e-3, 200 epochs** — determined by probe_diag.py on seed-0, n=4000, val only. All future probe numbers use this protocol. Exhaustive ablation of pooling, encoder, normalisation, and schedule recovered only +3.0 pp (0.571 → 0.601) over the naive baseline; residual −9.9 pp gap to Gate 1B floor (≥0.70) is a representation quality issue, not a probe configuration issue. Probe protocol is frozen; the fix must come from training.

[1.5d] **Probe protocol LOCKED (tracked variant): target encoder, last-2-layer concat (penultimate + final layer mean-pooled, concatenated), z-score, lr=3e-3, 200 epochs** — gave the best structural improvement (+2.1 pp at n=4000, seed-0). Carried as a secondary variant in all future sweep tables; primary is target mean+zscore for interpretability.

[1.5e] **H2 confirmed — masking task too easy; next run uses harder masking** — probe-vs-epoch curve (seed-0, checkpoints ep30/60/90/120/150) peaked at ep120 (0.604) and was flat/declining at ep150 (0.600); Δ(ep120→150) = −0.004 < 0.005 pre-registered threshold. Target ≈ context encoder gap throughout (≤ 0.011 at all epochs), consistent with weak self-supervised signal from easy masking. Fix: one from-scratch seed-0 run with harder masking — target block scale (0.20, 0.25), context scale (0.75, 0.90). H1 (resume to 300 epochs) ruled out.

[1.6] **Hardmask adoption: REJECTED** — Pre-registered rule R1 (from HANDOFF.md §8): ADOPT iff n=4000 locked probe ≥ 0.62 AND eff_rank ≥ 96/192. Result on fw1out6d/epoch_0150.ckpt (val only, target+zscore+lr-sweep+200ep): n=4000 = 0.587, eff_rank = 189.2. Eff_rank passes; probe fails (0.587 < 0.62 by 3.3 pp). Hardmask 0.587 is statistically indistinguishable from the reference seed distribution {0.564, 0.582, 0.601} — pretext difficulty raised +41% with no transfer change. REJECT branch fires.

[1.6] **Gate 1B floor: revised to low-to-mid 60s for this constrained setup** — Literature calibration for small ViT (ViT-Tiny, 3M params), masked-prediction SSL (I-JEPA style), no color augmentation, STL-10 96px, 100k unlabeled, 150 epochs, 4k probe images: the plausible ceiling is ≈0.62–0.65. Evidence: (a) reference seed-0 best variant 0.601, hardmask seed-0 = 0.587 — both converge below 0.62 with different masking configs; (b) I-JEPA paper with 300M-param ViT-H achieves much higher, but that is a 100× capacity advantage; (c) STL-10 is genuinely hard at 96px from scratch without color aug (ImageNet-pretrained ResNet-50 gets ~80%+ with fine-tuning; from-scratch ViT-Tiny at ~63% is consistent with our reference); (d) the probe-vs-epoch plateau at ep120 with both configs rules out duration as a lever; (e) target ≈ context gap ≤ 0.011 at all checkpoints rules out EMA quality as the cause. **Claim therefore scoped to:** (i) low-label transfer gaps — pretrained probe beats from-scratch at n=40/200/400 in 3/4 cells; (ii) energy AUROC ~0.76–0.78 on gaussian noise (with trained > random-init and pixel-std baselines). NOT claimed: absolute accuracy matching supervised or larger SSL models. Floor revision does not require changing the architecture; it requires honest scoping.

[1.6] **Checkpoint-saving bug: pred_loss is not a valid "best" criterion for JEPA runs with increasing EMA momentum** — For fw1out6d (hardmask), pred_loss at epoch 2 (0.2738) was lower than at epoch 150 (0.2918) because the EMA target starts close to the context encoder (easy prediction) and diverges as momentum ramps from 0.996→1.0. The checkpoint saver incorrectly labeled epoch 2 as "best". For future training runs: save both epoch_N every ckpt_every AND the last checkpoint as "best"; do not rely on lowest pred_loss as the selection criterion for SSL runs with EMA warmup. The verdict for fw1out6d was correctly computed on epoch_0150.ckpt.

[1.6] **R1 protocol deviation — probe ran on epoch_0150 not best.ckpt (justified):** fw1out6d/best.ckpt was epoch 2 due to the checkpoint-saving bug. epoch_0150.ckpt was used instead. Deviation justified: epoch_0150 finals match W&B (loss 0.2933, pred 0.2918, eff_rank 189.2); this is the trained model. Robustness note: reference peak-vs-final probe delta is ≤0.004; even a +0.017 allowance leaves hardmask (0.587) below both the 0.62 threshold and the reference (0.601). REJECT stands under any plausible checkpoint correction.

[1.6] **Checkpoint policy — canonical checkpoint is epoch_0150 for all runs:** best-by-lowest-pred-loss is DEPRECATED across all models and seeds. The canonical checkpoint for all evaluation (probes, energy, terminal benchmark R3) is epoch_0150.ckpt. This is an evaluation-side convention only — do NOT modify trainer code before the remaining Phase-1 launches. If early stopping ever matters, a val-metric criterion will be designed and pre-registered.

[1.6e] **Gate 1B floor revision — APPROVED by Anuj, 2026-07-10**

**Background.** The original Gate 1B criterion (ii) required probe accuracy ≥ 0.70 at n=4000 labels. That floor was set before any measurement and is not literature-calibrated for this regime.

**Regime characterisation.** ViT-Tiny (~3M params), masked-latent self-supervised pretraining (I-JEPA-style), no color augmentation, 96px input, STL-10 scale (100k unlabeled pretraining pool, 4k probe images). The plausible ceiling for self-supervised linear transfer in this regime is low-to-mid 60s: prior I-JEPA results with ViT-H and ImageNet-scale pretraining achieve much higher, but that represents a 100× capacity and 1000× data advantage; this setting is intentionally constrained for the edge-deployability story.

**Measured evidence for revision.**
- (a) Duration lever exhausted: probe-vs-epoch curve on ref seed-0 plateaued at ep120 (0.604), declined slightly to 0.600 at ep150; Δ(ep120→ep150) = −0.004, below the pre-registered 0.005 threshold. More training does not help.
- (b) Masking difficulty lever pulled and measured: hardmask raised pretext pred_loss by +41% (0.207 → 0.292), near-full-rank representation (eff_rank 189/192). Locked-probe result on fw1out6d/epoch_0150: n=4000 = 0.587, statistically indistinguishable from the reference seed distribution {0.564, 0.582, 0.601}. The harder self-supervised task produced no transfer gain.
- (c) Seed decomposition (session 1.6e): pooled probe σ_probe ≈ 0.0017 (below 0.005 → encoder variance is real, not measurement noise). Reference 3-seed result: 0.582 ± 0.018 (mean ± σ_seed over training seeds, per-encoder means {0.600, 0.565, 0.579}). The ~3.7% seed-to-seed spread is the irreducible variance of this recipe/regime, not a fixable tuning problem.
- (d) Both levers available pre-R3 (duration + masking difficulty) have been pulled and neither closes the gap. The remaining shortfall to 0.70 reflects the regime ceiling.

**Revised Gate 1B (APPROVED — Anuj, 2026-07-10):**
- (i) UNCHANGED: per-type corruption AUROC paired-bootstrap CI on margin vs scratch excludes 0 (primary gate).
- (ii) REVISED: absolute probe accuracy REPORTED with full two-lever and seed-spread analysis; not gated at 0.70. Reported value: 3-seed mean 0.582 ± 0.018 (val, n=4000, locked protocol, epoch_0150, per-encoder means {0.600, 0.565, 0.579}). Low-label transfer story (pretrained > scratch at n=40/200/400) reported separately.
- (iii) UNCHANGED: pretrained encoder beats from-scratch in the majority of low-label cells; confirmed with fresh A3 scratch runs at R3.
- (iii) UNCHANGED: pretrained encoder beats from-scratch in the majority of low-label cells (3/4 positive at seed-0; multi-seed confirmed at R3).

**Claim scoping (cocktail-report style).** What survives: (a) latent prediction error as a usable anomaly energy — competitive with pixel-level and random-init baselines on most corruption types; (b) label-efficient transfer advantage at low label counts (n=40/200/400); (c) edge-deployable at ~3M params (Phase 3 claim). What does not: absolute linear-probe accuracy matching supervised or large-model SSL — not this setting, not this claim. Debt named explicitly: small ViT, no color augmentation, constrained pretraining scale.

[1.6h] **OOD diagnostics — pre-registered readings fired (Jul 10):**

D1 (random-init single-model SVHN AUROC): 0.477 ≈ 0.5 → **training-specific**. The inversion (0.09–0.13) is a property of the trained encoder, not an architectural/statistical confound. Previous benchmark value (0.118) was a model-mismatch artifact.

D2 (loader parity): all three loaders (val, SVHN, CIFAR-10) use identical pipeline — Resize(96,BICUBIC)→CenterCrop(96)→Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]). PASS. Per-channel mean differences are real distributional signal (SVHN blue-channel shift, CIFAR-10 positive offset).

D3 (Mahalanobis on frozen target-encoder features, fit on 4k probe pool): SVHN=0.985, CIFAR-10=0.855. Context-encoder version (fit on 1k val): SVHN=0.998, CIFAR-10=0.949. **Target-feats SVHN=0.985 ≥ 0.9 → two-readout claim fires.** Pre-registered: Mahal-on-target-features is an additional OOD readout for R3, alongside energy AUROC; both reported from the same frozen encoder with no additional training.

D4 (Spearman energy vs Laplacian variance): pooled val+SVHN rho=0.770, val-only rho=0.328, SVHN-only rho=0.613. **|rho|=0.770 ≥ 0.5 → prediction-difficulty mechanism supported.** JEPA energy tracks high-frequency content / sharpness; SVHN digit patches have lower Laplacian variance → lower energy → trained model ranks them as more "normal" than they are.

D5: energy histograms saved to reports/figs/ (3 PNGs: jepa_ref_s0, random_init, mahal_target).

**Pre-registered claim language — two-readout branch fires:**
"Latent prediction error detects corruption in most types (11/15 vs pixel-std baseline). The same frozen encoder's feature density detects semantic domain shift (SVHN AUROC 0.985 via Mahalanobis on target-encoder features, probe-pool fit). Energy alone inverts on cross-domain OOD — mechanism: prediction difficulty decreases for low-complexity inputs (Spearman rho=0.770 between energy and Laplacian variance, pooled val+SVHN). The two readouts are complementary and operate on the same frozen encoder with no additional training."

**Claim scope clarification post-diagnostics:** the inversion is training-specific (D1), loader-artifact-free (D2), mechanistically supported (D4), and complemented by a working density readout (D3). The scoped-down framing (energy = corruption critic only) is NOT used — the two-readout framing is the pre-registered binding outcome given D3 SVHN ≥ 0.9.

[1.6h] **random_init baseline in terminal_benchmark.py is a model-mismatch comparison** — `random_init_energy` instantiates a fresh VisionJEPA each call; Stage 1 clean energies come from Model_A, each Stage 2 corruption cell from a different Model_B. AUROC measures inter-model energy variance, not intra-model response to corruption. Numbers swung from 0.456→0.132 for gaussian_noise between equivalent runs. Fix for R3: instantiate ONE random model in Stage 1 and reuse across all Stage 2 cells. Gate 1B criterion (i) is evaluated against pixel_std only; random_init column treated as decorative in the val table.

[1.6h] **glass_blur and fog required 3-line .venv patch to run** — imagecorruptions library incompatible with current scikit-image (multichannel kwarg removed → channel_axis=-1) and NumPy 2.0 (np.float_ removed → np.float64). Patch applied to .venv in place; not vendored into source. If .venv is rebuilt this patch must be reapplied. R3 must verify both types run before touching the test set.

[1.6h] **R3 RUNSHEET — FOR HUMAN APPROVAL (pre-registered before test set is opened)**

*Requires two approvals: (1) Gate 1B floor revision (DECISIONS.md §1.6e); (2) this runsheet.*

**Pre-R3 checklist (executor before `--unlock_test`):**
1. Verify .venv patch persists for glass_blur + fog; re-apply if needed.
2. Fix random_init baseline: instantiate ONE model in Stage 1, reuse across all Stage 2 + Stage 3 cells.
3. Add Mahal-on-target-features as second OOD readout in Stage 3 (fit on probe pool, target encoder — D3 baseline, pre-registered above).
4. Run from-scratch probe: 3 probe seeds × n∈{40,200,400,4000} on formal val split, full budget (200ep, lr-sweep); store for R3 gap column. (Not blocking — if not complete, R3 gap column is blank with note.)
5. Wire test-loader path into `terminal_benchmark.py main()` (currently `--split val` hardcoded; `--unlock_test` guard exists but test loader not implemented).
6. Smoke-test amended harness on val (no `--unlock_test`), confirm exit 0.

**Model list:**
- ref_s0: runs/tkqjawa0/epoch_0150.ckpt
- ref_s1: runs/lbd900za/epoch_0150.ckpt
- ref_s2: runs/gommvdgc/epoch_0150.ckpt
- hardmask_s0\*: runs/fw1out6d/epoch_0150.ckpt (single seed, REJECTED lever, labeled throughout)
- mae_trained: runs/eoofx7fk/epoch_0150.ckpt
- Baselines: pixel_std; random_init (fixed, single model); Mahal-on-context-feats (val-fit, 1k); Mahal-on-target-feats (probe-pool-fit, 4k, new D3)

**Frozen corruption list (15/15 types × 5 severities):**
gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression

**OOD stage — both readouts per model per set:**
- Readout A: JEPA energy AUROC (existing)
- Readout B: Mahal-on-target-features AUROC (probe-pool fit, D3 pre-registered)
- Sets: SVHN test, CIFAR-10 test (cached, unchanged)

**Probe grid:**
- Protocol: locked (target mean+zscore, lr-sweep {3e-3,1e-3,3e-4}, 200ep, 3 probe seeds per cell)
- All cells fresh at R3 — no injected numbers
- Scratch comparator: 3 probe seeds, same val split, full budget — provenance stated in report
- n ∈ {40, 200, 400, 4000}

**Headline tables in reports/terminal_test.md:**
1. Corruption AUROC (per-type, mean over severities): ref_mean±std_training_seed, hardmask_s0\*, pixel_std, random_init (fixed), mae_trained; margin column (trained − pixel_std)
2. Corruption detail: ref_s0 per-severity with bootstrap CIs
3. OOD (both readouts): energy AUROC + Mahal-on-target-feats AUROC × {SVHN, CIFAR-10} × all models
4. Probe grid: mean±σ_probe per cell; gap vs scratch (A3 fresh runs required; gap column must not be blank or populated from 1.5d numbers)
5. Wall-clock summary

**Reference row:** mean ± std_training_seed over {ref_s0, ref_s1, ref_s2} at epoch_0150.
**hardmask_s0\*** row: labeled "single-seed, REJECTED (R1)" in every table.

**Binding claim language (two-readout branch, D3 SVHN=0.985 ≥ 0.9 fires this):**
"Latent prediction error detects complexity-adding corruption (noise-family AUROC 0.74–0.80 at test, ≥0.92 at maximum severity) and inverts on complexity-reducing corruption; it exceeds the pixel-variance baseline on 8/15 types (3 clearly). The same frozen encoder's feature density detects semantic domain shift (mahal_tgt SVHN 0.986, CIFAR-10 0.864 at test, probe-pool fit, no additional training) and is strongest on precisely the energy-inverted types. Inversion mechanism: prediction difficulty, Spearman ρ=0.770 (pooled val+SVHN). Two readouts, one encoder."
[Claim approved — Anuj, 2026-07-17. Slots filled from terminal_test.md R3 run-2. No post-hoc reframing.]

**Wall-clock projection (8k test vs 1k val):**
Stage 2: ~2626s × 8 = ~21,000s. Stage 3: ~580s. Stage 4: ~400s. Stage 1: ~120s.
Total: ~22,000s (~6h). **Recommendation: caffeinate overnight.**

**R3 invocation (DO NOT RUN until both approvals + checklist complete):**
```
caffeinate -is uv run python scripts/terminal_benchmark.py \
  --ref_ckpts runs/tkqjawa0/epoch_0150.ckpt \
              runs/lbd900za/epoch_0150.ckpt \
              runs/gommvdgc/epoch_0150.ckpt \
  --hardmask_ckpt runs/fw1out6d/epoch_0150.ckpt \
  --mae_ckpt runs/eoofx7fk/epoch_0150.ckpt \
  --split test --unlock_test \
  --out reports/terminal_test.md
```
**(A1 CRITICAL: `--split test` not `--split val`; corrected from prior draft. A prior chat-recap line omitted gommvdgc from --ref_ckpts — caught pre-launch. The R3 command is read from THIS runsheet ONLY, never from a chat recap.)**

**R3 launch ritual (REQUIRED before running):**
1. Read the command from this runsheet — never from a chat summary or session recap.
2. Count 5 checkpoint paths in the command: tkqjawa0, lbd900za, gommvdgc, fw1out6d, eoofx7fk. All 5 must be present.
3. Confirm `--split test --unlock_test` (not `--split val --unlock_test`).
4a. Within 2 min: manifest prints `split=test n=8000`.
4b. Pre-launch: `uv run python scripts/patch_imagecorruptions.py` prints ALREADY_PATCHED for BOTH patches (glass_blur and fog).
4c. Before sleeping: Stage-2 log shows glass_blur completed (not SKIPPED).
4d. Morning: confirm fog completed (not SKIPPED).
5. Do not interrupt after Stage 2 begins unless a crash occurs.

---

[1.6i] **Runsheet amendments A1–A9 — IMPLEMENTED (2026-07-10)**

A1 CRITICAL: `--split test --unlock_test` (not `--split val --unlock_test`). Test loader wired into harness; split guard enforces mutual requirement. Negative guard tests added (tests/test_terminal_benchmark.py).

A2: Mask RNG already seeded (seed=0 in all energy calls). Residual |Δ| <0.001 between equivalent runs attributed to MPS non-determinism, not mask jitter.

A3: Scratch comparator: 3 training seeds × lr-sweep {1e-3,3e-4,1e-4} × 200ep. Launch command: `caffeinate -is uv run python scripts/train.py --config configs/phase1_ref.yaml --seed <S>` repeated per seed with appropriate lr override. (Not yet launched this session — tonight's task.)

A4: Context-feats val numbers labeled FIT-ON-EVAL-SET throughout harness output. Target-feats probe-pool = primary density readout (mahal_tgt in all tables).

A5: Binding OOD claim: SVHN=0.985, CIFAR-10=0.855 (mahal_tgt, D3, val); SVHN=0.998, CIFAR-10=0.949 (mahal_ctx, val-fit).

A6: mae_untrained restored to Stage 2 + Stage 3.

A7: Energy dumps: clean_*.npy arrays + ood_auroc_*.json written to reports/energy_dumps/ after Stage 3.

A8: scripts/patch_imagecorruptions.py — idempotent patcher. glass_blur (multichannel→channel_axis=-1) now applied by script; fog (np.float_→np.float64) was already patched. Script re-run confirms PATCHED (1) / ALREADY_PATCHED.

A9: Reference probe pinned to 0.582±0.018 (ref_s0 n=4000 mean±σ_training_seed from b803je03m).

[1.6i] **Two-sided readout validation (bsuvue5wt) — FAIL (2026-07-10)**

Pre-registered PASS condition: (A) ≥4/6 inverted types reach two-sided AUROC ≥0.60 AND (B) each detected type stays within 0.05 of one-sided.

Result: Condition A PASS (5/6: defocus=0.671, glass=0.620, motion=0.633, fog=0.861, contrast=0.961; frost=0.590 fails). Condition B FAIL (all 5 detected types exceed 0.05: gaussian_noise |Δ|=0.224, shot_noise=0.204, impulse_noise=0.190, brightness=0.113, jpeg_compression=0.073).

Root cause (corrected): clean-val mean ts=0.789 ≈ E|Z|=√(2/π)≈0.798 — this is *healthy* probe-pool calibration (the val distribution and probe-pool are well-matched), NOT the failure. The failure = the two-sided fold maps the clean low-energy tail onto *high* two-sided scores. Detected-type corruptions push energy *up* from μ_ref (correct direction one-sided); but the fold also pulls the clean low-energy images up to similar magnitudes, destroying rank separation where one-sided detection is only marginal (AUROC 0.64–0.84). Inverted types push energy far below μ_ref → fold pushes ts very high → AUROC high (5/6 pass).

Per pre-registered rule (FAIL on Condition B): two-sided readout NOT integrated into harness. Report attempt only. Primary OOD readout: mahal_tgt (D3, probe-pool-fit). One-sided JEPA energy: corruption readout.

**Binding scope:** two-sided readout may NOT be computed from R3 test dumps. Any future test-side use of a two-sided formulation requires a new pre-registration and explicit human approval before the test set is opened.

[1.6o] **Scratch comparator recipe underfit — Branch B2 fires (2026-07-12)**

s0_n4000 best val_acc = 0.580 ≤ 0.60. Recipe diff confirmed:
- A3 (run_scratch_comparator.py): batch=min(256,n), no data augmentation, ~3200 optimizer steps at n=4000.
- 1.5d (probe_sweep.py): batch=128, RandomResizedCrop(96,scale=(0.5,1.0))+RandomHorizontalFlip, ~6400 steps at n=4000.
- Concrete optimization disadvantage confirmed → Branch B fires.
- B1 vs B2: recipe-fixed rerun of 9 n=4000 cells (~5-6h) cannot complete by ~20:00 → Branch B2.

**Binding consequence:** R3 tonight WITHOUT scratch. Gap column in terminal_benchmark.py Stage 4 is BLANK-WITH-NOTE. Gate 1B(iii) evaluated post-hoc on val with recipe-fixed scratch. **Test set NEVER reopens for scratch.** Scratch recipe fix required before any future comparison.

**Pre-registered Gap definition (binding for eventual fix):** JEPA 3-ref-seed mean at n (mean of per-encoder means over probe seeds) minus scratch 3-seed mean at n (mean of best-lr val_acc over training seeds). Mean-vs-mean, not paired per training seed.

[1.6p] **R3 run-1 ABORTED — interruption exception invoked (2026-07-14)**

R3 run-1 (overnight Jul 12→13) aborted at ~9/75 Stage-2 cells: kIOGPUCommandBufferCallbackErrorOutOfMemory at shot_noise sev2. Interruption exception invoked under ritual step 5 (crash-equivalent; integrity doubt due to post-OOM non-monotone shot values). All run-1 Stage-2 numbers discarded; NUMBERS NOT USED IN ANY REPORT.

Root cause: Stage-2 evaluation loop sent full 8k tensor (884 MB) to MPS per model per cell; no cleanup between cells; MPS allocator exhausted at ~cell 10.

Infra fix (throughput-only, eval logic FROZEN): chunked GPU evaluation (chunk=1000), explicit `del + torch.mps.empty_cache()` after each cell, pixel_std moved to CPU-only.

Rule: infra fixes require val parity gate (energy max|Δ|≤1e-4, AUROC identical to 3 decimals) and zero eval-logic change before relaunching. Parity gate is a hard block; failure = STOP.

[1.6q] **R3 run-1 VOID IN ENTIRETY — Item-5 diagnostic (2026-07-16)**

Pre-registered diagnostic on run-2 completion: compare run-2 gaussian cells against run-1 pre-OOM values.
Result: gaussian sev1 Δ=0.168 (run-1=0.747, run-2=0.579). Exceeds expected ±0.05 noise band.
Interpretation: run-1 Stage-1 clean energies were also corrupted by the same MPS async dispatch issue (ref_s1 clean mean 0.3865 vs run-2's clean 0.2901 after sync fix). Non-monotone gaussian values were not solely post-OOM corruption — the entire run-1 was computing against stale clean baselines. **Run-1 VOID IN ENTIRETY** (not just Stage-2).

[1.6q] **MPS silent Stage-1 corruption — ref_s1 (lbd900za) in R3 run-2 (2026-07-16)**

Symptom: ref_s1 (lbd900za) clean test mean 0.2711 on test set (expected ~0.219 from independent run-1 measurement). All Stage-2 ref_s1 AUROCs near zero (0.07–0.19).

Root cause: MPS async dispatch race condition. `.cpu()` called on result of `image_energy()` returned before MPS GPU computation completed. For large K=8 × 8000-image batches, the returned CPU tensor contained stale or partial data.

Fix applied: `torch.mps.synchronize()` before `.cpu()` in `_jepa_energy` (terminal_benchmark.py). Verified: recomputed clean mean = 0.2190 (Decision 1=A validation PASS, within [0.216, 0.222]).

Consequence: ref_s1 Stage-2 rows VOID-INFRA. ref_s1 Stage-3 OOD recomputed (SVHN=0.1277, CIFAR-10=0.5048). ref_s0 Stage-2 is canonical (unaffected; corruption comparison loop uses clean_e[label] which was fetched before Stage-2 loop — only the clean energy tensor for the specific run had the race; ref_s0 clean mean 0.2169 was normal and agreed with prior val measurements).

Rule for future runs: always add `torch.mps.synchronize()` before any `.cpu()` call that follows MPS computation in evaluation scripts. If clean mean drifts >0.005 from prior measurements, treat as corrupted and STOP.

[1.6q] **Gate 1B(iii) evidence — scratch v2 (recipe-fixed) result (2026-07-17)**

Scratch v2 (batch=128, RandomResizedCrop+HFlip) — 36 runs, 3 seeds × 4 n × 3 lr, 3.94h:
- n=40: scratch mean 0.2487 ± 0.0074
- n=200: scratch mean 0.3857 ± 0.0055
- n=400: scratch mean 0.4260 ± 0.0104
- n=4000: scratch mean 0.6480 ± 0.0118

Binding gaps (JEPA val − scratch v2):
- n=40: +0.028
- n=200: +0.004
- n=400: +0.010
- n=4000: −0.065

~~**Gate 1B(iii): PASS**~~ — STRUCK. Replacement per 1.6r gate-language correction:

**Gate 1B(iii) evidence:** point gaps positive in all three low-label cells (+2.8/+0.4/+1.0pp) and −6.5pp at n=4000 to recipe-fixed scratch. Combined seed spreads (RSS of JEPA training-seed σ and scratch-v2 training-seed σ): n=40: σ_J=0.014, σ_S=0.007, RSS=0.016; n=200: σ_J=0.025, σ_S=0.006, RSS=0.026; n=400: σ_J=0.019, σ_S=0.010, RSS=0.022; n=4000: σ_J=0.018, σ_S=0.012, RSS=0.022. Only n=40 gap (+0.028) exceeds combined spread (0.016); n=200/400 gaps (+0.004/+0.010) within noise. **Gate decision = human, with spreads in view.**

PD5 check: s0_n4000=0.635 vs 1.5d target 0.636±0.02 → WITHIN BAND. Recipe fix reproduces 1.5d result; A3 underfit confirmed.

**Strawman guard arc (named paragraph for phase1.md):** A3 scratch (no aug, large batch) produced a +0.3pp gap at n=4000 that appeared to support the low-label claim. Pre-registered rule B2 disqualified A3 before R3: result favorable, recipe not matched, must rerun. Scratch v2 (batch=128, aug) confirms the A3 favorable number was an artifact of recipe underfit — at n=4000 the recipe-fixed baseline wins by 6.5pp. The reported gap evidence uses v2. A3 numbers appear in the report with explicit underfit label and are not used in any claim.

[1.6i] **Approvals ledger**

**Gate 1B floor revision APPROVED — Anuj, 2026-07-10.** Canonical value: 0.582±0.018 (3-seed mean±sample-σ, per-encoder means {0.600, 0.565, 0.579}, val n=4000 locked protocol epoch_0150). Replaces stale 0.583±0.016. Gate 1B(ii): reported, not gated.

**R3 runsheet APPROVED AS AMENDED (A1–A9) — Anuj, 2026-07-10.** Amendments A1–A9 incorporated; two-sided FAIL recorded; harness smoke-tested (exit 0); scratch loop (A3) launched tonight before R3.

---

[1.6s] **Band adjudication — val→test shift declared systematic (2026-07-15, approver=Anuj)**

Band check at n=4000 fired FAIL for ref_s0 (|Δ|=0.044) and ref_s2 (|Δ|=0.041); ref_s1 PASS (|Δ|=0.022). Post-hoc adjudication: the drop is systematic val-selection optimism + split shift — uniform −2..−4pp across all models and cells, direction-consistent with the corruption downshift (val n=1000 vs test n=8000; LR selected on val). The band tolerance (±0.03) was noise-calibrated for within-split MPS variation; it was not designed to absorb a systematic split-shift of this size. Reviewer's calibration error on record. Adjudication: band FAILs do not indicate a protocol violation or data leak — they measure a systematic effect of val-selection optimism + distributional shift that is a measured finding, not a tuning problem.

**How to apply:** phase1.md reports val and test probe grids side by side; the val→test gap (−2..−4pp) is named as "val-selection optimism" and reported as a finding. Band check result included verbatim with adjudication note. No recomputation triggered.

[1.6s] **Gate 1B(iii): PASS — human ruling (2026-07-15, approver=Anuj)**

"Gate 1B(iii): PASS — per the pre-registered criterion (pretrained > scratch point-estimate in all three low-label cells vs recipe-fixed scratch v2). Context carried in report: +2.8pp at n=40 exceeds combined spread; +0.4/+1.0pp at n=200/400 within noise; −6.5pp at n=4000 conceded to end-to-end. Val-side per PD2."

Evidence base: scratch v2 (batch=128, RandomResizedCrop+HFlip, 3 seeds × 4 n × 3 lr, status=ok for all 36 cells, epochs_completed=200). Combined seed spreads: n=40 RSS=0.016 < gap +0.028; n=200 RSS=0.026 > gap +0.004; n=400 RSS=0.022 > gap +0.010; n=4000 scratch wins by 6.5pp (outside spread). Note: "Gate 1B(iii): PASS [executor]" was struck in 1.6r (1.6r Item 2 gate-language correction); this entry records the human ruling that replaces it.

[1.6s] **Gate 1B(i): NOT MET as operationalized — human ruling (2026-07-15, approver=Anuj)**

"Gate 1B(i): NOT MET as operationalized (3/15 types clearly above pixel_std at test) — ruling confirmed by the human after reviewing the per-type table; superseded in narrative by the system-level finding (mahal_tgt on the same encoder: defocus 0.937 / contrast 0.977 / fog 0.920 on precisely the energy-inverted types)."

Background: pre-registered criterion required paired-margin bootstrap CIs excluding zero for majority of win types. Per-image corrupted energy arrays not dumped → CIs uncomputable post-hoc. Win count = 8/15 point wins, 3 clearly above (impulse +5.5pp, brightness +8.9pp, jpeg +8.2pp). NOT MET as originally operationalized; system-level story carried by the mahal_tgt complementarity finding.

[1.6s] **Stage-4b parity check — exact-4dp result (2026-07-17, superseded by 1.6t ruling)**

Full 48-cell eval=val run of probe_on_test.py: 6/48 PASS at round(.,4). Character: MPS probe-training non-determinism, no systematic bias, |Δ| ≤ 0.028 (max at small n), n=4000 max |Δ|=0.008. Status at end of 1.6s: PROVISIONAL, pending human ruling on parity standard. Superseded by 1.6t mean-level adjudication below.

[1.6t] **Commit-message accuracy — F10 scope extended (2026-07-17)**

Session 1.6s proposed commit message "1.6s: Stage-4b parity certified + gate rulings persisted + phase1.md finalization" while the parity check had FAILED (provisional status). This is an F10 violation: commit messages state what happened, not what was intended. Rule persisted: commit messages must match the actual state at commit time; conditional outcomes ("certified" when FAIL-pending) are prohibited. Applied to: final 1.6s commit message was not issued (caught in-session); 1.6t catch-up commit corrects the record.

[1.6t] **Mean-level parity adjudication — PASS; exact-4dp struck as spec error (2026-07-17, approver=Anuj)**

Exact-4dp parity was a spec error for a soft-determinism (MPS) pipeline: reviewer's error on record. No hand-tuned per-cell tolerance adopted. Mean-level rule pre-registered before computation (DECISIONS.md 1.6t, this entry), then applied.

**Rule (binding):** PASS iff (a) every |Δmean| ≤ 2·σ_combined where σ_combined = sqrt(σ_stored² + σ_new²)/√3, AND (b) mean of 16 signed deltas ≤ 0.005 in absolute value.

**16-row table (per-(model,n) 3-seed means, arithmetic from 48-cell diff):**

| Model | n | mean_new | mean_stored | Δmean | σ_combined | 2σ | (a) |
|---|---|---|---|---|---|---|---|
| ref_s0 | 40 | 0.2840 | 0.2937 | −0.0097 | 0.00835 | 0.01671 | PASS |
| ref_s0 | 200 | 0.4190 | 0.4147 | +0.0043 | 0.02029 | 0.04058 | PASS |
| ref_s0 | 400 | 0.4443 | 0.4550 | −0.0107 | 0.00692 | 0.01383 | PASS† |
| ref_s0 | 4000 | 0.6020 | 0.6030 | −0.0010 | 0.00082 | 0.00163 | PASS |
| ref_s1 | 40 | 0.2740 | 0.2690 | +0.0050 | 0.01031 | 0.02062 | PASS |
| ref_s1 | 200 | 0.3623 | 0.3653 | −0.0030 | 0.01194 | 0.02388 | PASS |
| ref_s1 | 400 | 0.4160 | 0.4170 | −0.0010 | 0.01471 | 0.02941 | PASS |
| ref_s1 | 4000 | 0.5653 | 0.5643 | +0.0010 | 0.00138 | 0.00275 | PASS |
| ref_s2 | 40 | 0.2647 | 0.2683 | −0.0037 | 0.01602 | 0.03204 | PASS |
| ref_s2 | 200 | 0.3917 | 0.3897 | +0.0020 | 0.01826 | 0.03651 | PASS |
| ref_s2 | 400 | 0.4407 | 0.4357 | +0.0050 | 0.00884 | 0.01768 | PASS |
| ref_s2 | 4000 | 0.5820 | 0.5803 | +0.0017 | 0.00156 | 0.00313 | PASS |
| hardmask | 40 | 0.2990 | 0.2950 | +0.0040 | 0.00408 | 0.00817 | PASS |
| hardmask | 200 | 0.4287 | 0.4180 | +0.0107 | 0.01629 | 0.03257 | PASS |
| hardmask | 400 | 0.4780 | 0.4813 | −0.0033 | 0.01479 | 0.02958 | PASS |
| hardmask | 4000 | 0.5903 | 0.5897 | +0.0007 | 0.00410 | 0.00820 | PASS |

*† Closest pass: ref_s0 n=400, ratio |Δ|/2σ = 0.773.*

Condition (a): **16/16 PASS.** Condition (b): signed-delta sum = +0.0020, mean = **+0.000125**, |+0.000125| ≤ 0.005 → **PASS.**

**MEAN-LEVEL PARITY CHECK: PASS.** Test probe grid promoted to CANONICAL. Phase1.md §7 finalized.

[1.6t] **Claim-language deviation log — "most types" → "8/15 types" (2026-07-17)**

The filled binding claim replaced template language "most types (N/15" with "8/15 types" — a template edit beyond fill-slots. Rationale accepted post-hoc: at N=8/15, "most" (>7.5) is technically true but would overclaim under the clearly-above standard (only 3 types clearly above at >5pp margin); the numeral-only form is more precise and less flattering to the system. Deviation logged; sentence stands. Rule going forward: fills only; any template edit requires same-session log entry in DECISIONS.md.

[1.6u] **F7 — fp32 on MPS; bfloat16 autocast is a no-op for most MPS ops (2026-07-18)**

Phase 1 training config specifies `use_amp: true` and loop.py calls `torch.autocast(device_type='mps', dtype=torch.bfloat16)`. However, MPS falls back to fp32 for most operations — bfloat16 autocast on MPS does not reduce numerical precision the way CUDA bfloat16 does. Practical effect: training runs in fp32. DECISIONS.md entries at [1.3] and [1.6] that state "bfloat16 AMP" referred to the config intent, not the actual numerical precision. Methods section of phase1.md updated to "fp32 on MPS" per this ruling. Config `use_amp: true` retained as-is (the autocast call is harmless).

[1.6u] **Methods-transcription rule — F2-class violation logged (2026-07-18)**

§3 of phase1.md was initially composed partly from memory (patch 12×12/64 tokens claimed; correct value from code is 8×8 patch, 12×12 grid = 144 tokens). This is an F2 violation: methods sections must be transcribed from config files and model source, never from memory. Rule persisted: for any future methods section, read configs/*.yaml and grep model source before writing. This instance recorded; §3 corrected in 1.6u.

[1.6u] **D2 — Two-axis conflation in §1 and §4.1 (2026-07-18)**

§1 and §4.1 used a single "bucket" scheme that conflated Axis 1 (absolute AUROC vs chance) with Axis 2 (margin vs pixel_std baseline). Specific errors: (i) shot_noise (AUROC 0.760, above-chance) was listed under "below or inverted" because it is a point-loss to pixel_std (−0.6pp); (ii) fog (0.078), contrast (0.019), frost (0.249) were listed as "borderline wins" without noting that both ref_s0 and pixel_std are below-chance on these types — the positive margin is between two inverted scores. §1 and §4.1 revised to name Axis 1 and Axis 2 explicitly. §9 claim language revised in 1.6v with human-approved complexity framing (see binding claim entry above). Deviation D2 complete: **approver=Anuj, 2026-07-17**. Rationale: template conflated detection with baseline comparison; three of eight point-wins occur on inverted types.

[1.6w] **phase1.md FINAL — dual sign-off on record (2026-07-18)**

Reviewer sign-off granted (external reviewer, Jul 17) after corrections C1–C9 applied (1.6u) and claim approved (1.6v). Human FINAL approval: Anuj, 2026-07-17. phase1.md footer updated from FINAL-PENDING-SIGNOFF to FINAL. No further edits to phase1.md permitted without both approvals being re-opened. Canonical export: reports/review_export/phase1.md.

---

[2.0] **Phase-2 pre-made decisions — persisted pre-launch (PD10/PD11 + reviewer, 2026-07-21)**

**(a) Warm-start checkpoint:** `runs/tkqjawa0/epoch_0150.ckpt` — canonical Phase-1 production checkpoint (seed 0, 150 epochs, eff_rank 175.3). Used as initialisation for encoder B (warm-start experiment). No other Phase-1 checkpoint is eligible.

**(b) Masking:** reference config (`configs/phase1_ref.yaml` block masking: target scale 0.15–0.20, context scale 0.85–1.00). Hardmask was R1-rejected and is not revisited without new pre-registered evidence showing a transfer benefit on aerial data.

**(c) Resolution:** 96×96, patch_size=8 → 144 tokens. Unchanged from Phase 1 for harness continuity and the Phase-3 deployability story (edge inference budget). Revisit only with documented evidence that a different resolution materially improves OOD AUROC on the val quarantine holdout.

**(d) Checkpoint saver fix — LANDS NOW, pre-registered (2026-07-21):** `best.ckpt` saving REMOVED ENTIRELY from `src/loop.py`. Rationale: `fw1out6d` evidence — EMA warmup causes pred_loss to be lowest at epoch 2, so the saver incorrectly labeled that as "best"; the canonical checkpoint was always `epoch_0150.ckpt`. Policy going forward: every-10-epoch periodic saves + final-epoch canonical (explicit save if `epochs % ckpt_every != 0`). Unit test added: `tests/test_no_best_ckpt.py` asserts no `best.ckpt` is produced after a 2-epoch run. train.py updated to unpack single return value.

[2.0] **AID ruling — deferred, pre-registered pool-enlargement (2026-07-21)**

Phase-2 primary claims run on RESISC45 alone (train pool = 22,400). AID (~10k images, 30 classes) is a deferred, pre-registered pool-enlargement comparison contingent on manual download. Trigger: if/when AID is downloaded and extracted to data/aid/, a new encoder is trained on RESISC45+AID pool and compared to the RESISC45-only encoder on the SAME pre-registered val metrics (quarantine AUROC + 40-class probe). No eval split ever contains AID images. AID is not a blocker for Phase-2 primary claims. Status: NOT DOWNLOADED (manual download from http://www.lmars.whu.edu.cn/prof_web/zhongyanfei/e-code.html required).

[2.1] **Pre-registered readings for 2.2/2.3 — committed BEFORE encoder A launches (2026-07-21)**

**Encoder A:** scratch-on-aerial, `configs/phase2_scratch.yaml`, 150 epochs, seed 0.
**Encoder B:** warm-start from `runs/tkqjawa0/epoch_0150.ckpt`, identical config/budget, seed 0.

**Gate 1A (per Phase-1 convention):** eff_rank ≥ 89% of d=192 = 171/192, spread and variance healthy, no collapse. Applies to both A and B.

**Comparison metric (primary, binding):** unseen-class energy AUROC — quarantine 5 classes as anomalies vs 40 normal val classes as in-distribution. Measured on VAL split only. Uses `image_energy` (K=8, ref_s0 collator settings) from the trained target encoder.

**Comparison metric (secondary):** locked-protocol linear probe on 40-class val (target encoder, mean-pool, z-score, lr-sweep {3e-3,1e-3,3e-4}, 200ep, 3 probe seeds, n=4000).

**Pre-registered decision rules (binding, fire after 2.3):**
- B_AUROC > A_AUROC + 0.03 → warm-start transfers; B becomes production candidate; report advantage explicitly.
- |B_AUROC − A_AUROC| ≤ 0.03 → equivalent; A is production (cleaner story: no cross-domain pretraining dependency).
- A_AUROC > B_AUROC + 0.03 → negative transfer; report honestly; A is production.

**Multi-seed rule (standing protocol):** ≥3 seeds required before any resume/report claim. The 2.2/2.3 launches are seed 0 only. Seeds 1 and 2 launch after Gate 1A confirms the recipe works (session 2.4 or later).

[2.0] **Phase-2 split freeze — RESISC45 + AID (design committed before data in view, 2026-07-21)**

Dataset: NWPU-RESISC45 (31,500 images, 45 classes × 700 images, 256×256).

**Quarantine (unseen-anomaly holdout):** 5 classes held out entirely from train/val/test of normal classes. Used only as OOD anomaly examples at Phase-2 terminal evaluation. Classes: `airplane`, `storage_tank`, `harbor`, `thermal_power_station`, `ship`. Quarantine list committed as `data/splits/resisc45_quarantine.json`.

**Normal-class splits:** remaining 40 classes (40 × 700 = 28,000 images); stratified 80/10/10, seed=0.
- train: 40 × 560 = 22,400 images (pretraining + probe pool)
- val: 40 × 70 = 2,800 images (committed; used for gate and probe selection)
- test: 40 × 70 = 2,800 images (**SEALED at creation; opens once at Phase-2 terminal session**)

Split index files committed: `data/splits/resisc45_train_idx.json`, `resisc45_val_idx.json`, `resisc45_test_idx.json` (lists of relative file paths). No split choice is made with data in view — this design entry precedes any download.

**AID:** ~10,000 images, 30 classes, joins the unlabeled pretraining pool only. No AID class appears in any evaluation split. AID images are never labeled for probe or anomaly tasks.

**Single-open test discipline:** `resisc45_test_idx.json` is sealed at creation. Terminal benchmark reads it once (Phase-2 session 2.4+). No evaluation script reads the test split before that session. Gate and probe selection use val only.

[2.2] **Gate 1A correction — no numeric rank bar (2026-07-21)**

The [2.1] entry "eff_rank ≥ 89% of d=192 = 171/192" is **STRUCK**. Gate 1A is a health-signature gate with no numeric threshold. Correct reading: eff_rank well off the collapse floor, mean-variance ~0.99, spread stable across training, no NaNs, loss curve sane (monotone-ish descent). The Phase-1 observations (eff_rank 172.6–175.3 across seeds) are empirical context for what "well off floor" looks like on this architecture — not a pass/fail criterion. Gate decision = human in all cases.

[2.2] **Warm-start verification — CLEAN (2026-07-21)**

`runs/tkqjawa0/epoch_0150.ckpt` loads into the Phase-2 VisionJEPA model (img_size=96, patch_size=8, d=192, 6 enc layers, 3 pred layers) with strict=True: 192/192 keys, 0 missing, 0 unexpected. Architecture compatibility confirmed. Checkpoint top-level key is `model_state` (not `model`). Encoder B warm-start is unblocked.

Launch mechanism: `--warm-start` flag (added to `scripts/train.py` 2026-07-21) loads model weights only; optimizer, scheduler, and epoch counter reset to zero. Distinct from `--resume` (which continues from checkpoint epoch with full optimizer state).

[2.2] **MIT LICENSE — approved and written (2026-07-21)**

MIT License, © 2026 Anuj Shah. Approved by Anuj (Jul 18). Written to `LICENSE`.
