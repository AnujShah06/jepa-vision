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
