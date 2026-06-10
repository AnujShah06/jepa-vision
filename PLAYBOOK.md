# JEPA Vision Extension — Execution Playbook

**From recipes to pixels to silicon: I-JEPA-mini → aerial imagery → edge deployment**

A step-by-step playbook designed to be executed by you + an LLM assistant, session by session, from scratch. Companion to the Cocktail-JEPA project; reuses its research skeleton (masked latent prediction, EMA target, anti-collapse diagnostics, energy-as-critic evaluation, multi-seed protocol, frozen-encoder transfer).

---

## How to use this document

Each phase is broken into numbered **steps**. Each step states: what to build, what "done" looks like, and what to log. **Decision gates** (marked ⛔) are hard checkpoints — do not proceed past a gate until its criterion is met or consciously waived in writing in `PROJECT_STATE.md`.

The central claim being built, phase by phase:

> *A small, from-scratch self-supervised JEPA learns visual representations whose latent prediction error is a usable anomaly/coherence energy; the frozen encoder transfers label-efficiently to a new domain (overhead imagery); and the whole thing deploys in real time on constrained hardware.*

Strong critic, label-efficient, edge-deployable. Notice what is **not** claimed: beating DINOv2/ImageNet pretraining in absolute accuracy. That is not the game; the game is the same honest framing as the cocktail report.

---

## Phase 0 — Foundation (Week 0, ~2 sessions)

### Step 0.1 — The LLM collaboration protocol

This project will span ~30+ LLM sessions. Sessions have no memory of each other unless you give them one. Create two files in the repo root on day one:

- **`PROJECT_STATE.md`** — the single source of truth. Sections: `Current phase/step`, `Last completed run (W&B link, config, result)`, `Open decisions`, `Next action`, `Waived gates + justification`. **End every session by having the LLM update this file. Start every session by pasting it.**
- **`DECISIONS.md`** — append-only log of irreversible choices (dataset versions, splits, frozen hyperparameters) with one-line rationales. This is what lets you write the final report honestly without archaeology.

Session rules (write these at the top of `PROJECT_STATE.md` so every session sees them):

1. One experiment or one component per session. No "while we're at it" scope creep.
2. The LLM proposes; the training run decides. No conclusion enters `PROJECT_STATE.md` without a W&B link.
3. Any number destined for the resume or report needs ≥3 seeds.
4. Test sets are touched only by the evaluation harness, never during development. Model selection uses validation only.

### Step 0.2 — Compute plan

Realistic options, pick one primary + one fallback:

| Option | Cost | Good for |
|---|---|---|
| Colab Pro / Pro+ (A100/L4 when lucky) | ~$10–50/mo | Phase 1 & 2 training |
| Lightning AI Studio (free monthly GPU credits) | $0–ish | Phase 1 prototyping |
| Kaggle (free P100/T4, 30h/wk) | $0 | Seeds, ablations |
| Purdue cluster (check Gilbreth/RCAC access for undergrads) | $0 | Everything — investigate first, it may solve compute entirely |
| Jetson Orin Nano (Phase 3 only) | ~$250 one-time | Edge benchmark |

**First action of Phase 1, before committing to any schedule: a one-epoch timing run.** Extrapolate total GPU-hours from measured reality, not guesses. Budget envelope to expect: Phase 1 main run ~20–40 GPU-hours, +2 seeds ~2×, ablations ~20h. Phase 2 similar. If the timing run says otherwise, shrink the model or image size — do not shrink the number of seeds.

### Step 0.3 — Repo scaffold

```
jepa-vision/
  PROJECT_STATE.md
  DECISIONS.md
  configs/            # yaml per experiment; config is the experiment
  src/
    data/             # datasets, masking, corruption
    models/           # vit.py, predictor.py, ema.py
    train.py
    diagnostics.py    # effective rank, variance, spread  ← port from cocktail repo
    eval/
      energy.py       # energy computation + AUROC harness ← port the harness shape
      probe.py        # linear/k-shot probes
      ood.py
  scripts/            # one entry point per experiment
  export/             # Phase 3: onnx, quantization, cpp/
  reports/
```

Port from the cocktail repo, adapting interfaces only: EMA update, the SIGReg-family regularizer, collapse diagnostics (effective rank / per-dim variance / spread), the multi-seed AUROC evaluation harness, W&B logging conventions. **These are your unfair advantage — reuse, don't rewrite.**

Fix global reproducibility now: seeded `torch`/`numpy`/`random`, deterministic dataloader workers, config-hash in every W&B run name.

⛔ **Gate 0:** repo runs an end-to-end smoke test (tiny model, 100 images, 2 epochs, diagnostics logging to W&B) on your chosen compute. Nothing else starts until this passes.

---

## Phase 1 — I-JEPA-mini on STL-10 (Weeks 1–4)

**Goal:** reproduce your cocktail result's *shape* in vision: energy separates clean from corrupted/OOD images; frozen encoder beats from-scratch under label scarcity.

### Step 1.1 — Data

- **Pretraining:** STL-10 *unlabeled* split — 100,000 images, 96×96. Built for exactly this. (`torchvision.datasets.STL10(split='unlabeled')`).
- **Probe train/val:** STL-10 labeled train (5,000) — carve 1,000 off as validation. **Probe test:** STL-10 test (8,000), quarantined.
- **Pretraining transforms:** RandomResizedCrop(96, scale 0.3–1.0) + horizontal flip + normalize. No color jitter for v1 — JEPA-family methods deliberately avoid heavy augmentation; that's part of the story.
- **OOD sets (download now, use in 1.5):** SVHN test, CIFAR-10 test (resized to 96).

Log dataset versions + checksums in `DECISIONS.md`.

### Step 1.2 — Architecture

Direct translation of the cocktail model, slot-for-patch:

| Cocktail-JEPA | Vision version |
|---|---|
| Ingredient slot | 8×8 image patch → 12×12 = 144 tokens |
| Set transformer, no pos-enc | ViT **with** 2-D sin-cos positional embeddings (images are not sets — this is the one real change) |
| Context encoder (d=192, 3 layers) | ViT-Tiny-ish: d=192, 6 layers, 3 heads (~6M params) |
| EMA target encoder | Same, EMA momentum 0.996 → 1.0 cosine schedule |
| Low-capacity predictor | 3-layer transformer, width 96, fed context tokens + learnable mask tokens with target positions' pos-embeddings |

**Masking (the heart of I-JEPA — get this right):** per image, sample **4 target blocks** (each 15–20% of the image, aspect ratio 0.75–1.5) and **1 context block** (85–100%), then *remove target patches from the context*. The predictor sees only context-encoded tokens and must predict the EMA-target embeddings of the target patches. Block masking, not random patch masking — random patches are too easy (interpolation) and produce weak representations. Implement masking as its own module with a visual unit test: render 10 sampled masks to a PNG and eyeball them before any training.

**Loss:** mean L2 (or smooth-L1) between predicted and EMA-target patch embeddings, averaged over target patches. Add your SIGReg-family regularizer on encoder outputs at small weight — keep the flag to disable it; that's a free ablation later.

### Step 1.3 — Training run protocol

1. **Timing run** (Step 0.2) → lock the schedule.
2. **Overfit test:** 500 images, no augmentation — loss must collapse toward zero. If it can't memorize, there's a bug, not a research finding.
3. **Reference run:** batch 256 (AMP, grad-accum if needed), AdamW lr 1e-3 with cosine decay + 10-epoch warmup, weight decay 0.04, ~150 epochs. Log every epoch: loss, **effective rank, per-dim variance, embedding spread** (your collapse dashboard — from the first run onward, exactly like last time).
4. Watch the diagnostics, not the loss. JEPA loss going down means little; effective rank staying high while loss falls is the health signal.
5. **Three seeds** of the final config once it's stable. (Five was right for the cocktail model; three is the standard vision compromise — log the choice in `DECISIONS.md`.)

⛔ **Gate 1A (collapse):** effective rank > ~50% of embedding dim and per-dim variance above floor for the whole run. If collapsed: raise EMA momentum start, check stop-gradient, raise regularizer weight — in that order.

### Step 1.4 — Defining the energy

Port the concept: **energy(image) = mean latent prediction error over target patches.** Two upgrades over the cocktail version, both matter:

- **Multi-mask averaging:** one mask sample is high-variance; average energy over K=8 independent mask samples per image at eval time. Sweep K on validation (1/4/8/16) — this is a nice little plot for the report.
- **Per-patch energy map:** before averaging, keep the per-patch errors. Reshaped to the 12×12 grid and upsampled, this is a *spatial anomaly heatmap* — the demo artifact that makes Phase 2 visually legible. Build it now while it's cheap.

### Step 1.5 — Evaluation (the part that makes it research)

**(a) Corruption benchmark** — the perturbation benchmark, vision edition. Apply CIFAR-10-C-style corruptions (gaussian noise, blur, fog, jpeg, elastic — use `imagecorruptions`, severities 1–5) to the quarantined STL test set. Metric: AUROC of energy separating clean vs corrupted, per corruption type and severity, mean over 3 seeds.

**(b) OOD benchmark:** energy on STL test (in-distribution) vs SVHN / CIFAR-10 (out). AUROC per OOD set.

**(c) Baselines** — non-negotiable, this is what made the cocktail report credible:

| Baseline | Role |
|---|---|
| Pixel statistics (mean/var/entropy of pixels) | The trivial floor |
| Random-init encoder, same energy recipe | Proves *training* (not architecture) carries the signal |
| Small pixel-space MAE, reconstruction error as energy | The "why latent, not pixels" comparison |
| Mahalanobis distance on frozen features | Standard OOD baseline; cheap |

**(d) Label-scarce transfer — the twelve-cell table reborn.** Frozen encoder + linear probe vs identical-architecture supervised-from-scratch, at label fractions {1%, 5%, 10%, 100%} of STL-10's 5k train, × 3 seeds. Report the full grid and the per-cell pretrained-minus-scratch gap. Honesty note from the cocktail project applies verbatim: from-scratch gets the same epoch budget and a tuned lr — don't beat a strawman.

⛔ **Gate 1B (proceed to Phase 2):** (i) corruption AUROC clearly above the random-init and pixel baselines on majority of corruption types; (ii) probe at 100% labels ≥ ~70% top-1 (sanity floor — small from-scratch SSL on STL-10 plausibly lands 70–80%); (iii) pretrained > scratch in most low-label cells. If (i) fails, the energy story is broken — debug masking difficulty and EMA schedule before anything else. Expect ~0.7–0.9 AUROC depending on corruption type; some (subtle jpeg at severity 1) will hover near chance — report them, the order-scramble lesson again: some perturbations are near-invisible to the representation by construction.

**Deliverable:** `reports/phase1.md` — the tables, the collapse dashboard screenshot, the energy heatmap figure, three honest paragraphs. One session with the LLM, same voice as the cocktail report.

---

## Phase 2 — The Defense Pivot: Aerial Imagery (Weeks 5–8)

**Goal:** the same skeleton on overhead imagery, producing the three defense-relevant artifacts: unlabeled-data pretraining, anomaly flagging with spatial heatmaps, and few-shot transfer to new classes.

### Step 2.1 — Data

- **RESISC45** — 31,500 remote-sensing scene images, 45 classes, 700 each, 256×256. The workhorse.
- **AID** — ~10,000 aerial images, 30 classes. Combine with RESISC45 for a ~40k unlabeled pretraining pool (strip labels for pretraining; that's the point).
- Resize/crop to 96 or 112 px to keep the Phase-1 architecture (and compute budget) intact; note the resolution choice in `DECISIONS.md` — full-res ViT is a stretch goal, not the plan.

**Split design — do this before training and freeze it (this is the experiment design, give it a full session):**

1. **Class quarantine for anomaly eval:** hold out **5 whole classes** from RESISC45 (pick semantically distinct ones, e.g. `airplane, storage_tank, harbor, thermal_power_station, ship`). These never appear in pretraining or probe training. They are the "anomalies."
2. Remaining 40 classes: 80/10/10 image-level train/val/test, as always.
3. Probe labels come only from the 40-class train split.

### Step 2.2 — Pretraining: one real experiment for free

Train **two** encoders, identical config, 3 seeds each if budget allows (2 acceptable, log it):

- **(A) From scratch** on the aerial pool.
- **(B) Warm-started** from the Phase-1 STL-10 checkpoint.

Whether natural-image JEPA pretraining transfers to overhead imagery is a legitimate question with a clean answer in your tables either way. This is the kind of result that turns "I trained a model" into "I ran a study" in an interview.

### Step 2.3 — Anomaly evaluation, aerial edition

- **(a) Unseen-class detection:** energy on test images of the 40 seen classes vs the 5 quarantined classes. AUROC, per quarantined class and pooled. This is the headline number of Phase 2 — "flags scene types never seen in training."
- **(b) Corruption suite:** rerun the Phase-1 corruption harness on aerial test images (it should be a config change, not new code — if it isn't, the harness needs refactoring, do it now).
- **(c) Heatmap demo:** paste a small anomalous object into a clean scene (simple copy-paste compositing, ~20 hand-made examples) and show the per-patch energy map lighting up on the inserted region. Not a quantitative claim — a figure. This single image does more work in a screen-share interview than any table.
- Baselines: same four as Phase 1, plus encoder (B) vs (A).

**Honesty rail:** unseen-class AUROC and corruption AUROC measure different things; don't blend them into one number. And state the known limitation up front: leave-class-out "anomaly" is a proxy, not real operational anomaly data — scope the claim exactly like the cocktail report scoped the perturbation benchmark.

### Step 2.4 — Label-scarce transfer with a strong external baseline

The twelve-cell grid again — frozen probe vs from-scratch at {1%, 5%, 10%, 100%} label fractions on the 40-class task — **plus one column the cocktail project never had: an ImageNet-pretrained baseline** (frozen ResNet-50 or DINOv2 ViT-S probe, off the shelf).

Expect to **lose** to DINOv2 in absolute accuracy. Report it anyway. The defensible claim writes itself: *a 6M-parameter domain-pretrained encoder, trained from scratch on 40k unlabeled images, closes X% of the gap to a 21M+ web-scale model while being INT8-deployable on a Jetson* — and Phase 3 cashes that last clause. A resume project that voluntarily benchmarks against the thing that beats it reads as research maturity; that's rarer than a good AUROC.

### Step 2.5 — Wrap

⛔ **Gate 2:** unseen-class AUROC meaningfully above random-init baseline AND low-label cells favor pretraining. Then: `reports/phase2.md`, repo README rewrite (this is the version recruiters see — lead with the heatmap figure), and the resume bullet update:

> *Pretrained JEPA on 40k unlabeled aerial images; energy flags unseen scene classes at [X] AUROC with per-patch anomaly heatmaps*
> *Frozen 6M-param encoder matches [Y]% of DINOv2 probe accuracy at 1% labels, trained entirely from scratch*

(Real numbers only, error bars where they fit.)

---

## Phase 3 — Edge Deployment (Weeks 9–11)

**Goal:** the encoder + energy head running in real time under constraint, measured, with a C++ harness. This is where "0.7 AUROC" becomes "N FPS at M watts."

### Step 3.1 — Inference graph definition

Freeze exactly what ships: **context encoder + predictor + fixed mask pattern → scalar energy + 12×12 heatmap.** Decisions to make and log: K mask samples at inference (start with the Phase-1 sweep's elbow, likely 4 — batch the K masks as one forward pass), fp32 reference outputs saved for parity testing.

### Step 3.2 — Export to ONNX

- `torch.onnx.export` (or `torch.export` + ONNX backend), opset ≥ 17, dynamic batch dim.
- **Parity gate:** ONNX Runtime (Python) outputs vs PyTorch within 1e-4 relative on 100 fixed images; energy AUROC on the Phase-2 benchmark identical to 3 decimals. Automate this as a script — you'll rerun it after every export tweak.

### Step 3.3 — Quantization

1. **Dynamic INT8** first (one line, no calibration) — get a number on the board.
2. **Static INT8** with a 500-image calibration set drawn from pretraining data (never test data).
3. Re-run the **full Phase-2 evaluation** on each quantized model. The deliverable is a parity table: fp32 / fp16 / int8-dynamic / int8-static × {AUROC, probe accuracy, model MB}. If int8 drops AUROC by more than ~0.02, try per-channel quantization and excluding LayerNorm/softmax from quantization before accepting the loss.

### Step 3.4 — The C++ harness (your real C++ on-ramp)

Scope it tightly — ~300 lines, not a framework:

```
export/cpp/
  CMakeLists.txt        # links onnxruntime; one external dep policy
  src/main.cpp          # load model → load image (stb_image) → preprocess
                        # → run K-mask batch → energy + heatmap → print/save
  bench/bench.cpp       # latency loop: 100 warmup + 1000 timed, report p50/p95
```

Build order with the LLM (one session each): (1) hello-world ONNX Runtime C++ session loading the model and printing output shapes; (2) preprocessing in C++ matching Python within tolerance — write the parity test first, this is where bugs live; (3) the energy/heatmap computation; (4) the benchmark loop. You know C and GDB from CS 240 — the new material is RAII, vectors/spans, and CMake, which is exactly the right C++ starter dose.

### Step 3.5 — Benchmark matrix

| Platform | Runs |
|---|---|
| Laptop CPU (guaranteed) | fp32 vs int8, 1/4/8 threads |
| Jetson Orin Nano (if purchased) | ONNX Runtime CPU; then TensorRT fp16/int8 (stretch) |

Report per cell: p50/p95 latency, FPS, model size, and the accuracy from 3.3's parity table. On Jetson, log power mode (`nvpmodel`) — that's the watts in "N FPS at M watts."

⛔ **Gate 3 / project done:** one sentence you can defend in any interview, fully instantiated: *"INT8 ViT-Tiny JEPA at [N] FPS / [p50] ms on [platform], retaining [AUROC] of fp32 anomaly performance, in a C++ ONNX Runtime harness I wrote."*

**Stretch (only if ahead of schedule):** distill the encoder to half-width with feature-matching loss; V-JEPA-style video extension; track-level anomaly with the original set transformer.

---

## Risk register (anticipated failure modes — the cocktail brief tradition)

| Risk | Symptom | Mitigation |
|---|---|---|
| Representational collapse | Effective rank crashes, loss suspiciously low | EMA schedule ↑, verify stop-grad, regularizer weight ↑ — diagnostics are already wired, trust them |
| Masking too easy | Loss falls fast, probe accuracy poor | Bigger target blocks, smaller context, verify block (not random-patch) masking |
| Compute blowout | Timing run says 100+ GPU-hrs | Shrink image size → 64px, depth → 4 layers; never cut seeds |
| Energy works on corruptions but not unseen classes | (a) low in Phase 2.3 | Expected risk — semantic OOD is harder than corruption OOD. Report both honestly; multi-mask K ↑ and energy ensembling across layers are the levers |
| Quantization accuracy cliff | int8 AUROC drop > 0.02 | Per-channel quant, exclude norm layers, fp16 fallback |
| LLM context drift across sessions | Re-explaining, contradictory advice | `PROJECT_STATE.md` discipline; paste it at session start, update at session end, no exceptions |
| Probe leakage | Numbers too good | Test sets touched by eval harness only; class quarantine fixed in `DECISIONS.md` before any Phase-2 training |

## Timeline at a glance

| Weeks | Milestone |
|---|---|
| 0 | Gate 0: scaffold + smoke test + timing run |
| 1–2 | Phase-1 reference model healthy (Gate 1A) |
| 3–4 | Phase-1 eval suite + 3 seeds (Gate 1B), `phase1.md` |
| 5–6 | Aerial splits frozen, encoders A & B trained |
| 7–8 | Phase-2 eval + README + **resume refresh before applications open** (Gate 2) |
| 9–10 | ONNX + quantization parity table |
| 11 | C++ harness + benchmark matrix (Gate 3) |

Front-load Phases 1–2: the August–October application window is the real deadline; Phase 3 can run alongside fall applications and lands a mid-cycle resume update.
