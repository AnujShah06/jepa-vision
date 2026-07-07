# JEPA-VISION — PROJECT HANDOFF & CONTINUITY DOCUMENT

**Prepared July 7, 2026, on loss of prior chat context. This document is the complete institutional memory of the project. The reviewer model (Claude Opus 4.8) and the executor (Claude Code) should treat it as authoritative. Where this document and any model's intuition conflict, this document wins unless the human overrides.**

---

## 0. How to bootstrap the new context (human does this once)

1. Commit this file to the repo root as `HANDOFF.md`. It lives beside `PLAYBOOK.md`, `PROJECT_STATE.md`, `DECISIONS.md`, `CLAUDE.md`.
2. Create a fresh Claude project. Add to project knowledge: `HANDOFF.md`, `PLAYBOOK.md`, and the Cocktail-JEPA Results Report (the predecessor project — it defines the house style for honest evaluation).
3. First message to Opus 4.8 in the new project, verbatim:

> You are the external reviewer for my JEPA-vision project. Read HANDOFF.md in project knowledge completely — it contains your operating manual (Section 9), the full project state, and all pre-registered rules. Claude Code is the executor; I am the gatekeeper. Confirm you've absorbed: (a) the run ledger, (b) the adoption-verdict decision rule currently in force, (c) the failure modes in Section 10. Then give me the next action per Section 4.

4. Nothing in the old chat needs recovering beyond what is here. If a gap is discovered, the repo (git log, W&B project `entropy_chess/jepa-vision`, PROJECT_STATE.md, DECISIONS.md) is the ground truth, in that order of reliability: **W&B > git log > state files > anyone's memory.**

---

## 1. Mission and stakes (why this project exists)

**Person:** Anuj Shah, Purdue CS, B.S. May 2028, US citizen, GPA 3.83. Rising junior recruiting for **Summer 2027 internships** at big tech / frontier labs and **defense tech** (Anduril, Palantir, Shield AI, Saronic tier). Applications open **August–October 2026** — that is the real deadline behind every scheduling decision.

**Prior work:** Cocktail-JEPA (repo: `github.com/AnujShah06/cocktail_JEPA`, public) — set-transformer JEPA on 7,420 recipes; energy-as-coherence-critic at 0.707 ± 0.003 AUROC over 5 seeds; scaling study showing duration-not-capacity; honest critic-vs-generator framing. Currently interning at MARCORSYSCOM (XGBoost MLOps).

**This project (jepa-vision):** the strategic extension identified in career planning — the resume's missing piece is *perception deployed under real-time constraints*. Three phases per PLAYBOOK.md:
- **Phase 1:** I-JEPA-mini on STL-10 — energy as anomaly detector + label-efficient transfer (IN PROGRESS, near completion)
- **Phase 2:** pivot to aerial imagery (RESISC45+AID) — the defense-relevant story (unlabeled pretraining, unseen-class anomaly flagging, few-shot transfer)
- **Phase 3:** edge deployment — ONNX, INT8, ~300-line C++ ONNX Runtime harness, "N FPS at M watts" on laptop CPU / Jetson

**The claim being built:** *a small from-scratch self-supervised JEPA whose latent prediction error is a usable anomaly energy; frozen-encoder label-efficient transfer; deployable in real time on constrained hardware.* Strong critic, label-efficient, edge-deployable. NOT claimed: beating DINOv2/ImageNet pretraining in absolute accuracy (Phase 2 deliberately benchmarks against DINOv2 expecting to lose — research maturity over flattery).

---

## 2. Operating model and roles

- **Claude Code (executor):** works in the repo. Knows ONLY repo files + what is pasted into its session. Has standing rules in `CLAUDE.md` (run ledger maintenance, end-of-session ritual, "tonight's command" requirement, CONTEXT SYNC blocks override repo content).
- **Opus 4.8 in chat (reviewer):** holds this history, verifies gate criteria against pre-registered rules, catches executor drift, designs the next session's copy-paste prompt, sequences the single-GPU night queue, enforces the honesty style. Full manual in Section 9.
- **Anuj (gatekeeper):** launches all training runs, eyeballs all visual gates (PNGs, W&B dashboards), decides all ⛔ gates personally, relays session summaries to the reviewer and reviewer corrections back via CONTEXT SYNC blocks.
- **Memory architecture:** the chat is NOT memory. `PROJECT_STATE.md` (run ledger + decision tree + next action) and `DECISIONS.md` (append-only irreversible choices) are memory. Every reviewer correction must be persisted into them via the next Claude Code prompt or it will be lost.

**Session protocol (unchanged from day 1):** one experiment/component per session; the run decides, not the argument (no conclusion without a W&B link); resume/report numbers need ≥3 seeds; test sets touched only by `src/eval/` code, model selection on validation only.

---

## 3. Current state snapshot — July 7, 2026

**Repo:** local `jepa-vision/`, GitHub `AnujShah06/jepa-vision` (private). ~105 tests passing at last full-suite run. Trains on Apple M-series MPS, fp32, no AMP: **~120 s/epoch, ~5 h per 150-epoch run**, launched with `caffeinate -is`. W&B project: `entropy_chess/jepa-vision`.

**Gap event:** Claude Fable 5 was suspended by U.S. export-control directive June 12–July 1, 2026. The project paused June 12–26+. Critically: **the hardmask seed-0 run WAS launched the evening of June 12 (20:32) and completed successfully, but no Claude Code session ever processed the result** — the last session's ledger predates the launch and wrongly says "queued."

### RUN LEDGER (verified truth as of this document)

| Run | Config | W&B id | Status | Key finals |
|---|---|---|---|---|
| Reference seed 0 | phase1_ref.yaml | **tkqjawa0** | DONE, Gate 1A passed (launched ~Jun 10) | loss 0.2096, pred 0.2071, eff_rank 175.3, spread 19.52, var 0.994 |
| Reference seed 1 | phase1_ref.yaml | **lbd900za** | DONE, Gate 1A passed (Jun 12 AM) | loss 0.2102, pred 0.2074, eff_rank 172.6, spread 19.54, var 0.995 |
| Hardmask seed 0 | phase1_hardmask.yaml | **fw1out6d** | DONE (Jun 12 20:32 → Jun 13). Gate 1A passed pending one verification: confirm the run's W&B/checkpoint config carries the hardmask masking block | loss 0.2933, pred 0.2918, eff_rank 189.2, spread 19.52, var 0.994 |
| Reference seed 2 | phase1_ref.yaml | — | NOT RUN. **Conditional**: runs only if hardmask is REJECTED | — |
| MAE baseline | mae_baseline.yaml | — | NOT RUN. Required regardless of branch. Entry point may still need wiring/smoke-test (was assigned, never confirmed done) | — |

Evidence that fw1out6d is the hardmask run (in case config verification is ambiguous): launch timestamp matches the "hardmask tonight" instruction; pred_loss 0.292 ≫ reference's 0.207 exactly as pre-registered for the harder task; eff_rank 189 ≫ reference's ~173–175; pred_loss curve shows the harder-task shape (mid-run peak, then decline).

**Interpretation of hardmask Gate 1A:** healthy and *as predicted for a harder task* — higher irreducible prediction error, near-full-rank representation (189/192 = 98.5%). Do NOT read the higher loss as worse training.

### The single most important fact

**The project is one probe run away from its central pending decision** (the hardmask adoption verdict, Section 8 rule R1). Everything branches from that number. It costs minutes of compute.

---

## 4. THE IMMEDIATE NEXT ACTION (verbatim Claude Code prompt)

Paste this into Claude Code as the first session of the new era:

```
CONTEXT SYNC — resolving the 19-day gap (Fable suspension, June 12–July 1). HANDOFF.md in the repo root is now authoritative history; read it if present. Apply the following before anything else.

== Ledger corrections (persist to PROJECT_STATE.md) ==
1. Hardmask seed 0 was LAUNCHED the evening of June 12 and COMPLETED: run fw1out6d, 150 epochs, final loss 0.2933, pred_loss 0.2918, eff_rank 189.2, spread 19.52, mean_var 0.994. Verify identity: check W&B config for fw1out6d (masking block is logged) or runs/fw1out6d checkpoint config — confirm it matches phase1_hardmask.yaml. Then mark: DONE, Gate 1A PASSED (pred_loss higher than reference as pre-registered for the harder task; rank 189/192).
2. Resolve seed-2/MAE with evidence: list runs/ and cross-check against W&B (known: tkqjawa0=ref s0, lbd900za=ref s1, fw1out6d=hardmask s0). Unknown run dirs get identified from config and added to the ledger. If only the three known runs exist: Reference seed 2 = NOT RUN, MAE = NOT RUN; delete any "STILL TRAINING" notes as stale fiction.
3. Verify the MAE entry point exists and smoke-test it (2 epochs, 100 images). Wire it if missing. Do not launch full MAE training.

== This session's experiment: the ADOPTION VERDICT (pre-registered) ==
Run the locked probe protocol (target-encoder mean-pooled features + z-score + linear head, lr sweep {3e-3,1e-3,3e-4}, 200 epochs, best-val) on runs/fw1out6d/best.ckpt, val split only, n=4000 and n=200. Also probe n=4000 on checkpoints {90, 120} if saved (mini probe-vs-epoch read for the hardmask model).

Decision rule (binding, from PROJECT_STATE.md / HANDOFF.md R1):
n=4000 ≥ 0.62 AND eff_rank ≥ 96/192 → ADOPT hardmask as production config; next two nights = hardmask seeds 1 and 2; reference demoted to 2-seed reference row.
Otherwise → REJECT; next night = reference seed 2; draft the Gate 1B floor revision in DECISIONS.md (literature-calibrated: small ViT, masked-prediction SSL, no color augmentation, STL-10 — plausible ceiling low-to-mid 60s; claim scoped to low-label gaps + energy results per the cocktail-report style).

Report which branch fired. Update ledger + decision tree. Commit: "1.6c: gap reconciliation + hardmask adoption verdict". End with tonight's exact command per standing rules.
```

---

## 5. Decision tree from here (both branches, night by night)

The GPU is the laptop; one run per night slot (~5 h); `caffeinate -is` always.

| Slot | If ADOPTED (probe ≥ 0.62) | If REJECTED (probe < 0.62) |
|---|---|---|
| Tonight | hardmask `--seed 1` | reference `--seed 2` |
| Night 2 | hardmask `--seed 2` | MAE baseline |
| Night 3 | MAE baseline | (free / start Phase 2 prep) |
| Weekend | **TERMINAL BENCHMARK SESSION** (Section 8, R3) — the sealed test set opens exactly once, everything runs | same |
| Next | `reports/phase1.md` + README rewrite + resume bullet, then Phase 2 kickoff | same, with revised floor documented |

Timeline pressure: Phase 1 must close the week of July 7–13. Phase 2 (aerial) runs mid-July → early August — mostly config changes on the battle-tested harness; the hardmask-vs-reference comparison slots directly into PLAYBOOK Step 2.2's scratch-vs-warm-start experiment. Phase 3 rides alongside fall applications. Applications open August–October; a mid-cycle resume refresh after Phase 2 is the plan of record.
---

## 6. Locked decisions and protocols (no re-deciding these)

**Architecture (both configs):** ViT d=192, 6 layers, 3 heads; patch 8 on 96×96 → 144 tokens; 2-D sin-cos pos-embed; predictor width 96 (`pred_width`, Linear d→96→d projections, Identity if equal), 3 layers; EMA target encoder, momentum 0.996→1.0 cosine; SIGReg-family regularizer (`sigreg_term` in src/models/loss.py, ported verbatim from cocktail repo — never rewrite). ~3.08M trainable / ~5.75M total params. Batch 256, AdamW lr 1e-3 cosine + warmup, 150 epochs.

**Masking:**
- Reference: 4 target blocks, scale 0.15–0.20, AR 0.75–1.5; context 0.85–1.00; measured realized: tgt-union mean 69.7 patches (p5/p95 52/87), ctx-after-removal mean 60.2 (41/80), fallback 0%.
- Hardmask: 4 targets, scale 0.20–0.25; context 0.75–0.90; realized: tgt-union 80.2 (60/100), ctx-after 44.3 (26/64), fallback 0%. Passed the pre-registered starvation gate (ctx p5 ≥ 20, fallback ≤ 2%) with no lever adjustments.
- Known caveat (documented, not a bug): context block realized area can reach ~75% vs nominal 85% minimum due to integer clamping at aspect 1.5; test lower bound deliberately 70% with comment.
- Training collator = one mask per batch (official I-JEPA convention). Eval energy uses per-image masks — separate path, do not "fix" the collator.

**Energy (inference):** smooth-L1 per-patch latent prediction error, target encoder runs once, context+predictor run K times with independent per-image masks, visit-count normalized. **K=8 locked** (rationale: σ −25% vs K=1; K=16 adds only ~6% further σ reduction at 2× cost; AUROC K-insensitive within ±0.02 noise on 1k val — chosen on variance/cost, NOT on AUROC). Per-patch energies retained pre-averaging → 12×12 grid → bilinear-upsampled heatmap overlay (energy_heatmap).

**Probe protocol (locked after diagnostics):** target-encoder mean-pooled patch features + z-score standardization + linear head; lr sweep {3e-3, 1e-3, 3e-4}; 200 epochs; best-val checkpoint. Tracked secondary variant: last-2-layer concat (+2.1pp in diagnostics — carry through multi-seed tables). From-scratch comparator: identical ViT-Tiny end-to-end, same budget, 3-lr sweep — never a strawman.

**Splits (frozen, committed):** `data/splits/stl10_val_idx.json` — 1,000 val images, stratified 100/class, seeded `random.Random(0)`, carved from STL-10 labeled train. Probe pool = the 4,000 complement. Label cells = {40, 200, 400, 4000} (report as fractions of the 4k pool). **Test = STL-10 test (8,000), SEALED — opened exactly once, at the terminal benchmark session.** OOD sets: SVHN test, CIFAR-10 test (resized 96). Corruption suite: imagecorruptions, CIFAR-10-C-style types × severities 1–5.

**Compute:** local MPS is primary (measured 833 img/s, 120s/epoch, peak tensor mem ~130MB); fp32 only (no AMP on MPS — logged decision); RunPod 4090 on-demand is the held fallback (RCAC access was DENIED; Colab rejected for disconnect risk); Kaggle free tier available for parallel ablations. MPS determinism is soft (seed comparisons valid; exact reruns may differ in 3rd decimal).

## 7. Complete results archive (all numbers produced so far, val-only)

**7.1 K-sweep (formal val, ref seed 0, gaussian-3):** K=1: AUROC .781 / σ .0367 · K=4: .747/.0298 · K=8: .763/.0276 · K=16: .792/.0258. (Earlier leaked-split version superseded; leak was 112/500 overlap, fixed by formal split + rerun.)

**7.2 Baseline sanity (val gaussian-3, K=8, ref seed 0):** pixel-std 0.722 · random-init JEPA 0.789 · Mahalanobis 0.604 · MAE-untrained 0.393 (inversion explained: norm_pix_loss makes noisy patches easier for an untrained decoder; expect discontinuity vs trained MAE) · JEPA-trained 0.764. **Interpretation on record: gaussian noise is a norm-level corruption detectable by any architecture (hence random-init 0.789) and is NON-PROBATIVE for the "training carries the signal" claim. Adjudication = per-corruption-type trained−random-init breakdown + semantic OOD (SVHN/CIFAR), where norm tricks don't help. The terminal benchmark's money column is `JEPA(trained) − JEPA(random-init)` per type.**

**7.3 Probe grid (ref seed 0, val, full budget, original context-mean protocol):** n=40: frozen .293 vs scratch .238 (+5.5pp) · n=200: .394 vs .384 (+1.0) · n=400: .435 vs .412 (+2.3) · n=4000: .573 vs .636 (−6.3). Low-label story present; **Gate 1B sanity floor (≥0.70 @ n=4000) FAILING.**

**7.4 Probe diagnostics (n=4000, ref seed 0):** context-mean baseline .571 → best variant (target + z-score + lr-sweep 200ep) **.601**. Feature engineering recovered +3.0pp of a −9.9pp shortfall → **representation quality, not probe configuration** (pre-registered fork, resolved).

**7.5 Probe-vs-epoch (ref seed 0, locked protocol, tgt n=4000):** ep30 .561 · ep60 .576 · ep90 .587 · ep120 .604 · ep150 .600 → **plateau at ~ep120; H2 confirmed (masking too easy); duration is NOT the lever here** — note the deliberate inversion of the cocktail finding (there: duration was the lever, capacity wasn't; here: duration exhausted, task difficulty is the lever). Target≈context probe gap ≤ 0.011 at every epoch (EMA target never differentiates — consistent with too-easy task).

**7.6 Training finals:** see Run Ledger (Section 3).

## 8. Pre-registered rules currently in force (binding)

**R1 — Hardmask adoption verdict (PENDING — the next action):** locked probe on fw1out6d, val, n=4000. ADOPT iff n=4000 ≥ 0.62 AND eff_rank ≥ 96/192 (rank already known: 189 ✓, so the probe number alone decides). Adopt → hardmask = production config, gets seeds 1&2, reference demoted to 2-seed reference row (cocktail-style reference-vs-final framing). Reject → reference seed 2 runs, and the Gate 1B floor is revised in DECISIONS.md with literature-calibrated rationale (small ViT, masked-prediction SSL, no color aug, STL-10 → plausible ceiling low-to-mid 60s; claim then scoped to low-label transfer gaps + energy results). Floor revision is only honest AFTER the masking lever was pulled and measured — which it now has been.

**R2 — Gate 1B (Phase 1 exit), as originally written:** (i) corruption AUROC clearly above random-init and pixel baselines on the majority of corruption types; (ii) probe @ full labels ≥ ~0.70 (subject to R1's revision clause); (iii) pretrained > scratch in most low-label cells. Expect some corruption types near chance (subtle jpeg sev-1) — report honestly, cocktail order-scramble precedent.

**R3 — Terminal benchmark session (the test set opens ONCE):** requires in hand: 2–3 seeds of the production config + 2 seeds of the demoted config (or 3 ref seeds if R1 rejects), trained MAE, the four cheap baselines, locked K=8, locked probe protocol. Runs: full corruption×severity grid + SVHN/CIFAR OOD + probe grid, all models × all seeds, on the SEALED test set, bootstrap CIs via ported src/eval/bootstrap.py. Outputs: reports/phase1.md (per-type trained−random-init table as headline, pooled second; probe grid with per-cell gaps), README rewrite, resume bullet.

**R4 — Overfit-floor claim (stated-not-verified, on record in DECISIONS.md):** the 0.125 fixed-target overfit floor was attributed to per-batch mask resampling; the falsifying experiment (cache one fixed mask per image → expect ~0) was pre-registered but not run. If Phase-1 eval shows anything strange, pull this thread first.

## 9. Reviewer operating manual (for Opus 4.8 — how the last reviewer worked; continue this)

**Response pattern for every session summary the human pastes:** (1) VERDICT — accept / hold / reject the session's claims, checking numbers against this document's rules, not against the session's own framing; (2) CORRECTIONS — anything to persist, delivered inside the next prompt's CONTEXT SYNC block, because chat words never reach Claude Code; (3) NEXT PROMPT — a complete copy-paste block (the human should never have to compose executor instructions); (4) TONIGHT'S COMMAND — exact shell line or "nothing tonight."

**Core duties:**
- Enforce pre-registered rules literally. When a result is unwelcome, the executor will invent comforting explanations (see Section 10, F1). The reviewer's job is to check the invented story against the ledger/W&B before it calcifies.
- Pre-register interpretations BEFORE runs launch ("if X → H1, if Y → H2, next action per branch") — this is the mechanism that has caught every drift so far.
- Prefer the cheapest discriminating experiment (probe-vs-epoch from existing checkpoints beat a 5-hour retrain; frozen-target diagnostic beat pipeline archaeology; mask-stats simulation beat a wasted training night).
- Sequence the single-GPU night queue explicitly; never let a conditional run (ref seed 2) burn a slot before its condition resolves.
- Keep the human in the gate loop: PNGs are eyeballed by the human, gates are decided by the human, "visually confirmed" from the executor does not count.
- Guard the sealed test set. It opens once, at R3, and never before.
- Enforce the honesty house style (from the cocktail report): scope claims to what survives; report the unflattering comparison; per-type breakdowns over pooled numbers; "reference vs final" framing when configs change; limitations sections that name real debts.
- Watch the calendar: applications open August–October 2026. When research and schedule conflict, say so explicitly and propose the scoped-down honest version.
- Prompts to the executor always include: CONTEXT SYNC header for corrections, the experiment, the pre-registered reading, what NOT to do, the commit message, and "end with tonight's command."

## 10. Failure modes already caught (institutional memory — do not relearn these)

- **F1 — Comforting fiction about run status.** The executor claimed tkqjawa0 was "an early checkpoint, not a full production run" to explain a disappointing probe number. It was the completed 150-epoch seed-0 run. Rule: run status comes from the ledger + W&B epoch count, never from the executor's narrative.
- **F2 — Unfilled blanks become facts.** "[FILL IN]" placeholders the human never filled mutated into "STILL TRAINING" ledger entries. Rule: unanswered status questions are resolved by listing runs/ and W&B, not carried forward.
- **F3 — Stale recaps.** Executor "next:" lines repeatedly lagged reality (e.g., "run seed 1" after seed 1 finished). Rule: next-actions derive from the decision tree in PROJECT_STATE.md only.
- **F4 — Eval-split leak.** The first K-sweep used indices 0–499 of labeled train (22% overlap with what became the val split). Caught, split formalized + committed, sweep rerun, conclusion unchanged. Rule: any tuning happens on the formal val split; splits are files, not conventions.
- **F5 — Single-corruption overinterpretation.** Random-init beating trained on gaussian noise nearly triggered a wrong panic; the correct diagnosis was "non-probative corruption type." Rule: per-type breakdowns before conclusions; semantic OOD is the discriminator.
- **F6 — Chat ≠ executor memory.** Reviewer instructions repeatedly failed to reach Claude Code because they were never pasted. Rule: every correction rides in a CONTEXT SYNC block; standing rules live in CLAUDE.md.
- **F7 — MPS quirks.** torch.arange needed explicit device= (crashed AUROC on MPS); AMP avoided on MPS; determinism is soft. Budget one "MPS weirdness" suspicion per new eval component.
- **F8 — Interpretation drift on truncated tests.** A truncated-budget probe tie was read as "expected before 3 seeds" when seeds were irrelevant (probes are per-encoder, minutes each; the truncated budget starved only the scratch model). Rule: identify the actual limiting variable before invoking seed variance.

## 11. Compressed chronology (for orientation, not authority — ledger wins)

Jun ~3–5: repo scaffold, components ported from cocktail_JEPA (loss/sigreg, diagnostics, evaluate/bootstrap, loop/checkpoint), uv env, 15 tests, GitHub push. Jun ~6–8: Gate 0 smoke test (d=64, rank 38/64, healthy); I-JEPA block masking + visual unit test + 14 tests; caveat on realized context area documented. Jun 9–10: timing run on MPS (120s/epoch → local training adopted, RunPod shelved); overfit gate — initial 0.24 floor challenged, frozen-target falsification run (0.125, mechanism identified, gate closed, R4 logged); prod config + checkpoint-resume verified; **ref seed 0 (tkqjawa0) trained overnight, Gate 1A passed.** Jun 10–11: Step 1.4 energy + heatmap + K-sweep; split leak caught (F4), formal val split committed, K=8 locked; baseline suite built, gaussian non-probativity established (F5); probe harness built. Jun 11–12: **ref seed 1 (lbd900za) passed**; full probe grid → Gate 1B floor failing (57%); F1 caught; probe diagnostics (+3.0pp, protocol locked); probe-vs-epoch → H2 (plateau ep120); hardmask designed with starvation gate, mask-stats passed first try, tests parameterized; run ledger + standing rules installed in CLAUDE.md. **Jun 12 20:32: hardmask seed 0 (fw1out6d) launched; completed overnight.** Jun 12–Jul 1: Fable export-control suspension; project paused; result never processed. **Jul 7: this handoff.**

## 12. Off-repo backlog (career items the repo work must not eclipse)

1. **Resume (overdue, never confirmed done):** add MARCORSYSCOM internship entry at top of experience (ML Engineering Intern — production XGBoost pipelines + MLOps specifics); add `github.com/AnujShah06` to the header; JEPA bullet → "0.707 ± 0.003 AUROC across five seeds"; fix "Google Collab"→"Colab" and "ObjectOriented"→"Object-Oriented"; drop Q# unless defensible, add SQL if true, add C++ only when Phase 3 exists; retitle racing role "Machine Learning Platform Engineer"; merge the two TA entries; add scale numbers to infra bullets (real figures only). After Phase 2: retitle project "Self-Supervised Anomaly Detection for Imagery (JEPA)" + aerial/edge bullets.
2. **Applications open Aug–Oct 2026** (big tech + defense tech, Summer 2027). Apply on what exists; refresh mid-cycle with Phase 2/3 results.
3. **Faculty lab outreach** (fall): the Phase-1 W&B dashboard + phase1.md + the cocktail report IS the email. Compute access is a side benefit; the resume line is the goal.
4. **LeetCode in Python** as the steady interview-prep habit; C++ arrives via the Phase-3 ONNX Runtime harness, not via C++ LeetCode.
5. MARCORSYSCOM: work the network, ask about clearance sponsorship, pursue the return offer / referrals.

## 13. Phase 2 & 3 pointers (so momentum survives Phase 1's close)

**Phase 2 (PLAYBOOK §2, mid-July → early Aug):** RESISC45 (31.5k, 45 cls) + AID (~10k) → ~40k unlabeled pool at 96–112px. FIRST, freeze split design in DECISIONS.md: quarantine 5 semantically distinct RESISC45 classes as "anomalies" (suggested: airplane, storage_tank, harbor, thermal_power_station, ship); remaining 40 classes split 80/10/10. Two encoders: (A) from-scratch on aerial, (B) warm-started from the Phase-1 production checkpoint — a real domain-transfer experiment for free; if R1 adopted hardmask, both use hardmask masking. Evals: unseen-class AUROC (headline), corruption suite rerun (must be config-only — refactor if not), copy-paste-composite heatmap demo (~20 images, figure not metric), probe grid + **frozen DINOv2 ViT-S / ImageNet ResNet-50 external baseline (expect to lose; report anyway; claim = gap-closure % + deployability)**. Honesty rail: leave-class-out is a proxy for operational anomaly, say so.

**Phase 3 (PLAYBOOK §3, alongside fall apps):** freeze inference graph (encoder+predictor+fixed-K masks → energy + heatmap; batch the K masks); ONNX export opset ≥17 + parity gate (1e-4 rel on 100 images, AUROC identical to 3 decimals, automated script); dynamic INT8 then static INT8 (500-image calibration from pretraining data, never test); full Phase-2 eval per quantization level → parity table (fp32/fp16/int8-dyn/int8-static × AUROC/probe/MB); accept int8 only if AUROC drop ≤ ~0.02 (else per-channel quant, exclude norms). C++ harness ~300 lines: CMake + onnxruntime + stb_image; build order = hello-world session → preprocessing parity (write the parity test FIRST — bugs live there) → energy/heatmap → bench loop (100 warmup + 1000 timed, p50/p95). Benchmark matrix: laptop CPU (1/4/8 threads) guaranteed; Jetson Orin Nano (~$250) + TensorRT as stretch; log power mode for the watts figure. Done = one defensible sentence: "INT8 ViT-Tiny JEPA at [N] FPS / [p50] ms on [platform], retaining [X] of fp32 anomaly AUROC, in a C++ ONNX Runtime harness I wrote."

*End of handoff. The next probe run decides the branch. Go run it.*
