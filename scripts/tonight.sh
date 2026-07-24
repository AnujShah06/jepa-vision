#!/usr/bin/env bash
# tonight.sh — 2.2. Encoder B (warm-start) launch.
#
# Pre-condition: Human has reviewed encoder A Gate 1A evidence and approved.
# Gate 1A is a health-signature gate — no numeric bar. Criteria:
#   - eff_rank well off collapse floor
#   - mean-variance ~0.99
#   - spread stable across training
#   - no NaNs
#   - sane (monotone-ish) loss curve
#
# Warm-start: loads model weights from runs/tkqjawa0/epoch_0150.ckpt (Phase-1
# seed-0 canonical, 192/192 keys clean). Optimizer/scheduler/epoch reset to 0.
# Same 150-epoch budget and config as encoder A (phase2_warmstart.yaml).
#
# DO NOT launch until human has confirmed encoder A Gate 1A.
# (If encoder A was never run, launch encoder A first via:
#   caffeinate -is uv run python scripts/train.py --config configs/phase2_scratch.yaml --seed 0)

caffeinate -is uv run python scripts/train.py \
  --config configs/phase2_warmstart.yaml \
  --warm-start runs/tkqjawa0/epoch_0150.ckpt \
  --seed 0
