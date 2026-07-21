#!/usr/bin/env bash
# tonight.sh — 2.0. Gate-0 smoke (2.1) required before any encoder launch.
# Run Gate-0 first; if PASS, encoder A launches tonight. If FAIL, stop.
#
# 2.1 Gate-0 smoke (dataloader + collapse diagnostics on RESISC45, d=64, 2 epochs):
# uv run python scripts/smoke_aerial.py   # (script TBD in session 2.1)
#
# If Gate-0 PASS → encoder A (scratch) launch:
# caffeinate -is uv run python scripts/train.py \
#   --config configs/phase2_ref.yaml --seed 0
#
# NOTHING TONIGHT until 2.1 Gate-0 passes. Run 2.1 first.
echo "2.0 complete. Run 2.1 Gate-0 smoke before any encoder launch."
echo "If Gate-0 PASS, launch encoder A with the command above."
exit 0
