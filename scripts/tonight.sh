#!/usr/bin/env bash
set -euo pipefail

# tonight.sh — 1.6k scratch comparator loop (A3)
# 36 runs: 3 training seeds {0,1,2} × n∈{40,200,400,4000} × lr∈{1e-3,3e-4,1e-4}
# 200 epochs each, batch=min(256,n), formal-val selection
# Writes: reports/scratch_manifest.json (idempotent — existing runs skipped)

CMD="caffeinate -is uv run python scripts/run_scratch_comparator.py"

echo "=== tonight.sh ==="
echo "Command: $CMD"
echo "Projected: ~3–4h, 36 runs"
echo "Output: reports/scratch_manifest.json"
echo "==================="

exec $CMD
