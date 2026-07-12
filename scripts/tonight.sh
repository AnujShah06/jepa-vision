#!/usr/bin/env bash
set -euo pipefail

# tonight.sh — no overnight run
# Scratch loop relaunches in daylight after the stratified_sample bug fix.
# Morning session will overwrite this file with the relaunch command.

echo "No overnight run — scratch loop relaunches in daylight after this fix"
exit 0
