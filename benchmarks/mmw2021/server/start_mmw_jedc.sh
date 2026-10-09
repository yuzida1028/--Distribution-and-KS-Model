#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PLAN="${1:?pass the absolute reference experiment_manifest.json}"
TAG="${2:-mmw971_sim_20260929}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'invalid tag'; exit 1; }
mkdir -p logs
LOG="logs/${TAG}_queue.log"
test ! -e "$LOG" || { echo 'Existing queue log preserved; use documented explicit resume.'; exit 1; }
ARGS=()
if test -f "config/$TAG/binding.json"; then ARGS+=(--resume); fi
nohup python -u server/run_mmw_jedc.py --reference-plan "$PLAN" --tag "$TAG" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
echo "Dispatched PID=$! log=$LOG; training not verified until a step log appears."
echo "tail -n 40 '$LOG'"
echo "tail -f 'logs/${TAG}_s9711.log'"
