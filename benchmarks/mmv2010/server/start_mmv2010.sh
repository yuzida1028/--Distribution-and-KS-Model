#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-mmv971_matched_20260929}"
PLAN="${2:-/root/rivermind-data/mu_functions_v2_main9_20260929/config/mu_functions_v2_mu971_main9_20260929_fix1/experiment_manifest.json}"
PROFILE="${3:-matched}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
mkdir -p logs
LOG="logs/mmv2010_${TAG}_${PROFILE}_$(date -u +%Y%m%dT%H%M%S).log"
ARGS=()
if [[ -f "config/mmv2010_${TAG}/binding.json" ]]; then ARGS+=(--resume); fi
nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u server/run_mmv2010.py --reference-plan "$PLAN" --tag "$TAG" --profile "$PROFILE" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
echo "Dispatched PID=$! log=$LOG; solver not verified started."
echo "tail -f '$LOG'"
echo "Actual solve progress: tail -f 'runs/mmv2010_queue_${TAG}/${PROFILE}_s9711.log'"
