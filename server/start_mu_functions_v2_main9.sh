#!/usr/bin/env bash
# Main nine only; preparation may freeze supplementary configs without running them.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${1:-mu971_main9_20260929}"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'invalid tag'; exit 1; }
mkdir -p logs
LOG="logs/main9_${TAG}.log"
test ! -e "$LOG" || { echo "Existing log preserved: $LOG; use explicit resume after checking active tasks."; exit 1; }
nohup bash -c '
set -e
TAG="$1"
python -u server/prepare_mu_functions_v2.py --tag "$TAG" --prepare-only
python -u server/run_jedc_mu_v1.py \
  --plan "config/mu_functions_v2_${TAG}/experiment_manifest.json" \
  --tag "$TAG" --phase main --allow-other-gpu --dev-audit
' main9 "$TAG" > "$LOG" 2>&1 < /dev/null &
echo "Dispatched main-nine preparation PID=$! log=$LOG; training not verified started."
echo "tail -n 60 '$LOG'"
echo "grep -R -m 1 'step 1/100000' 'runs/jedc_mu_v1_queue_${TAG}' --include='*.log'"
