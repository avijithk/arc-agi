#!/usr/bin/env bash
# Everything the Midway Report needs, split across the two AWS accounts.
#   bash scripts/run_midway.sh A    # instance A: baseline (dev + all 400), then TTT on the dev subset
#   bash scripts/run_midway.sh B    # instance B: baseline training, then TTT on all 400 (the long run)
# Both train the same baseline (same seed), so no adapter has to be copied between accounts.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f /opt/pytorch/bin/activate ]; then source /opt/pytorch/bin/activate; fi
mkdir -p runs
case "${1:-}" in
  A) FULL=1 bash scripts/run_baseline.sh
     bash scripts/run_ttt.sh ;;
  B) python -m arc_ttt.baseline.train --config configs/baseline.yaml 2>&1 | tee -a runs/baseline-train.out
     FULL=1 bash scripts/run_ttt.sh ;;
  *) echo "usage: bash scripts/run_midway.sh A|B"; exit 1 ;;
esac
