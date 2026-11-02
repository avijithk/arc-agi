#!/usr/bin/env bash
# Baseline: train, then evaluate on the 100-puzzle dev subset.
# Safe to rerun after a crash or spot reclaim: training resumes from its last
# checkpoint and evaluation skips puzzles already in predictions.
#   bash scripts/run_baseline.sh                  # 0.5B, dev subset
#   FULL=1 bash scripts/run_baseline.sh           # also all 400 eval puzzles
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f /opt/pytorch/bin/activate ]; then source /opt/pytorch/bin/activate; fi

CONFIG=${CONFIG:-configs/baseline.yaml}
RUN=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['out_dir'])")
mkdir -p "$RUN"

python -m arc_ttt.baseline.train --config "$CONFIG" 2>&1 | tee -a "$RUN/train.out"

python -m arc_ttt.baseline.evaluate --config "$CONFIG" --adapter "$RUN/adapter" \
  --subset configs/dev_subset.txt --out-dir "$RUN/eval-dev" 2>&1 | tee -a "$RUN/eval-dev.out"

if [ "${FULL:-0}" = "1" ]; then
  python -m arc_ttt.baseline.evaluate --config "$CONFIG" --adapter "$RUN/adapter" \
    --out-dir "$RUN/eval-full" 2>&1 | tee -a "$RUN/eval-full.out"
fi

echo "Done. Metrics: $RUN/eval-dev/metrics.json   Remember to stop the instance."
