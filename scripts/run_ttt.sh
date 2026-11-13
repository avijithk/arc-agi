#!/usr/bin/env bash
# Test-time training on top of the baseline adapter. Rerun the same command to resume.
#   bash scripts/run_ttt.sh                 # 100-puzzle dev subset
#   FULL=1 bash scripts/run_ttt.sh          # all 400 eval puzzles
#   HOLDOUT=1 bash scripts/run_ttt.sh       # hold one demo out (stopping-signal run)
#   CONFIG=configs/x.yaml OUT=runs/x SPLIT=conceptarc bash scripts/run_ttt.sh   # overrides
# Needs runs/baseline-0.5b/adapter (from scripts/run_baseline.sh) unless the config says otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f /opt/pytorch/bin/activate ]; then source /opt/pytorch/bin/activate; fi

CONFIG=${CONFIG:-configs/ttt.yaml}
NAME=$([ "${FULL:-0}" = "1" ] && echo full || echo dev)$([ "${HOLDOUT:-0}" = "1" ] && echo -holdout || true)
OUT=${OUT:-runs/ttt-0.5b/$NAME}
ARGS=(--config "$CONFIG" --out-dir "$OUT")
[ -n "${SPLIT:-}" ] && ARGS+=(--split "$SPLIT")
[ "${FULL:-0}" = "1" ] || ARGS+=(--subset configs/dev_subset.txt)
[ "${HOLDOUT:-0}" = "1" ] && ARGS+=(--holdout)
mkdir -p "$OUT"

python -m arc_ttt.ttt.run "${ARGS[@]}" 2>&1 | tee -a "$OUT/run.out"
python -m arc_ttt.ttt.analyze "$OUT" --out "$OUT/analysis" | tee "$OUT/analysis.out"
echo "Done: $OUT/analysis/analysis.md and accuracy_vs_compute.png. Remember to stop the instance."
