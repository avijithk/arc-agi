#!/usr/bin/env bash
# Final runs on the partner's instance (needs runs/baseline-rearc-0.5b/adapter copied in first).
# Run inside tmux:  bash scripts/run_final_partner.sh 2>&1 | tee -a runs/final.log
# Every step resumes if rerun. Shuts the instance down at the end unless NO_SHUTDOWN=1.
set -uo pipefail
cd "$(dirname "$0")/.."
source /opt/pytorch/bin/activate 2>/dev/null || true
test -f runs/baseline-rearc-0.5b/adapter/adapter_config.json || { echo "missing runs/baseline-rearc-0.5b/adapter"; exit 1; }

# 1. Timing (~40 min): answer cost with 1/2/4/8 views, no training, 30 dev puzzles
for k in 1 2 4 8; do
  python -m arc_ttt.ttt.run --config configs/timing/views$k.yaml --out-dir runs/timing/views$k \
    --subset configs/dev_subset.txt --limit 30
done

# 2. ConceptARC, unseen puzzles (~12 h): frozen triage predictor + EMA test
[ -d data/conceptarc ] || bash scripts/download_conceptarc.sh
SPLIT=conceptarc FULL=1 CONFIG=configs/ttt_final_conceptarc.yaml OUT=runs/conceptarc/final bash scripts/run_ttt.sh

# 3. All 400 ARC eval puzzles with EMA (~14 h): EMA on the full set + a second early-checkpoint run
FULL=1 CONFIG=configs/ttt_final_ema.yaml OUT=runs/ttt-rearc-0.5b/full-ema bash scripts/run_ttt.sh

echo "ALL FINAL RUNS DONE"
[ "${NO_SHUTDOWN:-0}" = "1" ] || sudo shutdown -h now
