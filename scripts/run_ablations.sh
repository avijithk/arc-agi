#!/usr/bin/env bash
# R7: every config in configs/ablations/ on the 100-puzzle dev subset, one after another.
# Resumable: finished puzzles (and finished ablations) are skipped on rerun.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f /opt/pytorch/bin/activate ]; then source /opt/pytorch/bin/activate; fi
for cfg in configs/ablations/*.yaml; do
  name=$(basename "$cfg" .yaml)
  echo "=== ablation: $name"
  CONFIG="$cfg" OUT="runs/ablations/$name" bash scripts/run_ttt.sh
done
python -m arc_ttt.ttt.analyze runs/ablations/* --out runs/ablations/combined-analysis | tail -5
