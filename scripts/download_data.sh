#!/usr/bin/env bash
# Downloads ARC-AGI-1 (400 training + 400 public evaluation tasks) into data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -d data/ARC-AGI ]; then
  git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/ARC-AGI
fi
ln -sfn ARC-AGI/data/training data/training
ln -sfn ARC-AGI/data/evaluation data/evaluation
echo "training:   $(ls data/training | wc -l) tasks"
echo "evaluation: $(ls data/evaluation | wc -l) tasks"
