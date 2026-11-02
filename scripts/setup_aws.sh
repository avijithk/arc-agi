#!/usr/bin/env bash
# One-time setup on a fresh g5.xlarge running an AWS Deep Learning AMI (PyTorch, Ubuntu).
# Run from the repo root after cloning:  bash scripts/setup_aws.sh
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,memory.total --format=csv

# Use the AMI's PyTorch environment if it has one, so torch isn't reinstalled.
if [ -f /opt/pytorch/bin/activate ]; then source /opt/pytorch/bin/activate; fi
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'bf16:', torch.cuda.is_bf16_supported())"

pip install -q transformers peft accelerate pyyaml tqdm matplotlib pytest
pip install -q -e .

bash scripts/download_data.sh
python -m pytest -q

# Pre-download the base model so the first run doesn't stall on it.
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; m='Qwen/Qwen2.5-0.5B'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m); print('model cached')"

echo "Setup done. Start a run inside tmux:  tmux new -s run   then   bash scripts/run_baseline.sh"
