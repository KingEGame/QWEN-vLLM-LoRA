#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
export PATH="$HOME/micromamba/envs/cc/bin:$HOME/.local/bin:$PATH"

if [ -f data/generated/raw_qa.jsonl ] && [ -s data/generated/raw_qa.jsonl ]; then
  mv data/generated/raw_qa.jsonl "data/generated/raw_qa.jsonl.bak.$(date +%s)"
  echo "Moved existing raw_qa.jsonl aside"
fi

python scripts/generate_training_data.py
cp data/generated/raw_qa.jsonl data/train.jsonl
python scripts/validate_dataset.py data/train.jsonl
echo "GEN_VALIDATE_OK lines=$(wc -l < data/train.jsonl)"
