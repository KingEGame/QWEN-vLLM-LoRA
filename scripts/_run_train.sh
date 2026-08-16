#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate
# shellcheck disable=SC1091
if [ -f scripts/wsl_runtime_env.sh ]; then
  source scripts/wsl_runtime_env.sh
fi

# Prefer Hugging Face token from standard cache (never commit tokens into the repo)
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "Starting QLoRA train (config-driven quality knobs)..."
python scripts/train_lora.py
echo "TRAIN_OK"
