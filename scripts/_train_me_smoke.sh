#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA
cd "$REPO"
source .venv/bin/activate
if [ -f scripts/wsl_runtime_env.sh ]; then source scripts/wsl_runtime_env.sh; fi
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
export MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1
export GRADIENT_ACCUMULATION_STEPS=8 GRADIENT_CHECKPOINTING=1
python scripts/train_lora.py \
  --data data/personal/me_assistant_smoke.jsonl \
  --output output/lora_me_assistant
echo ME_OK
