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
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

export MAX_SEQ_LENGTH=1024
export BATCH_SIZE=1
export NUM_EPOCHS=1
export GRADIENT_ACCUMULATION_STEPS=8
export GRADIENT_CHECKPOINTING=1

echo "=== train question-sharper (smoke knobs) ==="
python scripts/train_lora.py \
  --data data/personal/question_sharp.jsonl \
  --output output/lora_question_sharper
echo "SHARP_OK"

echo "=== train me-assistant (smoke knobs) ==="
python scripts/train_lora.py \
  --data data/personal/me_assistant.jsonl \
  --output output/lora_me_assistant
echo "ME_OK"
echo "TASK8_TRAIN_OK"
