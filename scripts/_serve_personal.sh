#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA
cd "$REPO"
source .venv/bin/activate
export LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant"
exec bash scripts/serve_with_lora.sh
