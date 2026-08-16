#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
source config/agent.env

PREPARE_ARGS=()
if [ "${AGENT_INCLUDE_TEACHER_CURATED:-0}" = "1" ]; then
  PREPARE_ARGS+=(--include-teacher-curated)
fi
python scripts/prepare_agent_dataset.py "${PREPARE_ARGS[@]}"
if pgrep -f '[v]llm serve' >/dev/null; then
  echo "ERROR: stop the vLLM server before QLoRA training to free VRAM." >&2
  exit 1
fi

export TRAIN_MODEL="$AGENT_BASE_MODEL"
export MAX_SEQ_LENGTH="${AGENT_TRAIN_MAX_SEQ_LENGTH:-1024}"
export BATCH_SIZE="${AGENT_TRAIN_BATCH_SIZE:-1}"
export NUM_EPOCHS="${AGENT_TRAIN_EPOCHS:-3}"
export GRADIENT_ACCUMULATION_STEPS="${AGENT_TRAIN_GRAD_ACCUM:-8}"
export GRADIENT_CHECKPOINTING=1
python scripts/train_lora.py --data "$AGENT_TRAIN_DATA" --output "$AGENT_ADAPTER_OUTPUT"
