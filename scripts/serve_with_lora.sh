#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/config/model.env"

QUANT_FLAG=()
if [ "${QUANTIZATION:-none}" != "none" ]; then
    QUANT_FLAG=(--quantization "$QUANTIZATION")
fi

EXTRA_FLAGS=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_FLAGS=($EXTRA_ARGS)
fi

ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
    exit 1
fi

echo "Starting vLLM server with LoRA: model=$MODEL adapter=$ADAPTER_NAME port=$PORT"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
    "${QUANT_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}"
