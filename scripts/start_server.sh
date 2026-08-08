#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config/model.env"

QUANT_FLAG=()
if [ "${QUANTIZATION:-none}" != "none" ]; then
    QUANT_FLAG=(--quantization "$QUANTIZATION")
fi

REASONING_FLAG=()
if [ -n "${REASONING_PARSER:-}" ]; then
    REASONING_FLAG=(--reasoning-parser "$REASONING_PARSER")
fi

LM_ONLY_FLAG=()
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ] || [ "${LANGUAGE_MODEL_ONLY:-}" = "true" ]; then
    LM_ONLY_FLAG=(--language-model-only)
fi

echo "Starting vLLM server: model=$MODEL port=$PORT max_model_len=$MAX_MODEL_LEN quantization=${QUANTIZATION:-none} reasoning_parser=${REASONING_PARSER:-none} language_model_only=${LANGUAGE_MODEL_ONLY:-0}"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${QUANT_FLAG[@]}" \
    "${REASONING_FLAG[@]}" \
    "${LM_ONLY_FLAG[@]}"
