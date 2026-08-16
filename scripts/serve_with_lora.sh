#!/usr/bin/env bash
# Multi-LoRA: LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" ./scripts/serve_with_lora.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
if [ -f "$REPO_ROOT/scripts/wsl_runtime_env.sh" ]; then
    source "$REPO_ROOT/scripts/wsl_runtime_env.sh"
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/config/model.env"

# Optional per-runtime overrides (for example the 1.7B personal-agent base).
MODEL="${MODEL_OVERRIDE:-$MODEL}"
MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-$MAX_MODEL_LEN}"
EXTRA_ARGS="${EXTRA_ARGS_OVERRIDE:-${EXTRA_ARGS:-}}"

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

MAX_SEQS_FLAG=()
if [ -n "${MAX_NUM_SEQS:-}" ]; then
    MAX_SEQS_FLAG=(--max-num-seqs "$MAX_NUM_SEQS")
fi

EXTRA_FLAGS=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_FLAGS=($EXTRA_ARGS)
fi

LORA_MODULE_ARGS=()
if [ -n "${LORA_MODULES:-}" ]; then
    IFS=',' read -r -a _mods <<< "$LORA_MODULES"
    for spec in "${_mods[@]}"; do
        spec_trimmed="$(echo "$spec" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        name="${spec_trimmed%%=*}"
        path="${spec_trimmed#*=}"
        if [[ "$path" != /* ]]; then
            path="$REPO_ROOT/$path"
        fi
        if [ ! -d "$path" ]; then
            echo "ERROR: LoRA path missing for $name: $path" >&2
            exit 1
        fi
        LORA_MODULE_ARGS+=("${name}=${path}")
    done
else
    ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
        exit 1
    fi
    LORA_MODULE_ARGS+=("${ADAPTER_NAME}=${ADAPTER_PATH}")
fi

echo "Starting vLLM server with LoRA modules: ${LORA_MODULE_ARGS[*]} port=$PORT"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "${LORA_MODULE_ARGS[@]}" \
    "${QUANT_FLAG[@]}" \
    "${REASONING_FLAG[@]}" \
    "${LM_ONLY_FLAG[@]}" \
    "${MAX_SEQS_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}"
