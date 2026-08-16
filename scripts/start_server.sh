#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
if [ -f "$REPO_ROOT/scripts/wsl_runtime_env.sh" ]; then
    source "$REPO_ROOT/scripts/wsl_runtime_env.sh"
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/config/model.env"

# First-run CUDA kernel builds can otherwise launch one compiler per CPU and
# exhaust system RAM while a large checkpoint is resident.
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

if [ -x "$REPO_ROOT/.venv/bin/vllm" ]; then
    export PATH="$REPO_ROOT/.venv/bin:$PATH"
    VLLM_BIN="$REPO_ROOT/.venv/bin/vllm"
elif command -v vllm >/dev/null 2>&1; then
    VLLM_BIN="$(command -v vllm)"
else
    echo "ERROR: vLLM not found. Run scripts/setup.sh first." >&2
    exit 1
fi

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

echo "Starting vLLM server: model=$MODEL port=$PORT max_model_len=$MAX_MODEL_LEN max_num_seqs=${MAX_NUM_SEQS:-default} quantization=${QUANTIZATION:-none} reasoning_parser=${REASONING_PARSER:-none} language_model_only=${LANGUAGE_MODEL_ONLY:-0}"

exec "$VLLM_BIN" serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${QUANT_FLAG[@]}" \
    "${REASONING_FLAG[@]}" \
    "${LM_ONLY_FLAG[@]}" \
    "${MAX_SEQS_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}"
