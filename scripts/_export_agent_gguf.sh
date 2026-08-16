#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
source config/agent.env

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "ERROR: llama.cpp missing at $LLAMA_CPP_DIR." >&2
  echo "Set LLAMA_CPP_DIR in config/agent.env after installing llama.cpp." >&2
  exit 1
fi
if [ ! -d "$AGENT_ADAPTER_OUTPUT" ]; then
  echo "ERROR: trained adapter missing at $AGENT_ADAPTER_OUTPUT." >&2
  exit 1
fi

python scripts/merge_agent_adapter.py \
  --model "$AGENT_BASE_MODEL" \
  --adapter "$AGENT_ADAPTER_OUTPUT" \
  --output "$AGENT_MERGED_OUTPUT"

F16_GGUF="${AGENT_GGUF_OUTPUT%.gguf}-f16.gguf"
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$AGENT_MERGED_OUTPUT" --outfile "$F16_GGUF" --outtype f16 --no-nextn

QUANTIZE="$LLAMA_CPP_DIR/build/bin/llama-quantize"
if [ ! -x "$QUANTIZE" ]; then
  QUANTIZE="$LLAMA_CPP_DIR/build-host/bin/llama-quantize"
fi
if [ ! -x "$QUANTIZE" ]; then
  QUANTIZE="$LLAMA_CPP_DIR/llama-quantize"
fi
if [ ! -x "$QUANTIZE" ]; then
  echo "ERROR: llama-quantize binary not found; build llama.cpp first." >&2
  exit 1
fi
"$QUANTIZE" "$F16_GGUF" "$AGENT_GGUF_OUTPUT" "$EDGE_QUANTIZATION"
echo "Edge model written to $AGENT_GGUF_OUTPUT"
