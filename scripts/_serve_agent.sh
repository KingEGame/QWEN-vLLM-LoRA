#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source config/agent.env

if [ ! -d "$AGENT_ADAPTER_OUTPUT" ]; then
  echo "ERROR: adapter missing at $AGENT_ADAPTER_OUTPUT; train it first." >&2
  exit 1
fi

export MODEL_OVERRIDE="$AGENT_BASE_MODEL"
export MAX_MODEL_LEN_OVERRIDE="${AGENT_SERVER_CONTEXT:-8192}"
export ADAPTER_NAME=personal-agent
export LORA_MODULES="personal-agent=$AGENT_ADAPTER_OUTPUT"
export EXTRA_ARGS_OVERRIDE="--enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder"
exec bash scripts/serve_with_lora.sh
