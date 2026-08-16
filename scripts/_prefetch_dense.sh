#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
export HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_XET_HIGH_PERFORMANCE=1

echo "Stopping leftover downloaders..."
# Kill by PID pattern that won't match this script's argv
pgrep -af 'train_lora|huggingface-cli|hf download' || true
pkill -9 -f 'python scripts/train_lora.py' 2>/dev/null || true
sleep 2

LOCK_DIR="$HOME/.cache/huggingface/hub/.locks/models--Qwen--Qwen3.6-27B"
if [ -d "$LOCK_DIR" ]; then
  echo "Removing stale locks in $LOCK_DIR"
  rm -rf "$LOCK_DIR"
fi

CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
echo "Cache now: $(du -sh "$CACHE" 2>/dev/null | awk '{print $1}')"
echo "Starting clean hf download (resumes completed shards)..."
hf download Qwen/Qwen3.6-27B
echo "Cache after: $(du -sh "$CACHE" 2>/dev/null | awk '{print $1}')"
echo "DOWNLOAD_OK"
