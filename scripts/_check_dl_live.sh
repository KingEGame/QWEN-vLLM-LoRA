#!/usr/bin/env bash
set -euo pipefail
CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
LOCKS="$HOME/.cache/huggingface/hub/.locks/models--Qwen--Qwen3.6-27B"

echo "=== locks ==="
if [ -d "$LOCKS" ]; then ls -la "$LOCKS" | head -40; else echo none; fi

echo
echo "=== sample incomplete size twice, 5s apart ==="
f=$(find "$CACHE/blobs" -name '*.incomplete' | head -1)
echo "file=$f"
stat -c '%s %y' "$f"
sleep 5
stat -c '%s %y' "$f"

echo
echo "=== process IO ==="
pid=$(pgrep -f 'hf download Qwen/Qwen3.6-27B' | head -1 || true)
echo "pid=$pid"
if [ -n "${pid:-}" ]; then
  cat "/proc/$pid/io" || true
  echo
  echo "open network-ish fds:"
  ls -l "/proc/$pid/fd" 2>/dev/null | rg 'socket|incomplete' || ls -l "/proc/$pid/fd" 2>/dev/null | head -30
fi

echo
echo "=== total size now ==="
du -sh "$CACHE"
