#!/usr/bin/env bash
set -euo pipefail
pid=$(pgrep -f 'hf download Qwen/Qwen3.6-27B' | head -1)
echo "pid=$pid"
echo "=== proc io t0 ==="
grep -E 'read_bytes|write_bytes' /proc/$pid/io
sleep 10
echo "=== proc io t10 ==="
grep -E 'read_bytes|write_bytes' /proc/$pid/io
echo
log=$(ls -t /home/supre/.cache/huggingface/xet/logs/xet_*_${pid}.log 2>/dev/null | head -1)
echo "log=$log"
if [ -n "$log" ]; then
  echo "=== last 40 xet log lines ==="
  tail -n 40 "$log"
fi
echo
echo "=== incomplete mtimes newest 5 ==="
find /home/supre/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/blobs -name '*.incomplete' -printf '%T@ %s %f\n' | sort -nr | head -5 | awk '{printf "%s  %8.2f MiB  %s\n", strftime("%H:%M:%S",$1), $2/1024/1024, $3}'
