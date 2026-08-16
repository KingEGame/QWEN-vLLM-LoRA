#!/usr/bin/env bash
set -euo pipefail
pid=$(pgrep -f 'hf download Qwen/Qwen3.6-27B' | head -1 || true)
echo "pid=${pid:-none}"
if [ -z "${pid:-}" ]; then
  echo "DOWNLOAD PROCESS GONE"
  exit 0
fi

echo "=== disk ==="
df -h /home/supre ~/.cache/huggingface | tail -n +1

echo
echo "=== proc write_bytes over 15s ==="
w0=$(awk '/write_bytes/{print $2}' /proc/$pid/io)
sleep 15
w1=$(awk '/write_bytes/{print $2}' /proc/$pid/io)
echo "write_delta=$((w1-w0)) bytes  ($(( (w1-w0)/1024/1024 )) MiB / 15s)"

echo
log=$(ls -t /home/supre/.cache/huggingface/xet/logs/xet_*_${pid}.log 2>/dev/null | head -1)
echo "log=$log"
echo "=== last concurrency / errors ==="
grep -E 'observed bytes|ERROR|error|failed|Warn|WARN' "$log" | tail -n 20

echo
echo "=== cache size ==="
du -sh /home/supre/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B
