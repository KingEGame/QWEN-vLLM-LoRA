#!/usr/bin/env bash
set -euo pipefail
CACHE="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
BLOBS="$CACHE/blobs"

echo "=== process ==="
pgrep -af 'hf download|train_lora' || echo "(no download process)"

echo
echo "=== total cache ==="
du -sh "$CACHE" 2>/dev/null || echo missing

echo
echo "=== incomplete shards (live bytes) ==="
if [ -d "$BLOBS" ]; then
  find "$BLOBS" -name '*.incomplete' -printf '%s\t%f\n' 2>/dev/null \
    | sort -nr \
    | head -20 \
    | awk '{ printf "%8.2f MiB  %s\n", $1/1024/1024, $2 }'
  echo
  find "$BLOBS" -name '*.incomplete' -printf '%s\n' 2>/dev/null \
    | awk '{s+=$1; n++} END {printf "incomplete_count=%d  incomplete_total=%.2f GiB\n", n+0, s/1024/1024/1024}'
fi

echo
echo "=== completed blobs (>100MB) ==="
find "$BLOBS" -type f ! -name '*.incomplete' -size +100M -printf '%s\n' 2>/dev/null \
  | awk '{s+=$1; n++} END {printf "done_large_files=%d  done_large_total=%.2f GiB\n", n+0, s/1024/1024/1024}'

echo
echo "=== sample growth over 8s ==="
before=$(du -sb "$CACHE" 2>/dev/null | awk '{print $1}')
sleep 8
after=$(du -sb "$CACHE" 2>/dev/null | awk '{print $1}')
delta=$((after - before))
printf "delta_bytes=%d  (%.2f MiB in 8s ≈ %.2f MiB/s)\n" "$delta" "$(echo "$delta/1024/1024" | bc -l)" "$(echo "$delta/1024/1024/8" | bc -l)"
