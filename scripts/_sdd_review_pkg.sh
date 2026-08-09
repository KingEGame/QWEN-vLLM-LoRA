#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA
BASE=${1:?}
HEAD=${2:?}
N=${3:?}
OUT=".superpowers/sdd/task-${N}-review-pkg.md"
{
  echo "# Review package"
  echo "BASE: $BASE"
  echo "HEAD: $(git rev-parse "$HEAD")"
  echo
  echo "## Commits"
  git log --oneline "${BASE}..${HEAD}"
  echo
  echo "## Stat"
  git diff --stat "${BASE}..${HEAD}"
  echo
  echo "## Diff"
  git diff -U10 "${BASE}..${HEAD}"
} > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
