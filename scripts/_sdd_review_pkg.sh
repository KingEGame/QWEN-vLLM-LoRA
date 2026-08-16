#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA
BASE=$1 HEAD=$2 NAME=$3
OUT=".superpowers/sdd/${NAME}-review-pkg.md"
{
  echo "BASE $BASE"
  echo "HEAD $HEAD"
  git log --oneline "${BASE}..${HEAD}"
  git diff --stat "${BASE}..${HEAD}"
  git diff -U5 "${BASE}..${HEAD}"
} > "$OUT"
echo "wrote $OUT"
