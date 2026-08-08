# Design: Serve Qwen3.6-27B (AWQ) on 24GB WSL GPU

**Date:** 2026-08-08  
**Status:** approved  
**Scope:** inference/serve only. LoRA training on 27B is explicitly deferred.

## Goal

Make this repo able to start an OpenAI-compatible vLLM server for **Qwen3.6-27B** on the current machine (WSL2 Ubuntu, single **RTX 5090 Laptop 24GB**), without changing the LoRA training path yet.

## Context

- Current default is `Qwen/Qwen3-4B-Instruct-2507` at bf16 (`QUANTIZATION=none`, `MAX_MODEL_LEN=32768`).
- Full/bf16 Qwen3.6-27B needs ~50GB+ VRAM — not viable on 24GB.
- Official `Qwen/Qwen3.6-27B-FP8` is ~27GB weights alone — high OOM risk on 24GB.
- User chose: **serve now, LoRA later**, via **AWQ 4-bit**.

## Decision

Use community AWQ weights that fit a single 24GB card:

| Key | Value |
|---|---|
| `MODEL` | `shawnw3i/Qwen3.6-27B-AWQ-MTP` |
| `QUANTIZATION` | `awq` |
| `MAX_MODEL_LEN` | `8192` (conservative first value) |
| `GPU_MEM_UTIL` | `0.90` |
| `PORT` | `8000` (unchanged) |

Serve command additions beyond what `start_server.sh` already supports:

- `--reasoning-parser qwen3` (Qwen3.6 thinking/reasoning format)
- `--language-model-only` (skip vision/multimodal profiling to free KV-cache VRAM if the checkpoint exposes multimodal components)

## Architecture / file changes

```
config/model.env          # switch MODEL + QUANTIZATION + MAX_MODEL_LEN
scripts/start_server.sh   # optional flags from model.env (REASONING_PARSER, LANGUAGE_MODEL_ONLY)
README.md                 # note 27B AWQ serve path + VRAM caveat
```

No changes to LoRA scripts (`train_lora.py`, `serve_with_lora.sh`, dataset pipeline) in this pass. They continue to assume the smaller base model until a later design.

### `config/model.env` shape

```env
MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP
PORT=8000
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.90
QUANTIZATION=awq
ADAPTER_NAME=support-adapter
REASONING_PARSER=qwen3
LANGUAGE_MODEL_ONLY=1
```

`start_server.sh` reads the new keys and only adds flags when set, so rolling back to the 4B model remains a config edit.

## Verification

1. `source .venv/bin/activate`
2. `./scripts/start_server.sh` — must load without OOM; first run downloads the AWQ weights from Hugging Face.
3. `python scripts/test_client.py` — exit 0 and a printed reply.

If OOM at 8192: lower `MAX_MODEL_LEN` (e.g. 4096) or `GPU_MEM_UTIL` (e.g. 0.85). If VRAM headroom remains, raise `MAX_MODEL_LEN` toward 16384.

## Out of scope

- LoRA / QLoRA fine-tuning of Qwen3.6-27B
- Official FP8 or bf16 checkpoints on this GPU
- Native 262k context / multi-GPU tensor parallel
- Switching Unsloth training defaults to 27B
- Guaranteeing quality parity with the official FP8 release

## Risks

- **Community AWQ quality** may differ slightly from official FP8/bf16.
- **Qwen3.6 hybrid architecture** needs a recent vLLM (this env already has `vllm==0.26.0`, above the documented `>=0.19.0` recommendation).
- **First download** is large; store under HF cache (prefer Linux filesystem if `/mnt/c` I/O is slow).
- **MTP speculative decoding** is available on this AWQ build but not enabled in v1 — keep the first serve path simple.
