# Qwen3-4B + vLLM Serving with LoRA Customization for Customer Support

## Goal

Prepare a self-contained set of scripts, config, and documentation that:

1. Serve a Qwen3 model through vLLM with a single "start command."
2. Explain how Qwen3 and vLLM actually work, so the setup isn't a black box.
3. Turn raw support docs into LoRA training data (via synthetic generation + human review), fine-tune a LoRA adapter for a customer-support assistant, and serve that adapter alongside the base model.

This repository is the authoring environment. The scripts are written and understood here, but are meant to be copied onto a separate Linux/WSL2 machine with an NVIDIA GPU and actually run there.

## Background & Constraints

- The user's local machine has a single GPU with **8GB VRAM**. The target execution machine's specs are **not yet known**, but should be assumed similarly constrained unless stated otherwise.
- The user initially referenced "Qwen3.6," which is a real, newer model family — but it ships only as **Qwen3.6-27B** (dense) and **Qwen3.6-35B-A3B** (MoE, 35B total params, 3B active). Both require significantly more VRAM than 8GB even quantized (27B at 4-bit ≈ 15-16GB; 35B-A3B at 4-bit ≈ 19-20GB, since MoE's *total* parameters must still reside in VRAM for vLLM regardless of active params per token).
- Given the 8GB constraint and the goal of a fully local, hands-on train+serve loop, this spec targets **Qwen3-4B-Instruct-2507** (previous generation, small dense model) instead. This is a deliberate downgrade from Qwen3.6, documented here so the reasoning isn't lost.
- Default precision is bf16. If the target machine turns out to be VRAM-constrained enough that bf16 doesn't fit alongside KV cache, the config can switch to the official `Qwen/Qwen3-4B-AWQ` 4-bit quantized checkpoint with no script changes — just a config value change.

## Architecture

```
QWEN+vLLM+LoRA/
├── config/
│   └── model.env                   # model name, port, context length, quant toggle
├── data/
│   ├── source_docs/                # raw FAQ/product docs dropped in by the user
│   ├── generated/                  # synthetic Q&A pairs, needs human review
│   └── train.jsonl                 # reviewed, final LoRA training data
├── scripts/
│   ├── setup.sh                    # one-time: venv + install vLLM/deps + GPU check
│   ├── start_server.sh             # THE start command: launches vLLM (base model only)
│   ├── test_client.py              # sends a sample chat request, prints the response
│   ├── generate_training_data.py   # turns source_docs/ into draft Q&A pairs
│   ├── validate_dataset.py         # sanity-checks train.jsonl before training
│   ├── train_lora.py               # QLoRA fine-tuning via Unsloth
│   └── serve_with_lora.sh          # start_server.sh + --enable-lora, serves base + adapter
├── output/
│   └── lora_adapter/               # trained adapter weights (git-ignored)
└── docs/
    └── how-it-works.md             # Qwen3 + vLLM + LoRA explainer
```

## Components

### `config/model.env`
Plain `key=value` file: `MODEL=Qwen/Qwen3-4B-Instruct-2507`, `PORT=8000`, `MAX_MODEL_LEN=32768`, `GPU_MEM_UTIL=0.90`, `QUANTIZATION=none`. Every script sources this file rather than hardcoding values, so swapping model/quantization/port is a one-line edit.

### `scripts/setup.sh`
One-time setup on the target machine: creates a Python venv, installs vLLM + CUDA-matched torch + dependencies, and checks `nvidia-smi` / `torch.cuda.is_available()` so GPU/driver problems surface immediately rather than mid-download.

### `scripts/start_server.sh`
The base "start command." Sources `config/model.env` and runs:
```
vllm serve $MODEL --port $PORT --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL [--quantization awq if QUANTIZATION=awq]
```
Starts an OpenAI-compatible API server. Runs in the foreground so startup errors (e.g. OOM) are immediately visible.

### `scripts/test_client.py`
Minimal script using the `openai` Python client pointed at `http://localhost:$PORT/v1`. Sends one chat completion request and prints the response. This is the pass/fail signal that the server is actually working end-to-end — exit code 0 plus a printed response means success.

### `data/source_docs/`
Where the user drops raw customer-support source material — product docs, FAQ text, existing help-center articles — as plain `.md`/`.txt` files. No format requirements beyond plain text.

### `scripts/generate_training_data.py`
Reuses the already-running vLLM server (no second model load) to synthesize training data:
1. Reads each file in `data/source_docs/`.
2. Chunks it into reasonably sized pieces (paragraph or fixed token-count based).
3. Prompts the model to produce N customer-support-style instruction/response pairs per chunk, requesting structured JSON output.
4. Writes all generated pairs to `data/generated/raw_qa.jsonl`.

### Human review (manual step, documented not scripted)
The user reads `data/generated/raw_qa.jsonl`, edits/discards low-quality pairs, and copies the approved ones into `data/train.jsonl`. Synthetic data needs a human check before it's used to train anything — this step is intentionally manual, not automated.

### `scripts/validate_dataset.py`
Sanity-checks `data/train.jsonl` before training starts: valid JSON per line, required fields present (`instruction`/`response` or equivalent messages format), no empty responses. Fails fast with a clear error pointing at the bad line, rather than letting training start on malformed data.

### `scripts/train_lora.py`
QLoRA fine-tuning using **Unsloth** — the standard approach for consumer/8GB-class GPUs, and the framework Qwen's own documentation recommends for SFT. Loads the base model in 4-bit, trains a LoRA adapter against `data/train.jsonl`, and saves the result to `output/lora_adapter/`. Training hyperparameters (rank, alpha, target modules, epochs, batch size, sequence length) live in this script with sensible small-GPU defaults, documented inline.

### `scripts/serve_with_lora.sh`
Same as `start_server.sh`, plus `--enable-lora --lora-modules support-adapter=output/lora_adapter`. This serves the base model and the fine-tuned adapter simultaneously — a client selects which one to hit via the `model` field in its request (`Qwen/Qwen3-4B-Instruct-2507` for base, `support-adapter` for the fine-tuned version).

### `docs/how-it-works.md`
Explains: what Qwen3 is and how it differs from Qwen3.6 (and why this project targets Qwen3-4B); what vLLM does differently from plain `transformers` (continuous batching, PagedAttention for KV cache, why that matters for throughput and VRAM); what LoRA is and why fine-tuning a small adapter instead of the full model is both cheaper and how it plugs into vLLM's multi-adapter serving.

## Data Flow

**Serving (base model):**
1. `setup.sh` once → venv + vLLM installed, GPU verified.
2. `start_server.sh` → vLLM loads Qwen3-4B → OpenAI-compatible server on `PORT`.
3. `test_client.py` (or any client) → `POST /v1/chat/completions` → response.

**Training data pipeline:**
1. User drops docs into `data/source_docs/`.
2. `start_server.sh` running (data generation reuses it) → `generate_training_data.py` → `data/generated/raw_qa.jsonl`.
3. Human review → `data/train.jsonl`.
4. `validate_dataset.py` → pass/fail check.
5. `train_lora.py` → `output/lora_adapter/`.
6. `serve_with_lora.sh` → base model + adapter both servable → client requests `model=support-adapter` to get customized responses.

## Error Handling

Kept minimal, at points where failures are actually likely on a fresh machine:
- `setup.sh` exits with a clear message if no CUDA-capable GPU or `nvidia-smi` is found (the likely failure mode for WSL2 without GPU passthrough configured).
- `start_server.sh` / `serve_with_lora.sh` fail fast with vLLM's own OOM error if the model doesn't fit in VRAM; `how-it-works.md` documents this specific failure and the fix (lower `GPU_MEM_UTIL`, or switch `config/model.env` to the AWQ checkpoint).
- `validate_dataset.py` fails fast on the first malformed line rather than letting `train_lora.py` fail deep into a training run.
- No retry logic, no automatic fallback model selection — failures are config changes, not automated behavior.

## Testing / Verification

Since execution happens on a machine other than this authoring environment, each script is self-verifying when run there:
- `setup.sh` ends by printing the detected GPU and confirming vLLM imports successfully.
- `start_server.sh` / `serve_with_lora.sh` block in the foreground, surfacing startup errors immediately.
- `test_client.py` is the pass/fail signal for serving.
- `validate_dataset.py` is the pass/fail signal for training data before `train_lora.py` runs.
- `train_lora.py` prints final training loss and the adapter save path on completion.

## Open Configuration Notes

- Target execution machine's GPU is currently unknown. Defaults assume an 8GB-class card (matching the local dev machine); `config/model.env` is the single place to adjust if the target machine turns out larger or smaller.
- If the target machine has significantly more VRAM, `MODEL` can be pointed at a larger Qwen3 size (8B/14B) or, if VRAM allows 24GB+, reconsidered against Qwen3.6-27B — that would be a follow-up config change, not a rewrite of these scripts.

## Out of Scope

- Multi-GPU / tensor-parallel serving.
- Automated data-quality scoring or filtering (review is manual).
- DPO/GRPO or other post-training methods beyond SFT-style LoRA.
- A production deployment story (load balancing, auth, monitoring) — this is a local/single-machine dev and training setup.
