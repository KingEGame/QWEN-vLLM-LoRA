# Qwen3.6-27B AWQ Serve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the default vLLM serve path to Qwen3.6-27B AWQ on a single 24GB GPU, with optional reasoning/language-only flags driven by `config/model.env`.

**Architecture:** Keep `config/model.env` as the single source of truth. Extend `scripts/start_server.sh` to emit `--reasoning-parser` and `--language-model-only` only when configured. Leave LoRA train/serve scripts unchanged.

**Tech Stack:** bash, vLLM ≥0.19 (env has 0.26.0), Hugging Face AWQ checkpoint `shawnw3i/Qwen3.6-27B-AWQ-MTP`.

## Global Constraints

- Serve-only; do not change LoRA training defaults or `scripts/train_lora.py`.
- Model: `shawnw3i/Qwen3.6-27B-AWQ-MTP`
- Quantization: `awq`
- Max context first value: `8192`
- GPU mem util: `0.90`
- Reasoning parser: `qwen3`
- Language-model-only: enabled (`LANGUAGE_MODEL_ONLY=1`)
- Do not enable MTP speculative decoding in this pass.
- Do not commit unless the user explicitly asks.

## File Structure

```
config/model.env          # MODIFY — 27B AWQ + new flag keys
scripts/start_server.sh   # MODIFY — honor REASONING_PARSER / LANGUAGE_MODEL_ONLY
README.md                 # MODIFY — document 27B AWQ default + rollback
```

No new Python modules. Existing pytest suite untouched (no GPU serve tests).

---

### Task 1: Point config at Qwen3.6-27B AWQ

**Files:**
- Modify: `config/model.env`
- Test: `cat config/model.env` (manual assertion of keys)

**Interfaces:**
- Consumes: design values from `docs/superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md`
- Produces: env keys `MODEL`, `QUANTIZATION`, `MAX_MODEL_LEN`, `REASONING_PARSER`, `LANGUAGE_MODEL_ONLY` for Task 2

- [ ] **Step 1: Replace `config/model.env` contents**

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

- [ ] **Step 2: Verify keys**

Run (WSL or Git Bash from repo root):

```bash
grep -E '^(MODEL|QUANTIZATION|MAX_MODEL_LEN|REASONING_PARSER|LANGUAGE_MODEL_ONLY)=' config/model.env
```

Expected:

```
MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP
MAX_MODEL_LEN=8192
QUANTIZATION=awq
REASONING_PARSER=qwen3
LANGUAGE_MODEL_ONLY=1
```

---

### Task 2: Teach `start_server.sh` the new optional flags

**Files:**
- Modify: `scripts/start_server.sh`
- Test: `bash -n scripts/start_server.sh` plus a dry echo of the assembled argv (no GPU required)

**Interfaces:**
- Consumes: `REASONING_PARSER`, `LANGUAGE_MODEL_ONLY`, existing `QUANTIZATION` / `MODEL` / `PORT` / `MAX_MODEL_LEN` / `GPU_MEM_UTIL` from `config/model.env`
- Produces: `vllm serve ...` with `--quantization awq`, `--reasoning-parser qwen3`, `--language-model-only` when keys are set

- [ ] **Step 1: Replace `scripts/start_server.sh` with**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config/model.env"

QUANT_FLAG=()
if [ "${QUANTIZATION:-none}" != "none" ]; then
    QUANT_FLAG=(--quantization "$QUANTIZATION")
fi

REASONING_FLAG=()
if [ -n "${REASONING_PARSER:-}" ]; then
    REASONING_FLAG=(--reasoning-parser "$REASONING_PARSER")
fi

LM_ONLY_FLAG=()
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ] || [ "${LANGUAGE_MODEL_ONLY:-}" = "true" ]; then
    LM_ONLY_FLAG=(--language-model-only)
fi

echo "Starting vLLM server: model=$MODEL port=$PORT max_model_len=$MAX_MODEL_LEN quantization=${QUANTIZATION:-none} reasoning_parser=${REASONING_PARSER:-none} language_model_only=${LANGUAGE_MODEL_ONLY:-0}"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${QUANT_FLAG[@]}" \
    "${REASONING_FLAG[@]}" \
    "${LM_ONLY_FLAG[@]}"
```

- [ ] **Step 2: Syntax-check**

```bash
bash -n scripts/start_server.sh && echo "start_server.sh: OK"
```

Expected: `start_server.sh: OK`

- [ ] **Step 3: Dry-run flag assembly (no vLLM launch)**

```bash
bash -c '
source config/model.env
QUANT_FLAG=(); [ "${QUANTIZATION:-none}" != "none" ] && QUANT_FLAG=(--quantization "$QUANTIZATION")
REASONING_FLAG=(); [ -n "${REASONING_PARSER:-}" ] && REASONING_FLAG=(--reasoning-parser "$REASONING_PARSER")
LM_ONLY_FLAG=(); { [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ] || [ "${LANGUAGE_MODEL_ONLY:-}" = "true" ]; } && LM_ONLY_FLAG=(--language-model-only)
printf "%s\n" vllm serve "$MODEL" --port "$PORT" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" "${QUANT_FLAG[@]}" "${REASONING_FLAG[@]}" "${LM_ONLY_FLAG[@]}"
'
```

Expected argv includes:

```
--quantization awq
--reasoning-parser qwen3
--language-model-only
```

and model `shawnw3i/Qwen3.6-27B-AWQ-MTP`.

---

### Task 3: Document the 27B default and rollback

**Files:**
- Modify: `README.md`
- Test: visual read of the onboarding section

**Interfaces:**
- Consumes: Task 1 config values
- Produces: operator-facing notes for serve + OOM knobs + how to revert to 4B

- [ ] **Step 1: Update README title/intro and add a short “Default model” section after onboarding**

Replace the top of `README.md` with:

```markdown
# Qwen3.6-27B (AWQ) + vLLM + LoRA

Serve **Qwen3.6-27B** via vLLM (AWQ 4-bit for single-GPU 24GB cards), then
optionally customize a smaller base model with a LoRA adapter trained on your
own docs. LoRA fine-tuning of the 27B checkpoint is not in this path yet.

Authored to run on **Linux or WSL2** with an NVIDIA GPU. Native Windows cannot
run the GPU stack; Windows teammates use the thin setup wrappers below, which
forward into WSL.

## Onboarding (setup only)

One command installs the Python venv, dependencies, and verifies CUDA/vLLM.
Starting the server and sending a test request are **manual** next steps.

**Windows (PowerShell or cmd):**

```bat
scripts\setup.cmd
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

**Linux / already inside WSL:**

```bash
./scripts/setup.sh
```

When setup finishes, activate the venv it created, then:

```bash
./scripts/start_server.sh
# in another terminal, with the venv active:
python scripts/test_client.py
```

## Default model (Qwen3.6-27B AWQ)

`config/model.env` defaults to:

- `MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP`
- `QUANTIZATION=awq`
- `MAX_MODEL_LEN=8192`
- `REASONING_PARSER=qwen3`
- `LANGUAGE_MODEL_ONLY=1`

First server start downloads the AWQ weights from Hugging Face (large).

If you OOM: lower `MAX_MODEL_LEN` (e.g. `4096`) or `GPU_MEM_UTIL` (e.g. `0.85`).
If VRAM remains, try raising `MAX_MODEL_LEN` toward `16384`.

To roll back to the small bf16 model for LoRA experiments, set in `config/model.env`:

```env
MODEL=Qwen/Qwen3-4B-Instruct-2507
MAX_MODEL_LEN=32768
QUANTIZATION=none
REASONING_PARSER=
LANGUAGE_MODEL_ONLY=0
```

## Troubleshooting

Driver, CUDA, and out-of-memory issues are documented in:

- [Design: Qwen + vLLM + LoRA setup](docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md)
- [Design: easy onboard setup scripts](docs/superpowers/specs/2026-08-06-easy-onboard-setup-scripts-design.md)
- [Design: Qwen3.6-27B AWQ serve](docs/superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md)

Tune model/port/context in `config/model.env` — setup does not rewrite it.

## Unit tests (no GPU required)

`pytest` is not installed by `setup.sh`; install it into the venv first:

```bash
pip install pytest
python -m pytest -v
```
```

- [ ] **Step 2: Confirm README links resolve**

```bash
test -f docs/superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md && echo README_LINK_OK
```

Expected: `README_LINK_OK`

---

### Task 4: GPU smoke (manual, on this machine)

**Files:** none (runtime verification)

**Interfaces:**
- Consumes: Tasks 1–2 serve path
- Produces: pass/fail that the 27B AWQ server answers chat completions

- [ ] **Step 1: Start server in WSL (foreground)**

```bash
cd /mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA
source .venv/bin/activate
./scripts/start_server.sh
```

Expected: model download (first time) then logs showing the server listening on port `8000` without CUDA OOM.

- [ ] **Step 2: In a second WSL shell, run the test client**

```bash
cd /mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA
source .venv/bin/activate
python scripts/test_client.py
```

Expected: exit code `0` and a non-empty model reply printed.

- [ ] **Step 3: If OOM, apply knobs from the design and retry**

Edit `config/model.env`: `MAX_MODEL_LEN=4096` and/or `GPU_MEM_UTIL=0.85`, then repeat Steps 1–2.

---

## Spec coverage self-check

| Spec requirement | Task |
|---|---|
| Switch MODEL to AWQ 27B | Task 1 |
| QUANTIZATION=awq, MAX_MODEL_LEN=8192 | Task 1 |
| `--reasoning-parser qwen3` | Task 2 |
| `--language-model-only` | Task 2 |
| README note + VRAM knobs | Task 3 |
| Leave LoRA scripts alone | (no task touches them) |
| Verify with start_server + test_client | Task 4 |
| No MTP in v1 | (not added to start_server.sh) |
