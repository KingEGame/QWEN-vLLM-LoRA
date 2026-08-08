# Qwen3.6-27B LoRA Train Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire config + scripts so operators can generate FAQ Q&A on the AWQ 27B server, QLoRA-train an adapter on dense `Qwen/Qwen3.6-27B`, and serve AWQ + adapter.

**Architecture:** Keep serving on AWQ (`MODEL`) and training on dense (`TRAIN_MODEL`). Reuse existing `generate_training_data.py` / `validate_dataset.py` / `train_lora.py`. Align `serve_with_lora.sh` with the flag pattern already used by `start_server.sh`. First-run data promotion is a documented `cp` (no new promote script unless validation forces it).

**Tech Stack:** vLLM serve, transformers + PEFT + bitsandbytes + TRL QLoRA, existing dataset helpers.

## Global Constraints

- Train base: `TRAIN_MODEL=Qwen/Qwen3.6-27B` (dense); never train on the AWQ checkpoint.
- Serve base stays `MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP`.
- First-run data: auto-copy `data/generated/raw_qa.jsonl` → `data/train.jsonl`.
- Smoke train knobs: `MAX_SEQ_LENGTH=1024`, `BATCH_SIZE=1`, `NUM_EPOCHS=1`.
- LoRA rank/alpha remain 16/16; adapter path `output/lora_adapter/`.
- Do not Unsloth-rewrite training in this plan.
- Stop the vLLM server before training (VRAM).
- Do not commit unless the user explicitly asks (or the chosen execution skill requires commits on a feature branch — then commit only plan-scoped files).

## File Structure

```
config/model.env           # MODIFY — add TRAIN_MODEL
scripts/train_lora.py      # MODIFY — safer 27B defaults / target-module fallback
scripts/serve_with_lora.sh # MODIFY — wsl_runtime_env + reasoning/lm-only/max-num-seqs flags
README.md                  # MODIFY — generate → train → serve-with-lora
```

No new Python modules required for v1.

---

### Task 1: Add `TRAIN_MODEL` to config

**Files:**
- Modify: `config/model.env`
- Test: grep keys

**Interfaces:**
- Consumes: design decision `TRAIN_MODEL=Qwen/Qwen3.6-27B`
- Produces: `TRAIN_MODEL` for `train_lora.py` (`os.environ` override still wins)

- [ ] **Step 1: Append / set in `config/model.env`**

Ensure these keys exist (keep existing serve keys):

```env
TRAIN_MODEL=Qwen/Qwen3.6-27B
```

Full expected file after edit (serve knobs unchanged from current master):

```env
MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP
PORT=8000
MAX_MODEL_LEN=4096
GPU_MEM_UTIL=0.92
MAX_NUM_SEQS=32
QUANTIZATION=awq
ADAPTER_NAME=support-adapter
REASONING_PARSER=qwen3
LANGUAGE_MODEL_ONLY=1
TRAIN_MODEL=Qwen/Qwen3.6-27B
```

- [ ] **Step 2: Verify**

```bash
grep -E '^(MODEL|TRAIN_MODEL|QUANTIZATION)=' config/model.env
```

Expected:

```
MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP
QUANTIZATION=awq
TRAIN_MODEL=Qwen/Qwen3.6-27B
```

---

### Task 2: Align `serve_with_lora.sh` with serve flags + WSL env

**Files:**
- Modify: `scripts/serve_with_lora.sh`
- Test: `bash -n scripts/serve_with_lora.sh` + dry-run argv assembly

**Interfaces:**
- Consumes: same optional keys as `start_server.sh` (`REASONING_PARSER`, `LANGUAGE_MODEL_ONLY`, `MAX_NUM_SEQS`, `QUANTIZATION`, `EXTRA_ARGS`) plus LoRA adapter dir
- Produces: `vllm serve` with `--enable-lora` and matching optional flags; sources `scripts/wsl_runtime_env.sh` when present

- [ ] **Step 1: Replace `scripts/serve_with_lora.sh` with**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
if [ -f "$REPO_ROOT/scripts/wsl_runtime_env.sh" ]; then
    source "$REPO_ROOT/scripts/wsl_runtime_env.sh"
fi
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

MAX_SEQS_FLAG=()
if [ -n "${MAX_NUM_SEQS:-}" ]; then
    MAX_SEQS_FLAG=(--max-num-seqs "$MAX_NUM_SEQS")
fi

EXTRA_FLAGS=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_FLAGS=($EXTRA_ARGS)
fi

ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
    exit 1
fi

echo "Starting vLLM server with LoRA: model=$MODEL adapter=$ADAPTER_NAME port=$PORT"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
    "${QUANT_FLAG[@]}" \
    "${REASONING_FLAG[@]}" \
    "${LM_ONLY_FLAG[@]}" \
    "${MAX_SEQS_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}"
```

Ensure LF line endings.

- [ ] **Step 2: Syntax-check**

```bash
bash -n scripts/serve_with_lora.sh && echo "serve_with_lora.sh: OK"
```

---

### Task 3: Harden `train_lora.py` for 27B QLoRA

**Files:**
- Modify: `scripts/train_lora.py`
- Test: unit-level — prefer a small pure helper for target-module resolution if extracted; otherwise smoke via import + dry path that fails on missing `train.jsonl` (existing behavior)

**Interfaces:**
- Consumes: `TRAIN_MODEL` from env/config; `data/train.jsonl`
- Produces: adapter at `output/lora_adapter/`; default smoke-friendly seq/batch when env unset **only if** documented — prefer keeping code defaults but document env overrides in README; optionally set code defaults to `1024`/`1` when `TRAIN_MODEL` contains `27B`

- [ ] **Step 1: Update module docstring** to mention Qwen3.6-27B + `TRAIN_MODEL`

- [ ] **Step 2: After loading the model, if PEFT target modules fail or to be proactive, resolve targets**

Implement this behavior in `main()` before `get_peft_model`:

```python
    target_modules = list(LORA_TARGET_MODULES)
    # If none of the named modules exist (architecture drift), fall back to all Linear names except embeddings/lm_head.
    named = {n.split(".")[-1] for n, _ in model.named_modules()}
    if not any(t in named for t in target_modules):
        import torch.nn as nn
        target_modules = sorted(
            {
                name.split(".")[-1]
                for name, module in model.named_modules()
                if isinstance(module, nn.Linear)
                and name.split(".")[-1] not in {"lm_head"}
            }
        )
        print(f"WARNING: default LoRA targets missing; using Linear modules: {target_modules}")
```

Use `target_modules=target_modules` in `LoraConfig`.

- [ ] **Step 3: When `TRAIN_MODEL` / resolved base contains `27B`, print a reminder to stop the vLLM server and recommend smoke env knobs** (do not hard-fail).

- [ ] **Step 4: Confirm missing-data path still works**

```bash
# from repo root with venv — expect exit 1 if train.jsonl absent
python scripts/train_lora.py ; echo exit:$?
```

Expected: error about missing `data/train.jsonl` (or validation) without traceback noise from unrelated imports if possible — current script imports torch only after validation, keep that order.

---

### Task 4: Document the 27B LoRA loop in README

**Files:**
- Modify: `README.md`
- Test: visual read + link to design exists

**Interfaces:**
- Consumes: Tasks 1–3 operator flow
- Produces: section “LoRA on Qwen3.6-27B (example FAQ)”

- [ ] **Step 1: Add a section after Default model** covering:

1. Start server, generate from example FAQ  
2. Auto-copy promote + validate  
3. Stop server  
4. Train with env knobs  
5. Serve with LoRA + test client  

Exact commands:

```bash
# 1) server already running with AWQ 27B
python scripts/generate_training_data.py

# 2) first-run promote (light review optional)
mkdir -p data
cp data/generated/raw_qa.jsonl data/train.jsonl
python scripts/validate_dataset.py data/train.jsonl

# 3) stop the vLLM server (Ctrl+C in its terminal), confirm VRAM free:
nvidia-smi

# 4) train (downloads dense Qwen/Qwen3.6-27B on first run — large)
TRAIN_MODEL=Qwen/Qwen3.6-27B MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
  python scripts/train_lora.py

# 5) serve base + adapter
./scripts/serve_with_lora.sh
# other terminal:
python scripts/test_client.py --model support-adapter
```

Also link:
`docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md`

- [ ] **Step 2: Confirm design file exists**

```bash
test -f docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md && echo LORA_DOC_OK
```

---

### Task 5: End-to-end GPU run (manual)

**Files:** none (runtime)

**Interfaces:**
- Consumes: Tasks 1–4
- Produces: trained adapter + successful adapter chat reply

- [ ] **Step 1: Ensure AWQ server is running** (`./scripts/start_server.sh`). If `data/generated/raw_qa.jsonl` already exists and is non-empty, move it aside first (generator refuses overwrite).

- [ ] **Step 2: Generate + promote + validate**

```bash
source .venv/bin/activate
python scripts/generate_training_data.py
cp data/generated/raw_qa.jsonl data/train.jsonl
python scripts/validate_dataset.py data/train.jsonl
```

Expected: validate prints success / line count > 0.

- [ ] **Step 3: Stop server; train smoke**

```bash
TRAIN_MODEL=Qwen/Qwen3.6-27B MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
  python scripts/train_lora.py
```

Expected: adapter files under `output/lora_adapter/`; final loss printed.

If OOM: set `MAX_SEQ_LENGTH=512` and retry.

- [ ] **Step 4: Serve with LoRA + test**

```bash
./scripts/serve_with_lora.sh
# other shell:
python scripts/test_client.py --model support-adapter
```

Expected: non-empty response, exit 0.

- [ ] **Step 5: If AWQ+adapter load fails**, capture the error in the task report; do not silently switch training bases — escalate (design risk: dense-trained adapter on AWQ serve).

---

## Spec coverage self-check

| Spec requirement | Task |
|---|---|
| `TRAIN_MODEL=Qwen/Qwen3.6-27B` | Task 1 |
| Generate from example FAQ via running server | Task 5 (uses existing generator) |
| Auto-copy promote + validate | Task 4 docs + Task 5 |
| Stop server before train | Task 4/5 |
| QLoRA train_lora.py / target fallback | Task 3 |
| serve_with_lora aligned flags | Task 2 |
| test_client on `support-adapter` | Task 5 |
| No Unsloth rewrite | (no task) |
| README | Task 4 |
