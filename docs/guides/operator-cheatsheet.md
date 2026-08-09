# Operator cheatsheet

Companion: [Architecture learning](architecture-learning.md)

All commands assume repo root and an activated `.venv` (from setup).

## 1) First-time setup

**Windows:**

```bat
scripts\setup.cmd
```

**WSL / Linux:**

```bash
./scripts/setup.sh
source .venv/bin/activate
```

Optional WSL toolchain (Triton/FlashInfer): `bash scripts/_install_usergcc.sh` then rely on [`scripts/wsl_runtime_env.sh`](../../scripts/wsl_runtime_env.sh).

## 2) Serve base AWQ (no LoRA)

Config: [`config/model.env`](../../config/model.env) (`MODEL`, `QUANTIZATION=awq`, …).

```bash
./scripts/start_server.sh
# other terminal:
python scripts/test_client.py
```

When: you want the stock 27B assistant. First run downloads AWQ weights.

## 3) FAQ LoRA loop

```bash
# server up:
python scripts/generate_training_data.py
cp data/generated/raw_qa.jsonl data/train.jsonl
python scripts/validate_dataset.py data/train.jsonl

# STOP server (Ctrl+C), check VRAM:
nvidia-smi

# train dense base (TRAIN_MODEL in config; override with env as needed)
MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
  GRADIENT_ACCUMULATION_STEPS=8 GRADIENT_CHECKPOINTING=1 \
  python scripts/train_lora.py

./scripts/serve_with_lora.sh
python scripts/test_client.py --model support-adapter
```

Train entry: [`scripts/train_lora.py`](../../scripts/train_lora.py#L50) (`--data` / `--output` optional).

## 4) Personal pipeline (sharper → me-assistant)

Sources: [`config/personal_sources.env`](../../config/personal_sources.env).

**WSL transcript path** (Windows Cursor data):

```bash
export AGENT_TRANSCRIPTS_DIR=/mnt/c/Users/supre/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
```

```bash
python scripts/extract_personal_candidates.py
# EDIT data/personal/candidates/*.jsonl  ← required for quality

python scripts/promote_personal_data.py --reviewed
python scripts/validate_dataset.py data/personal/question_sharp.jsonl
python scripts/validate_dataset.py data/personal/me_assistant.jsonl

# STOP server; train each adapter
MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
  GRADIENT_ACCUMULATION_STEPS=8 GRADIENT_CHECKPOINTING=1 \
  python scripts/train_lora.py \
    --data data/personal/question_sharp.jsonl \
    --output output/lora_question_sharper

MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
  GRADIENT_ACCUMULATION_STEPS=8 GRADIENT_CHECKPOINTING=1 \
  python scripts/train_lora.py \
    --data data/personal/me_assistant.jsonl \
    --output output/lora_me_assistant

LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" \
  ./scripts/serve_with_lora.sh

python scripts/personal_pipeline.py "okey so like why train OOM on 24gb?"
```

Promote gate: [`scripts/promote_personal_data.py`](../../scripts/promote_personal_data.py#L43-L49).  
Pipeline: [`scripts/personal_pipeline.py`](../../scripts/personal_pipeline.py#L22-L60).  
Multi-LoRA: [`scripts/serve_with_lora.sh`](../../scripts/serve_with_lora.sh#L2), [`#L40-L72`](../../scripts/serve_with_lora.sh#L40-L72).

## 5) Always before training

1. Stop vLLM.  
2. `nvidia-smi` → nearly free VRAM.  
3. Then run `train_lora.py`.

## Train knobs (env)

Read by [`scripts/train_lora.py`](../../scripts/train_lora.py):

- `MAX_SEQ_LENGTH`, `BATCH_SIZE`, `NUM_EPOCHS`
- `GRADIENT_ACCUMULATION_STEPS` (default 4)
- `GRADIENT_CHECKPOINTING` (default on)
- `TRAIN_MODEL` / `TRAIN_DATA` / `TRAIN_OUTPUT` (or CLI `--data` / `--output`)

Do **not** assume every `TRAIN_*` key in `config/model.env` is wired — prefer the env names above.

## Privacy / push checklist

- Do not commit HF tokens.
- `data/personal/**/*.jsonl` is gitignored — keep it that way.
- Push when docs + code on `master` are what you want remote; personal datasets stay local.
