# Operator cheatsheet

Legacy Qwen3.6 background: [Architecture learning](architecture-learning.md)
and [LoRA, QLoRA, and AWQ learning](lora-qlora-learning.md).
For the current adapter workflow, use the
[Qwen3.8 QLoRA training guide](qwen38-qlora-training.md).

Linux commands assume repo root and an activated `.venv` (from setup).
PowerShell commands use the repository's verified WSL path.

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

The WSL toolchain is required for first-run Triton/FlashInfer compilation:

```bash
bash scripts/_install_usergcc.sh
```

The server automatically sources
[`scripts/wsl_runtime_env.sh`](../../scripts/wsl_runtime_env.sh). It supplies
the CUDA headers/libraries, limits native compilation to a safe job count, and
uses heuristic FP4 tactics instead of the optional approximately 44-minute
autotune pass.

## 2) Serve Qwen3.8-27B NVFP4 (no LoRA)

Config: [`config/model.env`](../../config/model.env). The tested defaults are:

- `MODEL=unsloth/Qwen3.8-27B-NVFP4`
- 4,096-token context, one concurrent sequence
- 94% initial GPU-memory budget and an explicit 512MiB FP8 KV cache
- text-only mode, `qwen3` reasoning parser, and `qwen3_coder` tool parser

Inside WSL/Linux, run in the foreground:

```bash
./scripts/start_server.sh
```

From PowerShell, run in the foreground:

```powershell
wsl -d Ubuntu -e bash /mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA/scripts/start_server.sh
```

For a hidden background server with persistent logs, run from the repository
root in PowerShell:

```powershell
$qwenServerArgs = @('-d','Ubuntu','-e','bash','/mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA/scripts/start_server.sh')
Start-Process wsl.exe -ArgumentList $qwenServerArgs -WindowStyle Hidden `
  -RedirectStandardOutput '.\output\qwen3.8-server.stdout.log' `
  -RedirectStandardError '.\output\qwen3.8-server.stderr.log'
```

The checkpoint download is about 23.44GB. On the verified RTX 5090 Laptop GPU,
vLLM loads about 20.47GiB of weights into VRAM. The first launch may build
Blackwell (`sm120`) kernels; later launches reuse the caches.

## 3) Health and generation checks

From PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

Run the end-to-end smoke test through the repository client:

```powershell
wsl -d Ubuntu -e bash -lc "cd /mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA && .venv/bin/python scripts/test_client.py --prompt 'Reply with exactly: Qwen3.8 local server is working.'"
```

Expected response:

```text
Qwen3.8 local server is working.
```

## 4) Stop or restart

Foreground server: press `Ctrl+C` in its terminal.

Any PowerShell terminal:

```powershell
wsl -d Ubuntu -e bash -lc "pkill -TERM -x vllm"
```

Confirm VRAM was released:

```powershell
wsl -d Ubuntu -e nvidia-smi
```

Start it again with the command in section 2. Background-launch logs, when the
server was started through PowerShell `Start-Process`, are stored in:

```text
output/qwen3.8-server.stdout.log
output/qwen3.8-server.stderr.log
```

## 5) Current limits

- Official BF16 weights need roughly 54GB before runtime overhead and cannot
  fit on a single 24GB GPU.
- NVFP4 is the serving format that fits this machine.
- The server is intentionally text-only and MTP is not enabled.
- Existing Qwen3.6 adapters are incompatible with Qwen3.8.
- Retrain adapters against `Qwen/Qwen3.8-27B`; do not train the NVFP4 files.

## 6) FAQ LoRA loop (Qwen3.8 retraining required)

This workflow has not yet been revalidated end-to-end with Qwen3.8. Delete or
replace any Qwen3.6 adapter output before testing a newly trained adapter.

```bash
# Qwen3.8 NVFP4 server up:
python scripts/generate_training_data.py
cp data/generated/raw_qa.jsonl data/train.jsonl
python scripts/validate_dataset.py data/train.jsonl

# STOP server (Ctrl+C), check VRAM:
nvidia-smi

# train official source base (TRAIN_MODEL=Qwen/Qwen3.8-27B)
MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
  GRADIENT_ACCUMULATION_STEPS=8 GRADIENT_CHECKPOINTING=1 \
  python scripts/train_lora.py

./scripts/serve_with_lora.sh
python scripts/test_client.py --model support-adapter
```

Train entry: [`scripts/train_lora.py`](../../scripts/train_lora.py#L50) (`--data` / `--output` optional).

## 7) Personal pipeline (sharper → me-assistant)

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

If only the base Qwen3.8 model is currently listed by `/v1/models`, use the
temporary fallback instead of requesting unloaded adapter names:

```bash
python scripts/personal_pipeline.py --base-only "okey so like why train OOM on 24gb?"
```

Promote gate: [`scripts/promote_personal_data.py`](../../scripts/promote_personal_data.py#L43-L49).  
Pipeline: [`scripts/personal_pipeline.py`](../../scripts/personal_pipeline.py#L22-L60).  
Multi-LoRA: [`scripts/serve_with_lora.sh`](../../scripts/serve_with_lora.sh#L2), [`#L40-L72`](../../scripts/serve_with_lora.sh#L40-L72).

## 8) Always before training

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

## 9) Qwen3.8 startup troubleshooting

Check the last server messages from PowerShell:

```powershell
Get-Content .\output\qwen3.8-server.stdout.log -Tail 100
Get-Content .\output\qwen3.8-server.stderr.log -Tail 100
```

- `Free memory ... less than desired`: close other GPU workloads, then retry.
  Only reduce `GPU_MEM_UTIL` if the initial reservation check still fails.
- `No available memory for the cache blocks`: ensure `EXTRA_ARGS` still
  contains `--kv-cache-memory-bytes 512M`. This explicit allocation avoids the
  variable automatic profiling result that caused the original failure.
- `ninja: command not found`: rerun setup so the `ninja` requirement is
  installed. The launcher automatically adds `.venv/bin` to `PATH`.
- Missing `curand_kernel.h`: run `bash scripts/_install_usergcc.sh`; the CUDA
  environment must include `libcurand-dev`.
- `cannot find -lcuda`: ensure the server is launched through
  `scripts/start_server.sh`, which sources the WSL driver-library path.
- Linux kills `cicc` for out-of-memory: do not remove the `MAX_JOBS=4` and
  `NVCC_THREADS=1` defaults from the launcher.
- The API is unavailable while initialization is running. Wait for
  `Application startup complete` before testing port 8000.

## Privacy / push checklist

- Do not commit HF tokens.
- `data/personal/**/*.jsonl` is gitignored — keep it that way.
- Push when docs + code on `master` are what you want remote; personal datasets stay local.
