# Qwen3.8-27B (NVFP4) + vLLM + LoRA

Serve **Qwen3.8-27B** locally through vLLM using the Blackwell-friendly
[`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
checkpoint. The configuration is verified on an RTX 5090 Laptop GPU with 24GB
VRAM under WSL2. It provides an OpenAI-compatible API at
`http://localhost:8000/v1`.

The repository still contains LoRA/QLoRA workflows from the earlier Qwen3.6
deployment. Qwen3.6 adapters are **not compatible** with Qwen3.8; retrain them
against [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) before
attempting to attach them to this serving model.

Authored to run on **Linux or WSL2** with an NVIDIA GPU. Native Windows cannot
run the GPU stack; Windows teammates use the thin setup wrappers below, which
forward into WSL.

## Guides

- [Architecture learning](docs/guides/architecture-learning.md) — original Qwen3.6 architecture and LoRA background
- [Operator cheatsheet](docs/guides/operator-cheatsheet.md) — commands by scenario
- [Qwen3.8 QLoRA training](docs/guides/qwen38-qlora-training.md) — ordered commands, flags, memory limits, and current-system diagram
- [LoRA, QLoRA, and AWQ learning](docs/guides/lora-qlora-learning.md) — legacy Qwen3.6 training/serving formats and compatibility background

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

When setup finishes, activate the venv it created, then start the server from
inside WSL/Linux:

```bash
./scripts/start_server.sh
# in another terminal, with the venv active:
python scripts/test_client.py
```

### Windows/WSL operating commands

From PowerShell, start the server in the foreground:

```powershell
wsl -d Ubuntu -e bash /mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA/scripts/start_server.sh
```

Or keep it attached to a hidden background WSL process and write logs:

```powershell
$qwenServerArgs = @('-d','Ubuntu','-e','bash','/mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA/scripts/start_server.sh')
Start-Process wsl.exe -ArgumentList $qwenServerArgs -WindowStyle Hidden `
  -RedirectStandardOutput '.\output\qwen3.8-server.stdout.log' `
  -RedirectStandardError '.\output\qwen3.8-server.stderr.log'
```

Check it and send the verified smoke test:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
wsl -d Ubuntu -e bash -lc "cd /mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA && .venv/bin/python scripts/test_client.py --prompt 'Reply with exactly: Qwen3.8 local server is working.'"
```

Stop the server cleanly:

```powershell
wsl -d Ubuntu -e bash -lc "pkill -TERM -x vllm"
```

The background server logs are
`output/qwen3.8-server.stdout.log` and `output/qwen3.8-server.stderr.log`.

## Default model (Qwen3.8-27B NVFP4)

`config/model.env` defaults to:

- `MODEL=unsloth/Qwen3.8-27B-NVFP4`
- `QUANTIZATION=none` (quantization is declared by checkpoint metadata)
- `MAX_MODEL_LEN=4096`
- `MAX_NUM_SEQS=1`
- `GPU_MEM_UTIL=0.94`
- `REASONING_PARSER=qwen3`
- `LANGUAGE_MODEL_ONLY=1`
- `EXTRA_ARGS="--kv-cache-dtype fp8 --kv-cache-memory-bytes 512M --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder"`

The downloaded snapshot is about 23.44GB. vLLM loads about 20.47GiB of model
weights into VRAM and uses an explicit 512MiB FP8 KV cache. The explicit cache
size avoids startup failures caused by fluctuating Windows display-GPU memory
being counted during vLLM profiling. The configured request limit is 4,096
tokens.

The official BF16 27B checkpoint needs roughly 54GB for weights alone because
BF16 uses two bytes per parameter. It cannot fit on a single 24GB GPU. NVFP4
stores most weights at approximately four bits and is the practical local
serving format for this machine.

On WSL, `scripts/start_server.sh` sources `scripts/wsl_runtime_env.sh` for a
user-space GCC/CUDA toolkit (micromamba env `cc`) needed by Triton/FlashInfer.
Create it once if missing:

```bash
# one-time: user-space gcc + CUDA 13.3 toolkit (no sudo)
bash scripts/_install_usergcc.sh
```

Blackwell-only env vars (`FLASHINFER_CUDA_ARCH_LIST`, `TORCH_CUDA_ARCH_LIST`)
are set only when `nvidia-smi` reports compute capability major >= 12.
`VLLM_USE_FLASHINFER_SAMPLER` defaults to `0`. FP4 exhaustive autotuning is
also skipped because it was estimated at about 44 minutes on the laptop GPU;
FlashInfer uses its default heuristic tactics instead. Native compilation is
bounded to four jobs to avoid exhausting 32GB of system RAM.

Pass extra vLLM flags via `EXTRA_ARGS`, e.g.
`EXTRA_ARGS="--enforce-eager" ./scripts/start_server.sh`.

If startup reports insufficient free GPU memory, stop other GPU applications
or lower `GPU_MEM_UTIL` slightly. If generation runs out of cache, lower
`MAX_MODEL_LEN`; do not raise it beyond the tested value without rechecking
VRAM usage.

To roll back to the small bf16 model for LoRA experiments, set in `config/model.env`:

```env
MODEL=Qwen/Qwen3-4B-Instruct-2507
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=
QUANTIZATION=none
REASONING_PARSER=
LANGUAGE_MODEL_ONLY=0
```

## LoRA on Qwen3.8-27B (retraining required)

Follow the complete [Qwen3.8 QLoRA training guide](docs/guides/qwen38-qlora-training.md)
for the ordered commands, flag definitions, memory checks, and architecture
diagram.

End-to-end generate → train → serve-with-LoRA using `data/source_docs/example_faq.md`.
Generation uses the running NVFP4 server; training loads the official
`Qwen/Qwen3.8-27B` source with QLoRA. Do not train against the NVFP4 inference
checkpoint, and do not reuse adapters produced for Qwen3.6. The Qwen3.8
training and adapter-serving path should be treated as experimental until a
new adapter has completed an end-to-end validation.

```bash
# 1) Qwen3.8 NVFP4 server already running
python scripts/generate_training_data.py

# 2) first-run promote (light review optional)
mkdir -p data
cp data/generated/raw_qa.jsonl data/train.jsonl
python scripts/validate_dataset.py data/train.jsonl

# 3) stop the vLLM server (Ctrl+C in its terminal), confirm VRAM free:
nvidia-smi

# 4) train (downloads Qwen/Qwen3.8-27B source weights on first run — large)
# Train knobs via env (defaults in train_lora.py): MAX_SEQ_LENGTH, BATCH_SIZE,
# NUM_EPOCHS, GRADIENT_ACCUMULATION_STEPS, GRADIENT_CHECKPOINTING.
# Example 27B smoke: MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
#   GRADIENT_ACCUMULATION_STEPS=8 GRADIENT_CHECKPOINTING=1 python scripts/train_lora.py
python scripts/train_lora.py

# 5) serve base + adapter
./scripts/serve_with_lora.sh
# other terminal:
python scripts/test_client.py --model support-adapter
```

## Personal tech pipeline (question-sharper → me-assistant)

Two newly trained LoRA adapters on the same Qwen3.8 NVFP4 serving base: sharpen messy tech thoughts into clear
questions, then answer in your preferred style. Sources and paths are in
[`config/personal_sources.env`](config/personal_sources.env); full design in
[Personal tech pipeline design](docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md).

**WSL note:** `~/.cursor/...` in config resolves to the WSL home, not Windows
Cursor data. When extracting from WSL, point at the Windows agent-transcripts
folder:

```bash
export AGENT_TRANSCRIPTS_DIR=/mnt/c/Users/supre/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
```

```bash
# 1) extract candidates from transcripts + configured markdown
python scripts/extract_personal_candidates.py
# review/edit under data/personal/candidates/

# 2) promote after human review (strips metadata, validates)
python scripts/promote_personal_data.py --reviewed
python scripts/validate_dataset.py data/personal/question_sharp.jsonl
python scripts/validate_dataset.py data/personal/me_assistant.jsonl

# 3) stop vLLM; confirm VRAM free (nvidia-smi), then train each adapter
# Same train env knobs as above (MAX_SEQ_LENGTH, BATCH_SIZE, NUM_EPOCHS, etc.).
python scripts/train_lora.py --data data/personal/question_sharp.jsonl --output output/lora_question_sharper
python scripts/train_lora.py --data data/personal/me_assistant.jsonl --output output/lora_me_assistant

# 4) serve base + both newly retrained Qwen3.8 adapters
LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" ./scripts/serve_with_lora.sh

# 5) run the chained pipeline (optional run log: output/personal_runs.jsonl)
python scripts/personal_pipeline.py "how do i fix oom when starting vllm on 24gb"
```

Until both Qwen3.8 adapters have been retrained and loaded, use the running
base model for both stages. This disables Qwen thinking so the sharpening
stage returns usable response content:

```bash
python scripts/personal_pipeline.py --base-only "how do i fix oom when starting vllm on 24gb"
```

Personal train/candidate JSONL under `data/personal/` may contain private chat
text — keep it local unless you explicitly version sanitized data.

## Troubleshooting

For the persistent tool-using small assistant, see
[Persistent personal-agent pipeline](docs/guides/personal-agent-pipeline.md).

Driver, CUDA, and out-of-memory issues are documented in:

- [Architecture learning](docs/guides/architecture-learning.md) (Qwen3.6 background)
- [Operator cheatsheet](docs/guides/operator-cheatsheet.md)
- [LoRA, QLoRA, and AWQ learning](docs/guides/lora-qlora-learning.md) (Qwen3.6 background)
- [Design: Qwen + vLLM + LoRA setup](docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md)
- [Design: easy onboard setup scripts](docs/superpowers/specs/2026-08-06-easy-onboard-setup-scripts-design.md)
- [Design: Qwen3.6-27B AWQ serve](docs/superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md)
- [Design: Qwen3.6-27B LoRA train](docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md)
- [Design: personal tech pipeline](docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md)

Tune model/port/context in `config/model.env` — setup does not rewrite it.

## Unit tests (no GPU required)

`pytest` is not installed by `setup.sh`; install it into the venv first:

```bash
pip install pytest
python -m pytest -v
```
