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
- `MAX_MODEL_LEN=4096`
- `MAX_NUM_SEQS=32`
- `GPU_MEM_UTIL=0.92`
- `REASONING_PARSER=qwen3`
- `LANGUAGE_MODEL_ONLY=1`

First server start downloads the AWQ weights from Hugging Face (large).

On WSL, `scripts/start_server.sh` sources `scripts/wsl_runtime_env.sh` for a
user-space GCC/CUDA toolkit (micromamba env `cc`) needed by Triton/FlashInfer.
Create it once if missing:

```bash
# one-time: user-space gcc + CUDA 13.3 toolkit (no sudo)
bash scripts/_install_usergcc.sh
```

Blackwell-only env vars (`FLASHINFER_CUDA_ARCH_LIST`, `TORCH_CUDA_ARCH_LIST`)
are set only when `nvidia-smi` reports compute capability major >= 12.
`VLLM_USE_FLASHINFER_SAMPLER` defaults to `0` (avoids curand.h JIT); override
to `1` if you have full CUDA math headers.

Pass extra vLLM flags via `EXTRA_ARGS`, e.g.
`EXTRA_ARGS="--enforce-eager" ./scripts/start_server.sh`.

If you OOM or hit Mamba-cache errors: lower `MAX_MODEL_LEN` / `MAX_NUM_SEQS`,
or raise `GPU_MEM_UTIL` slightly. If VRAM remains, try raising `MAX_MODEL_LEN`
toward `8192`.

To roll back to the small bf16 model for LoRA experiments, set in `config/model.env`:

```env
MODEL=Qwen/Qwen3-4B-Instruct-2507
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=
QUANTIZATION=none
REASONING_PARSER=
LANGUAGE_MODEL_ONLY=0
```

## LoRA on Qwen3.6-27B (example FAQ)

End-to-end generate → train → serve-with-LoRA using `data/source_docs/example_faq.md`.
Generation uses the running AWQ server; training loads dense `Qwen/Qwen3.6-27B` in 4-bit
(QLoRA cannot train on the AWQ checkpoint).

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
# Knobs live in config/model.env (batch=2, epochs=3, accum=8, ~70% free GPU/CPU).
python scripts/train_lora.py

# 5) serve base + adapter
./scripts/serve_with_lora.sh
# other terminal:
python scripts/test_client.py --model support-adapter
```

## Personal tech pipeline (question-sharper → me-assistant)

Two LoRA adapters on the same AWQ base: sharpen messy tech thoughts into clear
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
python scripts/train_lora.py --data data/personal/question_sharp.jsonl --output output/lora_question_sharper
python scripts/train_lora.py --data data/personal/me_assistant.jsonl --output output/lora_me_assistant

# 4) serve base + both adapters
LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" ./scripts/serve_with_lora.sh

# 5) run the chained pipeline (optional run log: output/personal_runs.jsonl)
python scripts/personal_pipeline.py "how do i fix oom when starting vllm on 24gb"
```

Personal train/candidate JSONL under `data/personal/` may contain private chat
text — keep it local unless you explicitly version sanitized data.

## Troubleshooting

Driver, CUDA, and out-of-memory issues are documented in:

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
