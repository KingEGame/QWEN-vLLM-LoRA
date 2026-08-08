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
