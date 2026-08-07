# Easy Onboard Setup Scripts (Windows + Linux/WSL)

## Goal

Give mixed teammates (Windows+WSL and native Linux) **one setup command** that installs Python deps and verifies GPU/CUDA/vLLM, then stops. Starting the server and running the test client remain manual next steps.

## Background & Constraints

- The repo already has `scripts/setup.sh` (GPU check → `.venv` → `pip install -r requirements.txt` → torch/vLLM verify) and `scripts/start_server.sh`.
- There is no Windows entry point today. Native Windows cannot install/run vLLM; GPU work runs in Linux or WSL2 with NVIDIA GPU passthrough.
- Audience: teammates on **mixed** machines (Windows+WSL and native Linux).
- Success bar: **setup only** — install + verify. Serve/test are not automated by these scripts.
- Windows entry points are **thin wrappers** that detect WSL/Ubuntu and invoke the real Linux `setup.sh`.
- Keep `requirements.txt` **as-is**. No CUDA pin switching, no VRAM auto-profiling, no automatic `model.env` rewriting. Driver/CUDA/OOM troubleshooting stays in docs.

## Architecture

One source of truth on Linux/WSL; Windows only forwards:

```
Windows teammate                 Linux teammate
─────────────────                ──────────────
scripts/setup.cmd  ─┐
scripts/setup.ps1  ─┼─→  wsl -d Ubuntu  →  scripts/setup.sh
                    ┘                         │
                                              ▼
                                    .venv + pip install -r requirements.txt
                                    + nvidia-smi / torch.cuda / vllm checks
```

After success, print activation + next-step commands. Do **not** start the server.

## Components

### `scripts/setup.sh` (source of truth, lightly hardened)

- Check `nvidia-smi`; on failure print a clear WSL GPU-passthrough / Windows driver hint and exit ≠0.
- Create/recreate repo-local `.venv` with `python3 -m venv --clear`.
- Upgrade pip and `pip install -r requirements.txt`.
- Verify `torch.cuda.is_available()` and `import vllm`.
- Print: how to activate the venv, then next manual steps (`./scripts/start_server.sh`, then `python scripts/test_client.py`).
- No CUDA pin logic, no VRAM detection, no edits to `config/model.env`.

### `scripts/setup.cmd` (Windows entry)

- Resolve repo root from the script location.
- Require `wsl` and an Ubuntu distro (prefer a distro whose name is exactly `Ubuntu`, else the first installed name matching `Ubuntu*`); if missing, exit ≠0 with a short install hint (`wsl --install -d Ubuntu`).
- Convert the Windows repo path to a WSL `/mnt/...` path and run `bash scripts/setup.sh` inside that distro.
- Propagate WSL’s exit code.

### `scripts/setup.ps1` (Windows entry, same behavior)

- Same checks and launch path as `.cmd`, for PowerShell users.
- Pure forwarder — no second install implementation.

### Docs (minimal)

- Short onboarding blurb (README or equivalent): Windows → `setup.cmd` / `setup.ps1`; Linux → `./scripts/setup.sh`; then manual serve + test.
- Point driver/CUDA/OOM troubleshooting at existing design docs rather than encoding it in setup.

## Data Flow

1. Teammate clones the repo.
2. Runs `scripts/setup.cmd` or `scripts/setup.ps1` (Windows) or `./scripts/setup.sh` (Linux / already-in-WSL).
3. Windows wrapper validates WSL + Ubuntu, then runs `setup.sh` at the repo’s `/mnt/...` path.
4. `setup.sh` performs GPU check → venv → install → CUDA/vLLM verify.
5. Prints activation + next-step serve/test commands and exits.
6. Teammate runs serve/test manually when ready.

## Error Handling

Fail fast with a clear message; no silent fallbacks or alternate dependency stacks.

| Failure | Behavior |
|---|---|
| No `wsl` / no Ubuntu (Windows wrappers) | Exit ≠0 with install hint |
| No `nvidia-smi` | Exit ≠0 with WSL GPU-passthrough / Windows driver hint |
| Missing `python3` / `venv` | Exit ≠0 with package-install hint |
| pip install / import / CUDA assert fails | Exit ≠0; surface the underlying error |

## Out of Scope

- Windows wrappers for `start_server` / `test_client`
- Auto-selecting CUDA 12.x vs 13.x torch/vLLM builds
- Auto-writing 6GB vs 8GB+ `model.env` profiles
- Unsloth / LoRA training environment setup
- Installing Ubuntu or NVIDIA drivers automatically

## Testing / Verification

- On a GPU machine: run the platform entry once. Success = “Setup complete” plus CUDA OK and vLLM OK lines.
- On the authoring machine: `bash -n scripts/setup.sh`. Review Windows scripts for path/WSL invocation correctness (full GPU install may not be runnable there).
- Existing pytest suite is unchanged and unrelated to these wrappers.

## Relationship to Prior Spec

This extends the pipeline described in `docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md` by adding a Windows-friendly entry to the existing one-time `setup.sh` step. It does not change the serve / data-gen / train / LoRA-serve flow.
