# Easy Onboard Setup Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add thin Windows (`setup.cmd` / `setup.ps1`) entry points and lightly harden `setup.sh` so mixed Windows+WSL and Linux teammates can run one setup command that installs deps and verifies GPU/CUDA/vLLM.

**Architecture:** `scripts/setup.sh` remains the single real installer (venv + `requirements.txt` + CUDA/vLLM checks). Windows scripts only locate WSL+Ubuntu, map the repo path to `/mnt/...`, and invoke that bash script. No CUDA pin switching, no VRAM auto-config, no server start.

**Tech Stack:** bash, Windows `cmd.exe`, PowerShell, WSL2 Ubuntu, existing `requirements.txt` / vLLM stack.

## Global Constraints

- Keep `requirements.txt` as-is — no version pin changes in this plan.
- Do not rewrite `config/model.env`.
- Do not start the vLLM server or run `test_client.py` from setup.
- Windows wrappers are pure forwarders — no second install implementation.
- Prefer WSL distro name exactly `Ubuntu`, else first installed name matching `Ubuntu*`.
- Fail fast with clear messages; no silent fallbacks or alternate dependency stacks.
- Shell scripts under `scripts/*.sh` must stay LF (`*.sh text eol=lf` in `.gitattributes`).

## File Structure

```
scripts/
  setup.sh          # MODIFY — harden checks + next-step printout
  setup.cmd         # CREATE — Windows cmd → WSL → setup.sh
  setup.ps1         # CREATE — PowerShell → WSL → setup.sh
README.md           # CREATE — short onboarding blurb (repo has none today)
```

No new Python modules. Existing pytest suite is untouched.

---

### Task 1: Harden `scripts/setup.sh`

**Files:**
- Modify: `scripts/setup.sh`
- Test: syntax check via `bash -n` (no new pytest; GPU install is environment-dependent)

**Interfaces:**
- Consumes: `requirements.txt`, system `python3` / `nvidia-smi`
- Produces: repo-local `.venv/`; stdout ending with activation + next-step commands; exit 0 on success

- [ ] **Step 1: Replace `scripts/setup.sh` with the hardened version**

Full file contents:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

echo "== Checking for NVIDIA GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found." >&2
    echo "On WSL2, install/update the Windows NVIDIA driver with WSL support;" >&2
    echo "do not install a separate Linux NVIDIA driver inside WSL." >&2
    echo "On native Linux, install the NVIDIA proprietary driver and retry." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== Checking for Python 3 + venv =="
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found. On Ubuntu/Debian: sudo apt-get install -y python3 python3-venv python3-pip" >&2
    exit 1
fi
if ! python3 -c "import venv" >/dev/null 2>&1; then
    echo "ERROR: Python venv module missing. On Ubuntu/Debian: sudo apt-get install -y python3-venv" >&2
    exit 1
fi

echo "== Creating virtual environment at $VENV_DIR =="
python3 -m venv --clear "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "== Installing dependencies =="
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo "== Verifying vLLM and CUDA =="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print('CUDA OK:', torch.cuda.get_device_name(0))"
python3 -c "import vllm; print('vLLM OK:', vllm.__version__)"

echo ""
echo "Setup complete."
echo "Activate with:  source $VENV_DIR/bin/activate"
echo "Next steps (manual):"
echo "  1. ./scripts/start_server.sh"
echo "  2. python scripts/test_client.py"
```

- [ ] **Step 2: Syntax-check the script**

Run (from repo root, Git Bash or WSL):

```bash
bash -n scripts/setup.sh && echo "setup.sh: OK"
```

Expected: `setup.sh: OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/setup.sh
git commit -m "feat: harden setup.sh with python checks and next-step hints"
```

---

### Task 2: Add `scripts/setup.cmd` (Windows → WSL)

**Files:**
- Create: `scripts/setup.cmd`

**Interfaces:**
- Consumes: `wsl.exe`, an Ubuntu (or `Ubuntu*`) distro, `scripts/setup.sh`
- Produces: same side effects as `setup.sh` inside WSL; exits with WSL’s exit code

- [ ] **Step 1: Create `scripts/setup.cmd`**

Full file contents:

```bat
@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Resolve repo root = parent of this scripts\ directory
set "SCRIPT_DIR=%~dp0"
rem Strip trailing backslash for path math
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "REPO_ROOT=%%~fI"

where wsl >nul 2>&1
if errorlevel 1 (
  echo ERROR: wsl.exe not found. Install WSL2 first:
  echo   wsl --install -d Ubuntu
  echo Then reboot and re-run scripts\setup.cmd
  exit /b 1
)

rem Prefer exact name Ubuntu; else first Ubuntu* from `wsl -l -q`
set "DISTRO="
for /f "usebackq delims=" %%D in (`wsl -l -q`) do (
  if /I "%%D"=="Ubuntu" set "DISTRO=Ubuntu"
)
if not defined DISTRO (
  for /f "usebackq delims=" %%D in (`wsl -l -q`) do (
    echo %%D | findstr /I /B "Ubuntu" >nul && (
      if not defined DISTRO set "DISTRO=%%D"
    )
  )
)
if not defined DISTRO (
  echo ERROR: No Ubuntu WSL distro found. Install one with:
  echo   wsl --install -d Ubuntu
  exit /b 1
)

rem Convert Windows path to WSL /mnt/<drive>/...
for /f "usebackq delims=" %%P in (`wsl -d "%DISTRO%" -e wslpath -a "%REPO_ROOT%"`) do set "WSL_REPO=%%P"
if not defined WSL_REPO (
  echo ERROR: Failed to convert repo path to a WSL path: %REPO_ROOT%
  exit /b 1
)

echo Using WSL distro: %DISTRO%
echo Repo in WSL: %WSL_REPO%
echo Running scripts/setup.sh ...

wsl -d "%DISTRO%" -e bash -lc "cd \"%WSL_REPO%\" && bash scripts/setup.sh"
exit /b %ERRORLEVEL%
```

- [ ] **Step 2: Smoke-check the wrapper without running the full install**

From PowerShell at the repo root (does not install deps if you stop after the dry path check — here we only verify the script exists and `wsl` is callable):

```powershell
Test-Path .\scripts\setup.cmd
wsl -l -q
```

Expected: `True`, and at least one Ubuntu-related distro name listed (or a clear failure later when someone without Ubuntu runs setup — that is intentional).

Optional deeper check (skips pip if you interrupt): open `cmd.exe` and run `scripts\setup.cmd` on a machine with WSL+GPU when ready to actually install.

- [ ] **Step 3: Commit**

```bash
git add scripts/setup.cmd
git commit -m "feat: add Windows setup.cmd wrapper that runs setup.sh in WSL"
```

---

### Task 3: Add `scripts/setup.ps1` (Windows → WSL)

**Files:**
- Create: `scripts/setup.ps1`

**Interfaces:**
- Consumes: same as `setup.cmd` (`wsl`, Ubuntu/`Ubuntu*`, `setup.sh`)
- Produces: same as `setup.cmd`

- [ ] **Step 1: Create `scripts/setup.ps1`**

Full file contents:

```powershell
#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

function Get-UbuntuDistroName {
    $names = @(wsl -l -q 2>$null | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ })
    if (-not $names) {
        return $null
    }
    $exact = $names | Where-Object { $_ -eq "Ubuntu" } | Select-Object -First 1
    if ($exact) {
        return $exact
    }
    return ($names | Where-Object { $_ -like "Ubuntu*" } | Select-Object -First 1)
}

if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Error @"
wsl.exe not found. Install WSL2 first:
  wsl --install -d Ubuntu
Then reboot and re-run scripts\setup.ps1
"@
    exit 1
}

$distro = Get-UbuntuDistroName
if (-not $distro) {
    Write-Error @"
No Ubuntu WSL distro found. Install one with:
  wsl --install -d Ubuntu
"@
    exit 1
}

$wslRepo = (wsl -d $distro -e wslpath -a $RepoRoot).Trim()
if (-not $wslRepo) {
    Write-Error "Failed to convert repo path to a WSL path: $RepoRoot"
    exit 1
}

Write-Host "Using WSL distro: $distro"
Write-Host "Repo in WSL: $wslRepo"
Write-Host "Running scripts/setup.sh ..."

wsl -d $distro -e bash -lc "cd `"$wslRepo`" && bash scripts/setup.sh"
exit $LASTEXITCODE
```

- [ ] **Step 2: Syntax-check PowerShell**

```powershell
powershell -NoProfile -Command "& { $null = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path '.\scripts\setup.ps1'), [ref]$null, [ref]$errs); if ($errs) { $errs | ForEach-Object { $_.ToString() }; exit 1 } else { 'setup.ps1: OK' } }"
```

Expected: `setup.ps1: OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/setup.ps1
git commit -m "feat: add Windows setup.ps1 wrapper that runs setup.sh in WSL"
```

---

### Task 4: Add short README onboarding blurb

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the three setup entry points from Tasks 1–3
- Produces: human-facing day-1 instructions; links to existing design docs for troubleshooting

- [ ] **Step 1: Create `README.md`**

Full file contents:

```markdown
# Qwen3-4B + vLLM + LoRA

Serve Qwen3-4B via vLLM, then customize it for customer support with a LoRA
adapter trained on your own docs.

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

## Troubleshooting

Driver, CUDA, and out-of-memory issues are documented in:

- [Design: Qwen + vLLM + LoRA setup](docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md)
- [Design: easy onboard setup scripts](docs/superpowers/specs/2026-08-06-easy-onboard-setup-scripts-design.md)

Tune model/port/context in `config/model.env` — setup does not rewrite it.

## Unit tests (no GPU required)

```bash
python -m pytest -v
```
```

- [ ] **Step 2: Skim for accuracy**

Confirm the README names match the files created in Tasks 1–3 (`setup.sh`, `setup.cmd`, `setup.ps1`, `start_server.sh`, `test_client.py`).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with Windows/Linux setup onboarding"
```

---

### Task 5: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Syntax-check bash**

```bash
bash -n scripts/setup.sh && echo "setup.sh: OK"
```

Expected: `setup.sh: OK`

- [ ] **Step 2: Confirm Windows entry files exist**

```powershell
Get-Item scripts\setup.cmd, scripts\setup.ps1, scripts\setup.sh, README.md | Select-Object Name, Length
```

Expected: all four present, non-zero length.

- [ ] **Step 3: Parse-check PowerShell**

```powershell
powershell -NoProfile -Command "& { $null = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path '.\scripts\setup.ps1'), [ref]$null, [ref]$errs); if ($errs) { $errs | ForEach-Object { $_.ToString() }; exit 1 } else { 'setup.ps1: OK' } }"
```

Expected: `setup.ps1: OK`

- [ ] **Step 4 (GPU machine only): Optional live setup**

On a machine with WSL2 Ubuntu + NVIDIA GPU passthrough:

```bat
scripts\setup.cmd
```

Expected: `CUDA OK: ...`, `vLLM OK: ...`, and the “Next steps (manual)” lines. Do **not** require this step on the authoring machine if GPU/WSL install is already known-good or blocked.

- [ ] **Step 5: Confirm git history**

```bash
git log --oneline -5
```

Expected: commits for Tasks 1–4 present (harden setup.sh, setup.cmd, setup.ps1, README).

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| Harden `setup.sh` (GPU hint, venv, install, verify, next steps) | Task 1 |
| `setup.cmd` thin WSL forwarder + Ubuntu selection + path convert | Task 2 |
| `setup.ps1` same behavior | Task 3 |
| Minimal onboarding docs + troubleshooting pointers | Task 4 |
| Authoring-machine syntax checks; GPU live run optional | Task 5 |
| Keep `requirements.txt` / `model.env` untouched; no serve/test wrappers | Global Constraints + Out of Scope (no tasks add them) |
