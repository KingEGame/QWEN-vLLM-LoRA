#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

echo "== Checking for NVIDIA GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. On WSL2, make sure GPU passthrough is set up" >&2
    echo "(install the Windows NVIDIA driver with WSL support; do not install a separate Linux driver)." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== Creating virtual environment at $VENV_DIR =="
python3 -m venv --clear "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "== Installing dependencies =="
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo "== Verifying vLLM and CUDA =="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print('CUDA OK:', torch.cuda.get_device_name(0))"
python3 -c "import vllm; print('vLLM OK:', vllm.__version__)"

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
