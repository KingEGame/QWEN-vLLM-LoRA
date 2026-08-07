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
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
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
