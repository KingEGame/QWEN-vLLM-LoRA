#!/usr/bin/env bash
set -euo pipefail
curl -L "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o /tmp/micromamba.tar.bz2
python3 <<'PY'
import tarfile
from pathlib import Path
Path("/tmp/mm").mkdir(exist_ok=True)
with tarfile.open("/tmp/micromamba.tar.bz2", "r:bz2") as tf:
    tf.extractall("/tmp/mm")
print("extracted")
PY
mkdir -p "$HOME/.local/bin"
cp /tmp/mm/bin/micromamba "$HOME/.local/bin/micromamba"
chmod +x "$HOME/.local/bin/micromamba"
"$HOME/.local/bin/micromamba" --version
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
"$HOME/.local/bin/micromamba" create -y -n cc -c conda-forge cxx-compiler c-compiler
"$HOME/.local/bin/micromamba" run -n cc bash -lc 'which gcc; gcc --version | head -1'
