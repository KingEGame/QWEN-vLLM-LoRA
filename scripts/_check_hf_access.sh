#!/usr/bin/env bash
set -euo pipefail
source /mnt/c/Users/supre/Documents/QWEN-vLLM-LoRA/.venv/bin/activate
export HF_TOKEN="$(tr -d '\r\n' < "$HOME/.cache/huggingface/token")"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
python <<'PY'
from huggingface_hub import HfApi, model_info, hf_hub_download
import os
token = os.environ.get("HF_TOKEN")
api = HfApi(token=token)
try:
    who = api.whoami()
    print("logged_in_as:", who.get("name") or who.get("fullname"))
except Exception as e:
    print("whoami_error:", type(e).__name__, e)
try:
    info = model_info("Qwen/Qwen3.6-27B", token=token)
    print("model_id:", info.id)
    print("gated:", getattr(info, "gated", None))
    print("private:", getattr(info, "private", None))
    print("disabled:", getattr(info, "disabled", None))
    print("siblings:", len(info.siblings or []))
except Exception as e:
    print("model_info_error:", type(e).__name__, e)
try:
    path = hf_hub_download("Qwen/Qwen3.6-27B", "config.json", token=token)
    print("config_download_ok:", path)
except Exception as e:
    print("config_download_error:", type(e).__name__, e)
PY
