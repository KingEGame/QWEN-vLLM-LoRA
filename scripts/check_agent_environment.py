#!/usr/bin/env python3
"""Check personal-agent dataset, runtime, model, and export prerequisites."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.env_config import load_env_file


REPO_ROOT = Path(__file__).resolve().parent.parent


def relative_path(config: dict[str, str], key: str, default: str) -> Path:
    path = Path(config.get(key, default))
    return path if path.is_absolute() else REPO_ROOT / path


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def collect_status(config: dict[str, str]) -> dict:
    train_data = relative_path(config, "AGENT_TRAIN_DATA", "data/personal/agent_train.jsonl")
    adapter = relative_path(config, "AGENT_ADAPTER_OUTPUT", "output/lora_personal_agent")
    llama_cpp = relative_path(config, "LLAMA_CPP_DIR", "external/llama.cpp")
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("openai", "torch", "transformers", "peft", "trl", "datasets")
    }
    return {
        "agent_model": config.get("AGENT_MODEL"),
        "base_model": config.get("AGENT_BASE_MODEL"),
        "teacher_model": config.get("AGENT_TEACHER_MODEL"),
        "train_data": str(train_data),
        "train_examples": count_jsonl(train_data),
        "minimum_train_examples": int(config.get("AGENT_MIN_TRAIN_EXAMPLES", "20")),
        "adapter_exists": adapter.is_dir(),
        "llama_cpp_exists": llama_cpp.is_dir(),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "packages": packages,
        "task_sync_provider": config.get("AGENT_TASK_SYNC_PROVIDER", "local"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-training-ready", action="store_true")
    args = parser.parse_args()
    config_path = REPO_ROOT / "config/agent.env"
    if not config_path.exists():
        print(f"ERROR: missing {config_path}", file=sys.stderr)
        return 1
    status = collect_status(load_env_file(config_path))
    print(json.dumps(status, indent=2, sort_keys=True))
    if args.require_training_ready:
        ready = (
            status["train_examples"] >= status["minimum_train_examples"]
            and status["nvidia_smi_available"]
            and all(status["packages"].values())
        )
        return 0 if ready else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
