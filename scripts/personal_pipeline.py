#!/usr/bin/env python3
"""Run personal tech pipeline: sharpen question, then answer as me-assistant."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from scripts.lib.env_config import load_env_file
from scripts.lib.personal_pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "output" / "personal_runs.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Messy tech thought / question")
    parser.add_argument("--sharp-model", default="question-sharper")
    parser.add_argument("--answer-model", default="me-assistant")
    parser.add_argument(
        "--base-only",
        action="store_true",
        help=(
            "use config/model.env MODEL for both stages and disable Qwen "
            "thinking; useful until Qwen3.8 personal adapters are trained"
        ),
    )
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    config = load_env_file(REPO_ROOT / "config" / "model.env")
    port = config.get("PORT", "8000")
    sharp_model = config["MODEL"] if args.base_only else args.sharp_model
    answer_model = config["MODEL"] if args.base_only else args.answer_model
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

    try:
        result = run_pipeline(
            client,
            args.prompt,
            sharp_model=sharp_model,
            answer_model=answer_model,
            base_only=args.base_only,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Raw: {result['raw']}")
    print(f"Sharpened: {result['sharpened']}")
    print(f"Answer: {result['answer']}")

    if not args.no_log:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Logged → {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
