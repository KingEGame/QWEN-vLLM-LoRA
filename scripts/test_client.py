#!/usr/bin/env python3
"""CLI: send one test chat request to the running vLLM server.

Verifies the server is actually serving correctly. Exit code 0 plus a
printed response means success.

Usage:
  python scripts/test_client.py                            # hits the base model
  python scripts/test_client.py --model support-adapter    # hits the LoRA adapter
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from scripts.lib.env_config import load_env_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Model name to query (defaults to config/model.env MODEL)")
    parser.add_argument("--prompt", default="What can you help me with?", help="Prompt to send")
    args = parser.parse_args()

    config = load_env_file(REPO_ROOT / "config" / "model.env")
    port = config.get("PORT", "8000")
    model = args.model or config["MODEL"]

    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": args.prompt}],
        )
    except Exception as exc:
        print(f"FAILED: could not reach server or get a response: {exc}", file=sys.stderr)
        return 1

    reply = completion.choices[0].message.content
    print(f"Model: {model}")
    print(f"Prompt: {args.prompt}")
    print(f"Response: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
