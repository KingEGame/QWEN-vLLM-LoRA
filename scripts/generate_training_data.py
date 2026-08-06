#!/usr/bin/env python3
"""CLI: generate draft customer-support Q&A pairs from data/source_docs/.

Reuses the vLLM server started by scripts/start_server.sh (must already be
running). Writes draft pairs to data/generated/raw_qa.jsonl for human review
-- do NOT feed this file directly into training.

Usage: python scripts/generate_training_data.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from scripts.lib.chunking import chunk_text
from scripts.lib.data_gen import build_generation_prompt, parse_generated_response
from scripts.lib.env_config import load_env_file

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DOCS_DIR = REPO_ROOT / "data" / "source_docs"
OUTPUT_PATH = REPO_ROOT / "data" / "generated" / "raw_qa.jsonl"
PAIRS_PER_CHUNK = 3
MAX_CHUNK_CHARS = 2000


def main() -> int:
    config_path = REPO_ROOT / "config" / "model.env"
    if not config_path.exists():
        print(f"Error: {config_path} not found. Did you run scripts/setup.sh?", file=sys.stderr)
        return 1

    config = load_env_file(config_path)
    model = config.get("MODEL")
    if not model:
        print(f"Error: MODEL not set in {config_path}", file=sys.stderr)
        return 1
    port = config.get("PORT", "8000")

    source_files = sorted(SOURCE_DOCS_DIR.glob("*.md")) + sorted(SOURCE_DOCS_DIR.glob("*.txt"))
    if not source_files:
        print(f"No .md/.txt files found in {SOURCE_DOCS_DIR}", file=sys.stderr)
        return 1

    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        print(f"Error: {OUTPUT_PATH} already exists and is not empty.", file=sys.stderr)
        print("It may contain human-reviewed edits from a previous run -- move or delete it first.", file=sys.stderr)
        return 1

    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    failed_chunks = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_file:
        for source_file in source_files:
            text = source_file.read_text(encoding="utf-8")
            chunks = chunk_text(text, max_chars=MAX_CHUNK_CHARS)

            for chunk in chunks:
                prompt = build_generation_prompt(chunk, num_pairs=PAIRS_PER_CHUNK)
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                    )
                except Exception as exc:
                    failed_chunks += 1
                    print(f"  Warning: a chunk in {source_file.name} failed, skipping: {exc}", file=sys.stderr)
                    continue

                response_text = completion.choices[0].message.content or ""
                pairs = parse_generated_response(response_text)

                for pair in pairs:
                    out_file.write(json.dumps(pair) + "\n")
                    total_pairs += 1

            print(f"{source_file.name}: {len(chunks)} chunk(s) processed")

    print(f"Wrote {total_pairs} draft pairs to {OUTPUT_PATH}")
    if failed_chunks:
        print(f"Warning: {failed_chunks} chunk(s) failed and were skipped (see warnings above)", file=sys.stderr)
    print("Review and edit before copying approved pairs into data/train.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
