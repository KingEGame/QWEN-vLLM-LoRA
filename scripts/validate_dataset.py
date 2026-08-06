#!/usr/bin/env python3
"""CLI: validate a LoRA training dataset before spending GPU time on it.

Usage: python scripts/validate_dataset.py data/train.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.dataset_validation import validate_dataset_file


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_dataset.py <path-to-train.jsonl>", file=sys.stderr)
        return 2

    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f"Error: {dataset_path} does not exist", file=sys.stderr)
        return 2

    try:
        errors = validate_dataset_file(dataset_path)
    except UnicodeDecodeError as exc:
        print(f"Error: {dataset_path} is not valid UTF-8 text: {exc}", file=sys.stderr)
        return 2

    if errors:
        print(f"FAILED: {len(errors)} invalid line(s) in {dataset_path}")
        for error in errors:
            print(f"  {error}")
        return 1

    line_count = sum(1 for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if line_count == 0:
        print(f"FAILED: {dataset_path} contains no training examples (file is empty or all blank lines)")
        return 1

    print(f"OK: {dataset_path} is valid ({line_count} training examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
