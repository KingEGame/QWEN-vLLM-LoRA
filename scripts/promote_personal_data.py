#!/usr/bin/env python3
"""Promote reviewed personal candidates to train JSONL (instruction/response only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.dataset_validation import validate_dataset_file

REPO_ROOT = Path(__file__).resolve().parent.parent
CAND = REPO_ROOT / "data" / "personal" / "candidates"
OUT = REPO_ROOT / "data" / "personal"


def _strip_meta(path: Path, dest: Path) -> int:
    rows: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"ERROR: {path}:{i} invalid JSON: {exc}", file=sys.stderr)
            raise
        try:
            row = {"instruction": obj["instruction"], "response": obj["response"]}
        except KeyError as exc:
            print(f"ERROR: {path}:{i} missing field {exc}", file=sys.stderr)
            raise
        rows.append(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviewed",
        action="store_true",
        help="Required confirmation that candidates were human-reviewed",
    )
    args = parser.parse_args()
    if not args.reviewed:
        print("ERROR: refusing to promote without --reviewed", file=sys.stderr)
        return 1

    mapping = [
        (CAND / "question_sharp.jsonl", OUT / "question_sharp.jsonl"),
        (CAND / "me_assistant.jsonl", OUT / "me_assistant.jsonl"),
    ]
    for src, dest in mapping:
        if not src.exists():
            print(f"ERROR: missing {src}", file=sys.stderr)
            return 1
        try:
            n = _strip_meta(src, dest)
        except (json.JSONDecodeError, KeyError):
            return 1
        if n == 0:
            print(f"ERROR: {src} produced 0 rows after promote", file=sys.stderr)
            return 1
        errors = validate_dataset_file(dest)
        if errors:
            print(f"ERROR: {dest} invalid:", file=sys.stderr)
            print("\n".join(errors[:20]), file=sys.stderr)
            return 1
        print(f"Promoted {n} rows → {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
