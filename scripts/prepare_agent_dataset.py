#!/usr/bin/env python3
"""Build the small-agent LoRA dataset from explicitly human-approved examples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.dataset_validation import validate_dataset_file
from scripts.lib.env_config import load_env_file


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "data/personal/qwen_dataset_toolkit/personal-assistant-data/curated/human_approved.jsonl"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/personal/agent_train.jsonl"
TEACHER_CURATED = DEFAULT_INPUT.with_name("teacher_curated.jsonl")
EXCLUSION_FILES = (
    DEFAULT_INPUT.with_name("needs_context.jsonl"),
    DEFAULT_INPUT.with_name("needs_edit.jsonl"),
    REPO_ROOT / "data/personal/qwen_dataset_toolkit/personal-assistant-data/rejected/human_rejected.jsonl",
)


def extract_pair(record: dict, allow_teacher_curated: bool = False) -> dict | None:
    status = record.get("review_status")
    provisional = allow_teacher_curated and status == "pending_human_review"
    if status != "human_approved" and not provisional:
        return None
    if provisional and record.get("fact_check_required") is True:
        return None
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    user = next((m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
    assistant = next((m.get("content") for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "assistant"), None)
    if not isinstance(user, str) or not user.strip() or not isinstance(assistant, str) or not assistant.strip():
        return None
    return {"instruction": user.strip(), "response": assistant.strip()}


def build_dataset(
    paths: list[Path],
    *,
    allow_teacher_curated: bool = False,
    excluded_ids: set[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id") in (excluded_ids or set()):
                continue
            pair = extract_pair(record, allow_teacher_curated=allow_teacher_curated)
            if pair is None:
                continue
            key = (pair["instruction"].casefold(), pair["response"].casefold())
            if key not in seen:
                seen.add(key)
                rows.append(pair)
    return rows


def load_excluded_ids(paths: tuple[Path, ...] = EXCLUSION_FILES) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record_id = json.loads(line).get("id")
                if isinstance(record_id, str):
                    excluded.add(record_id)
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum", type=int, default=None)
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument(
        "--include-teacher-curated",
        action="store_true",
        help="Build a provisional v0 from teacher rewrites, excluding context/edit/rejected/fact-check rows",
    )
    args = parser.parse_args()
    config = load_env_file(REPO_ROOT / "config/agent.env")
    minimum = args.minimum if args.minimum is not None else int(config.get("AGENT_MIN_TRAIN_EXAMPLES", "20"))
    inputs = args.input or [DEFAULT_INPUT]
    if args.include_teacher_curated and TEACHER_CURATED not in inputs:
        inputs.append(TEACHER_CURATED)
    rows = build_dataset(
        inputs,
        allow_teacher_curated=args.include_teacher_curated,
        excluded_ids=load_excluded_ids() if args.include_teacher_curated else set(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    errors = validate_dataset_file(args.output)
    if errors:
        print("ERROR: generated dataset failed validation", file=sys.stderr)
        return 1
    label = "approved + provisional teacher-curated" if args.include_teacher_curated else "human-approved"
    print(f"Prepared {len(rows)} {label} examples -> {args.output}")
    if len(rows) < minimum and not args.allow_small:
        print(
            f"ERROR: need at least {minimum} approved examples before training; "
            "continue reviewing or use --allow-small only for a smoke test.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
