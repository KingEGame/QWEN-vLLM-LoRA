"""Validation logic for LoRA training data in JSONL format.

Each line must be a JSON object with non-empty "instruction" and
"response" string fields.
"""
import json
from pathlib import Path

REQUIRED_FIELDS = ("instruction", "response")


def validate_jsonl_line(line: str) -> tuple[bool, str]:
    """Validate a single JSONL line. Returns (is_valid, error_message)."""
    stripped = line.strip()
    if not stripped:
        return True, ""  # blank lines are allowed and ignored by the trainer

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc}"

    if not isinstance(obj, dict):
        return False, "line is not a JSON object"

    for field in REQUIRED_FIELDS:
        if field not in obj:
            return False, f"missing required field '{field}'"
        if not isinstance(obj[field], str) or not obj[field].strip():
            return False, f"field '{field}' must be a non-empty string"

    return True, ""


def validate_dataset_file(path: Path) -> list[str]:
    """Validate every line of a JSONL dataset file.

    Returns a list of error strings, one per invalid line, formatted as
    "line N: <error>". An empty list means the file is valid.
    """
    errors: list[str] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        is_valid, error = validate_jsonl_line(line)
        if not is_valid:
            errors.append(f"line {line_number}: {error}")
    return errors
