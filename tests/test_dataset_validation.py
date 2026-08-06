from pathlib import Path

from scripts.lib.dataset_validation import validate_dataset_file, validate_jsonl_line


def test_validate_jsonl_line_accepts_valid_pair():
    line = '{"instruction": "How do I reset my password?", "response": "Go to Settings > Security."}'

    is_valid, error = validate_jsonl_line(line)

    assert is_valid is True
    assert error == ""


def test_validate_jsonl_line_rejects_invalid_json():
    is_valid, error = validate_jsonl_line("{not json")

    assert is_valid is False
    assert "invalid JSON" in error


def test_validate_jsonl_line_rejects_missing_field():
    line = '{"instruction": "How do I reset my password?"}'

    is_valid, error = validate_jsonl_line(line)

    assert is_valid is False
    assert "response" in error


def test_validate_jsonl_line_rejects_empty_response():
    line = '{"instruction": "How do I reset my password?", "response": "   "}'

    is_valid, error = validate_jsonl_line(line)

    assert is_valid is False
    assert "response" in error


def test_validate_jsonl_line_allows_blank_line():
    is_valid, error = validate_jsonl_line("   ")

    assert is_valid is True


def test_validate_dataset_file_reports_line_numbers(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"instruction": "Q1", "response": "A1"}\n'
        '{not json}\n'
        '{"instruction": "Q3", "response": "A3"}\n'
    )

    errors = validate_dataset_file(dataset)

    assert len(errors) == 1
    assert errors[0].startswith("line 2:")


def test_validate_dataset_file_returns_empty_for_valid_file(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"instruction": "Q1", "response": "A1"}\n')

    errors = validate_dataset_file(dataset)

    assert errors == []
