import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_dataset.py"


def test_cli_exits_zero_for_valid_dataset(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"instruction": "Q1", "response": "A1"}\n')

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(dataset)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK" in result.stdout


def test_cli_exits_one_for_invalid_dataset(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{not json}\n")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(dataset)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "FAILED" in result.stdout


def test_cli_exits_two_for_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.jsonl"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(missing)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
