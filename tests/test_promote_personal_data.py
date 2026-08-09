import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.promote_personal_data import _strip_meta


def test_strip_meta_strips_source_and_kind(tmp_path: Path):
    src = tmp_path / "candidates.jsonl"
    src.write_text(
        json.dumps(
            {
                "instruction": "How do I train?",
                "response": "Stop the server first.",
                "source": "docs/x.md",
                "kind": "me_assistant",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out.jsonl"
    n = _strip_meta(src, dest)
    assert n == 1
    row = json.loads(dest.read_text(encoding="utf-8").strip())
    assert row == {"instruction": "How do I train?", "response": "Stop the server first."}


def test_strip_meta_zero_rows(tmp_path: Path):
    src = tmp_path / "empty.jsonl"
    src.write_text("\n\n", encoding="utf-8")
    dest = tmp_path / "out.jsonl"
    assert _strip_meta(src, dest) == 0
    assert dest.read_text(encoding="utf-8") == ""


def test_strip_meta_missing_field(tmp_path: Path, capsys):
    src = tmp_path / "bad.jsonl"
    src.write_text(json.dumps({"instruction": "only instruction"}) + "\n", encoding="utf-8")
    dest = tmp_path / "out.jsonl"
    with pytest.raises(KeyError):
        _strip_meta(src, dest)
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err
    assert "missing field" in captured.err


def test_main_rejects_zero_row_promote(tmp_path: Path, capsys):
    cand = tmp_path / "candidates"
    cand.mkdir()
    (cand / "question_sharp.jsonl").write_text("\n", encoding="utf-8")
    (cand / "me_assistant.jsonl").write_text(
        json.dumps({"instruction": "q", "response": "a", "kind": "me_assistant"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "personal"

    import scripts.promote_personal_data as promote_mod

    with patch.object(promote_mod, "CAND", cand), patch.object(promote_mod, "OUT", out):
        old_argv = sys.argv
        sys.argv = ["promote_personal_data.py", "--reviewed"]
        try:
            rc = promote_mod.main()
        finally:
            sys.argv = old_argv
    captured = capsys.readouterr()
    assert rc == 1
    assert "0 rows" in captured.err
