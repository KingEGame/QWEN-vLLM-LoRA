import json
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.extract_personal_candidates import _transcripts_root, main


def test_transcripts_root_empty():
    assert _transcripts_root("") is None
    assert _transcripts_root("   ") is None


def test_transcripts_root_expands_path(tmp_path: Path):
    d = tmp_path / "transcripts"
    d.mkdir()
    assert _transcripts_root(str(d)) == d


def test_empty_transcripts_dir_skips_rglob_cwd(tmp_path: Path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    cwd_jsonl = repo / "should_not_scan.jsonl"
    cwd_jsonl.write_text(
        '{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\\nsecret\\n</user_query>"}]}}\n',
        encoding="utf-8",
    )
    out_dir = repo / "data" / "personal" / "candidates"
    monkeypatch.chdir(repo)
    monkeypatch.setenv("AGENT_TRANSCRIPTS_DIR", "")
    monkeypatch.setenv("MARKDOWN_GLOBS", "missing_glob_*.md")

    import scripts.extract_personal_candidates as extract_mod

    with patch.object(extract_mod, "REPO_ROOT", repo), patch.object(
        extract_mod, "OUT_DIR", out_dir
    ), patch.object(extract_mod, "load_env_file", return_value={}):
        old_argv = sys.argv
        sys.argv = ["extract_personal_candidates.py"]
        try:
            rc = extract_mod.main()
        finally:
            sys.argv = old_argv

    captured = capsys.readouterr()
    assert rc == 0
    assert "skipping transcript extraction" in captured.err.lower()
    sharp = out_dir / "question_sharp.jsonl"
    assert sharp.exists()
    assert "secret" not in sharp.read_text(encoding="utf-8")
