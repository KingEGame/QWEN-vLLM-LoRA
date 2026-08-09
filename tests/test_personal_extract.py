from pathlib import Path

from scripts.lib.personal_extract import (
    draft_sharpen,
    extract_user_query,
    iter_transcript_qa_pairs,
    iter_transcript_user_texts,
    pairs_from_markdown,
)


def test_extract_user_query_from_wrapper():
    raw = "<user_query>\nokey can we fix the train OOM?\n</user_query>"
    assert extract_user_query(raw) == "okey can we fix the train OOM?"


def test_extract_user_query_plain_fallback():
    assert extract_user_query("plain question about LoRA") == "plain question about LoRA"


def test_draft_sharpen_collapses_whitespace():
    messy = "okey   so like\n\ncan we train  two adapters??"
    sharp = draft_sharpen(messy)
    assert "  " not in sharp
    assert "?" in sharp or sharp.endswith("adapters")


def test_iter_transcript_user_texts(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\\nfix download\\n</user_query>"}]}}\n'
        '{"role":"assistant","message":{"content":[{"type":"text","text":"checking..."}]}}\n',
        encoding="utf-8",
    )
    texts = iter_transcript_user_texts(p)
    assert texts == ["fix download"]


def test_iter_transcript_qa_pairs(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\\nhow do I serve LoRA?\\n</user_query>"}]}}\n'
        '{"role":"assistant","message":{"content":[{"type":"text","text":"Use serve_with_lora.sh"}]}}\n',
        encoding="utf-8",
    )
    pairs = iter_transcript_qa_pairs(p)
    assert pairs == [("how do I serve LoRA?", "Use serve_with_lora.sh")]


def test_pairs_from_markdown_heading_chunks():
    md = "# Serve\n\nUse AWQ + adapter.\n\n# Train\n\nStop server first.\n"
    pairs = pairs_from_markdown(md, source="docs/x.md")
    assert len(pairs) >= 1
    assert all(p["kind"] == "me_assistant" for p in pairs)
    assert all(p["instruction"] and p["response"] for p in pairs)
