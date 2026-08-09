### Task 2: Personal extract library (TDD)

**Files:**
- Create: `scripts/lib/personal_extract.py`
- Test: `tests/test_personal_extract.py`

**Interfaces:**
- Consumes: Cursor transcript JSONL lines with `role` + `message.content[].text`; markdown files as text
- Produces:
  - `extract_user_query(text: str) -> str | None`
  - `iter_transcript_user_texts(path: Path) -> list[str]`
  - `iter_transcript_qa_pairs(path: Path) -> list[tuple[str, str]]`
  - `draft_sharpen(messy: str) -> str`
  - `pairs_from_markdown(text: str, source: str) -> list[dict]`
  - Candidate dict shape: `{"instruction": str, "response": str, "source": str, "kind": "sharpen"|"me_assistant"}`

- [ ] **Step 1: Write failing tests** in `tests/test_personal_extract.py`

```python
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
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_personal_extract.py -v
```

Expected: import / missing module failures.

- [ ] **Step 3: Implement `scripts/lib/personal_extract.py`**

```python
"""Extract personal LoRA candidate pairs from Cursor transcripts and markdown."""
from __future__ import annotations

import json
import re
from pathlib import Path

_USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL | re.IGNORECASE)


def extract_user_query(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    m = _USER_QUERY_RE.search(text)
    if m:
        q = m.group(1).strip()
        return q or None
    # Skip obvious system/tool dumps
    if text.startswith("{" ) and '"role"' in text:
        return None
    return text


def draft_sharpen(messy: str) -> str:
    """Heuristic draft only — human must review before train promote."""
    cleaned = re.sub(r"\s+", " ", (messy or "").strip())
    if not cleaned:
        return ""
    if cleaned[-1] not in ".?!":
        cleaned += "?"
    # Prefer a single question-shaped line
    if len(cleaned) > 240:
        cleaned = cleaned[:237].rstrip() + "..."
    return cleaned[0].upper() + cleaned[1:] if cleaned else ""


def _message_text(obj: dict) -> str:
    msg = obj.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def iter_transcript_user_texts(path: Path) -> list[str]:
    out: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("role") != "user":
            continue
        q = extract_user_query(_message_text(obj))
        if q:
            out.append(q)
    return out


def iter_transcript_qa_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = obj.get("role")
        text = _message_text(obj)
        if role == "user":
            pending = extract_user_query(text)
        elif role == "assistant" and pending:
            # First text paragraph only — drop huge tool dumps
            reply = text.strip().split("\n\n")[0].strip()
            if reply and len(reply) < 4000:
                pairs.append((pending, reply))
            pending = None
    return pairs


def pairs_from_markdown(text: str, source: str) -> list[dict]:
    """Turn markdown H1/H2 sections into me_assistant candidates."""
    sections = re.split(r"(?m)^#{1,2}\s+", text)
    out: list[dict] = []
    # sections[0] is preface before first heading
    parts = re.findall(r"(?m)^(#{1,2}\s+.+)$(.*?)(?=^#{1,2}\s+|\Z)", text, flags=re.DOTALL)
    if not parts:
        body = text.strip()
        if body:
            title = Path(source).stem.replace("-", " ")
            out.append(
                {
                    "instruction": f"What should I know about {title}?",
                    "response": body[:2000],
                    "source": source,
                    "kind": "me_assistant",
                }
            )
        return out
    for heading_line, body in parts:
        title = re.sub(r"^#{1,2}\s+", "", heading_line).strip()
        body = body.strip()
        if not title or not body:
            continue
        out.append(
            {
                "instruction": f"Explain: {title}",
                "response": body[:2000],
                "source": source,
                "kind": "me_assistant",
            }
        )
    return out


def sharpen_candidates_from_texts(texts: list[str], source: str) -> list[dict]:
    out: list[dict] = []
    for t in texts:
        sharp = draft_sharpen(t)
        if not sharp or sharp == t:
            # still keep if messy enough (length or newlines originally)
            if len(t) < 20:
                continue
        out.append(
            {
                "instruction": t,
                "response": sharp or draft_sharpen(t),
                "source": source,
                "kind": "sharpen",
            }
        )
    return out
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_personal_extract.py -v
```

Expected: all PASSED. If `test_draft_sharpen_*` is brittle, adjust assertion to match `draft_sharpen` behavior without weakening the function.

- [ ] **Step 5: Commit** (if requested)

```bash
git add scripts/lib/personal_extract.py tests/test_personal_extract.py
git commit -m "feat: extract personal LoRA candidates from transcripts and markdown"
```

---

