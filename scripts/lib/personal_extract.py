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
    if text.startswith("{") and '"role"' in text:
        return None
    return text


def _is_trivial_sharpen(messy: str, sharp: str) -> bool:
    """True when draft only capitalizes a short single-token input and adds ?."""
    cleaned = re.sub(r"\s+", " ", (messy or "").strip())
    if not cleaned or " " in cleaned or len(cleaned) >= 20:
        return False
    expected = cleaned[0].upper() + cleaned[1:]
    if expected[-1] not in ".?!":
        expected += "?"
    return sharp == expected


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
    out: list[dict] = []
    parts = re.findall(
        r"(?m)^(#{1,2}\s+[^\n]+)$" r"(.*?)(?=^#{1,2}\s+|\Z)", text, flags=re.DOTALL
    )
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
        if len(t.strip()) < 20:
            continue
        sharp = draft_sharpen(t)
        if not sharp or _is_trivial_sharpen(t, sharp):
            continue
        out.append(
            {
                "instruction": t,
                "response": sharp,
                "source": source,
                "kind": "sharpen",
            }
        )
    return out
