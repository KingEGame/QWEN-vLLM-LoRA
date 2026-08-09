#!/usr/bin/env python3
"""Mine transcripts + markdown into personal candidate JSONL files."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.env_config import load_env_file
from scripts.lib.personal_extract import (
    iter_transcript_qa_pairs,
    iter_transcript_user_texts,
    pairs_from_markdown,
    sharpen_candidates_from_texts,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "personal" / "candidates"


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).expanduser()


def _transcripts_root(raw: str) -> Path | None:
    if not (raw or "").strip():
        return None
    return _expand(raw)


def main() -> int:
    cfg = load_env_file(REPO_ROOT / "config" / "personal_sources.env")
    transcripts_raw = os.environ.get("AGENT_TRANSCRIPTS_DIR") or cfg.get("AGENT_TRANSCRIPTS_DIR", "")
    transcripts = _transcripts_root(transcripts_raw)
    globs = (
        os.environ.get("MARKDOWN_GLOBS") or cfg.get("MARKDOWN_GLOBS", "README.md")
    ).split(",")

    sharpen: list[dict] = []
    me: list[dict] = []

    if transcripts is not None and transcripts.is_dir():
        for path in sorted(transcripts.rglob("*.jsonl")):
            src = str(path)
            sharpen.extend(sharpen_candidates_from_texts(iter_transcript_user_texts(path), src))
            for q, a in iter_transcript_qa_pairs(path):
                me.append(
                    {
                        "instruction": q,
                        "response": a,
                        "source": src,
                        "kind": "me_assistant",
                    }
                )
    elif transcripts is not None:
        print(f"WARNING: transcripts dir missing: {transcripts}", file=sys.stderr)
    else:
        print(
            "WARNING: AGENT_TRANSCRIPTS_DIR not set or empty; skipping transcript extraction",
            file=sys.stderr,
        )

    for pattern in globs:
        pattern = pattern.strip()
        if not pattern:
            continue
        for path in sorted(REPO_ROOT.glob(pattern)):
            if not path.is_file():
                continue
            me.extend(pairs_from_markdown(path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sharp_path = OUT_DIR / "question_sharp.jsonl"
    me_path = OUT_DIR / "me_assistant.jsonl"
    sharp_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in sharpen) + ("\n" if sharpen else ""), encoding="utf-8")
    me_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in me) + ("\n" if me else ""), encoding="utf-8")
    print(f"Wrote {len(sharpen)} sharpen candidates → {sharp_path}")
    print(f"Wrote {len(me)} me_assistant candidates → {me_path}")
    print("Review/edit candidates, then: python scripts/promote_personal_data.py --reviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
