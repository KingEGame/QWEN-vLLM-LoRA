### Task 3: Extract + promote CLIs

**Files:**
- Create: `scripts/extract_personal_candidates.py`
- Create: `scripts/promote_personal_data.py`
- Modify: none required in lib beyond Task 2

**Interfaces:**
- Consumes: `config/personal_sources.env`, `load_env_file`, extract helpers
- Produces:
  - `data/personal/candidates/question_sharp.jsonl`
  - `data/personal/candidates/me_assistant.jsonl`
  - On promote: `data/personal/question_sharp.jsonl`, `data/personal/me_assistant.jsonl` (instruction/response only)
- Promote requires `--reviewed` flag (no silent promote)

- [ ] **Step 1: Implement `scripts/extract_personal_candidates.py`**

```python
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


def main() -> int:
    cfg = load_env_file(REPO_ROOT / "config" / "personal_sources.env")
    transcripts = _expand(
        os.environ.get("AGENT_TRANSCRIPTS_DIR") or cfg.get("AGENT_TRANSCRIPTS_DIR", "")
    )
    globs = (
        os.environ.get("MARKDOWN_GLOBS") or cfg.get("MARKDOWN_GLOBS", "README.md")
    ).split(",")

    sharpen: list[dict] = []
    me: list[dict] = []

    if transcripts.is_dir():
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
    else:
        print(f"WARNING: transcripts dir missing: {transcripts}", file=sys.stderr)

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
```

- [ ] **Step 2: Implement `scripts/promote_personal_data.py`**

```python
#!/usr/bin/env python3
"""Promote reviewed personal candidates to train JSONL (instruction/response only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.dataset_validation import validate_dataset_file

REPO_ROOT = Path(__file__).resolve().parent.parent
CAND = REPO_ROOT / "data" / "personal" / "candidates"
OUT = REPO_ROOT / "data" / "personal"


def _strip_meta(path: Path, dest: Path) -> int:
    rows: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        row = {"instruction": obj["instruction"], "response": obj["response"]}
        rows.append(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviewed",
        action="store_true",
        help="Required confirmation that candidates were human-reviewed",
    )
    args = parser.parse_args()
    if not args.reviewed:
        print("ERROR: refusing to promote without --reviewed", file=sys.stderr)
        return 1

    mapping = [
        (CAND / "question_sharp.jsonl", OUT / "question_sharp.jsonl"),
        (CAND / "me_assistant.jsonl", OUT / "me_assistant.jsonl"),
    ]
    for src, dest in mapping:
        if not src.exists():
            print(f"ERROR: missing {src}", file=sys.stderr)
            return 1
        n = _strip_meta(src, dest)
        errors = validate_dataset_file(dest)
        if errors:
            print(f"ERROR: {dest} invalid:", file=sys.stderr)
            print("\n".join(errors[:20]), file=sys.stderr)
            return 1
        print(f"Promoted {n} rows → {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Smoke extract (no GPU)**

```bash
python scripts/extract_personal_candidates.py
python scripts/promote_personal_data.py
# expect ERROR without --reviewed
python scripts/promote_personal_data.py --reviewed
python scripts/validate_dataset.py data/personal/question_sharp.jsonl
python scripts/validate_dataset.py data/personal/me_assistant.jsonl
```

Expected: candidate counts printed; promote without flag fails; with flag validates clean (after any empty-file edge: ensure extract produced ≥1 row or skip validate with a clear message — if zero rows, print ERROR and exit 1 in promote).

- [ ] **Step 4: Commit** (if requested)

```bash
git add scripts/extract_personal_candidates.py scripts/promote_personal_data.py
git commit -m "feat: extract and promote personal train datasets"
```

---

