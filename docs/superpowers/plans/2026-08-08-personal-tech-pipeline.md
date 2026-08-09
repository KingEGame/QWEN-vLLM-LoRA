# Personal Tech Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine Cursor transcripts + repo markdown into two reviewable datasets, train two QLoRA adapters (question-sharper and me-assistant), serve both on AWQ, and chain them with a logging pipeline client.

**Architecture:** Keep the existing FAQ LoRA loop untouched as the default. Add a parallel personal path under `data/personal/` and `output/lora_*`. Extraction writes **candidates** only; promotion into train JSONL is explicit after review. `train_lora.py` gains `--data` / `--output`. Serve registers both LoRA modules. `personal_pipeline.py` calls sharper then me-assistant and appends `output/personal_runs.jsonl`.

**Tech Stack:** Python 3.12, existing `scripts.lib.dataset_validation`, transformers/PEFT/TRL QLoRA train path, vLLM `--enable-lora` multi-module serve, OpenAI-compatible client.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md`
- Domain v1: tech / coding / ML setup only
- Two adapters (pipeline), not one multi-task model
- Train base: dense `TRAIN_MODEL=Qwen/Qwen3.6-27B`; never train on AWQ
- Serve base: existing AWQ `MODEL` from `config/model.env`
- LoRA rank/alpha stay 16/16
- Personal JSONL under `data/personal/` must be gitignored (private chat text)
- No auto-promote without review
- Stop vLLM before training (VRAM)
- Do not commit unless the user explicitly asks (or the chosen execution skill requires commits — then commit only plan-scoped files)

## File Structure

```
.gitignore                                      # MODIFY — ignore data/personal train+candidates
data/personal/candidates/.gitkeep               # CREATE
data/personal/README.md                         # CREATE — review/promote instructions
config/personal_sources.env                     # CREATE — transcript + markdown roots
scripts/lib/personal_extract.py                 # CREATE — pure extract helpers
scripts/extract_personal_candidates.py          # CREATE — CLI writer
scripts/promote_personal_data.py                # CREATE — candidates → train JSONL after review flag
scripts/train_lora.py                           # MODIFY — --data / --output
scripts/serve_with_lora.sh                      # MODIFY — multi adapter via LORA_MODULES
scripts/personal_pipeline.py                    # CREATE — sharper → me-assistant + log
tests/test_personal_extract.py                  # CREATE
tests/test_personal_pipeline.py                 # CREATE
README.md                                       # MODIFY — personal pipeline section
```

---

### Task 1: Privacy paths + personal data scaffold

**Files:**
- Modify: `.gitignore`
- Create: `data/personal/candidates/.gitkeep`
- Create: `data/personal/README.md`
- Create: `config/personal_sources.env`

**Interfaces:**
- Consumes: design privacy rule (local-only personal JSONL)
- Produces: ignored `data/personal/**/*.jsonl`; documented source config keys `AGENT_TRANSCRIPTS_DIR`, `MARKDOWN_GLOBS`

- [ ] **Step 1: Update `.gitignore`**

Append:

```gitignore
# Personal pipeline datasets (may contain private chat text)
data/personal/**/*.jsonl
!data/personal/candidates/.gitkeep
```

Keep existing `data/train.jsonl` / `data/generated/*` rules.

- [ ] **Step 2: Create scaffold files**

`data/personal/candidates/.gitkeep` — empty file.

`data/personal/README.md`:

```markdown
# Personal tech datasets

1. Run `python scripts/extract_personal_candidates.py`
2. Review/edit files under `candidates/`
3. Promote: `python scripts/promote_personal_data.py --reviewed`
4. Train each adapter with `scripts/train_lora.py --data ... --output ...`
```

`config/personal_sources.env`:

```env
# Absolute or ~ paths OK. Override with env vars of the same name.
AGENT_TRANSCRIPTS_DIR=~/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
MARKDOWN_GLOBS=docs/superpowers/specs/*.md,docs/superpowers/plans/*.md,README.md
```

- [ ] **Step 3: Verify ignore**

```bash
git check-ignore -v data/personal/candidates/foo.jsonl
```

Expected: a matching `.gitignore` rule line.

- [ ] **Step 4: Commit** (only if user/execution skill requests)

```bash
git add .gitignore data/personal/candidates/.gitkeep data/personal/README.md config/personal_sources.env
git commit -m "chore: scaffold personal pipeline data paths"
```

---

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

### Task 4: Train script `--data` / `--output`

**Files:**
- Modify: `scripts/train_lora.py`

**Interfaces:**
- Consumes: CLI `--data` (Path), `--output` (Path); env `TRAIN_DATA`, `TRAIN_OUTPUT` as overrides
- Produces: adapter written to `--output` (default remains `output/lora_adapter`)

- [ ] **Step 1: Add argparse at top of `main()`**

Replace fixed `TRAIN_DATA_PATH` / `OUTPUT_DIR` usage with:

```python
import argparse

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Train JSONL path (default data/train.jsonl)")
    parser.add_argument("--output", default=None, help="Adapter output dir (default output/lora_adapter)")
    args = parser.parse_args()

    train_data = Path(
        args.data
        or os.environ.get("TRAIN_DATA")
        or (REPO_ROOT / "data" / "train.jsonl")
    )
    output_dir = Path(
        args.output
        or os.environ.get("TRAIN_OUTPUT")
        or (REPO_ROOT / "output" / "lora_adapter")
    )
```

Then replace every `TRAIN_DATA_PATH` → `train_data` and `OUTPUT_DIR` → `output_dir` in `main()`.

Keep module-level constants for backward-compatible imports if tests reference them, or update tests if any break.

- [ ] **Step 2: Verify help + dry path check**

```bash
python scripts/train_lora.py --help
```

Expected: shows `--data` and `--output`.

```bash
python scripts/train_lora.py --data /no/such.jsonl --output /tmp/x; echo EXIT:$?
```

Expected: ERROR about missing file, non-zero exit (no GPU load).

- [ ] **Step 3: Commit** (if requested)

```bash
git add scripts/train_lora.py
git commit -m "feat: train_lora accepts --data and --output paths"
```

---

### Task 5: Multi-LoRA serve

**Files:**
- Modify: `scripts/serve_with_lora.sh`

**Interfaces:**
- Consumes: optional `LORA_MODULES` env (comma-separated `name=rel/or/abs/path`)
- Produces: `vllm serve ... --lora-modules name=path [name2=path2 ...]`
- Default when unset: keep today’s single `ADAPTER_NAME=output/lora_adapter` behavior

- [ ] **Step 1: Replace adapter resolution block** in `scripts/serve_with_lora.sh`

Replace the single `ADAPTER_PATH` / `--lora-modules` section with:

```bash
LORA_MODULE_ARGS=()
if [ -n "${LORA_MODULES:-}" ]; then
    IFS=',' read -r -a _mods <<< "$LORA_MODULES"
    for spec in "${_mods[@]}"; do
        spec_trimmed="$(echo "$spec" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        name="${spec_trimmed%%=*}"
        path="${spec_trimmed#*=}"
        if [[ "$path" != /* ]]; then
            path="$REPO_ROOT/$path"
        fi
        if [ ! -d "$path" ]; then
            echo "ERROR: LoRA path missing for $name: $path" >&2
            exit 1
        fi
        LORA_MODULE_ARGS+=("${name}=${path}")
    done
else
    ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
        exit 1
    fi
    LORA_MODULE_ARGS+=("${ADAPTER_NAME}=${ADAPTER_PATH}")
fi

echo "Starting vLLM server with LoRA modules: ${LORA_MODULE_ARGS[*]} port=$PORT"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "${LORA_MODULE_ARGS[@]}" \
    "${QUANT_FLAG[@]}" \
    "${REASONING_FLAG[@]}" \
    "${LM_ONLY_FLAG[@]}" \
    "${MAX_SEQS_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}"
```

- [ ] **Step 2: Syntax check**

```bash
bash -n scripts/serve_with_lora.sh
```

Expected: no output, exit 0.

- [ ] **Step 3: Document personal invoke** (comment at top of script or README in Task 7)

Personal serve example:

```bash
LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" \
  ./scripts/serve_with_lora.sh
```

- [ ] **Step 4: Commit** (if requested)

```bash
git add scripts/serve_with_lora.sh
git commit -m "feat: serve_with_lora supports multiple LORA_MODULES"
```

---

### Task 6: Pipeline client (TDD)

**Files:**
- Create: `scripts/personal_pipeline.py`
- Create: `scripts/lib/personal_pipeline.py` (pure functions for testability)
- Test: `tests/test_personal_pipeline.py`

**Interfaces:**
- Consumes: OpenAI-compatible client; model names `question-sharper`, `me-assistant`; port from config
- Produces:
  - `run_pipeline(client, raw: str, *, sharp_model: str, answer_model: str) -> dict` with keys `raw`, `sharpened`, `answer`
  - CLI prints both steps; appends JSON line to `output/personal_runs.jsonl` unless `--no-log`
  - Empty sharpened → exit 1, no assistant call

- [ ] **Step 1: Write failing tests**

```python
from types import SimpleNamespace

from scripts.lib.personal_pipeline import run_pipeline


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, *, model, messages):
        self.calls.append((model, messages[0]["content"]))
        if model == "question-sharper":
            text = "How do I free VRAM before QLoRA training?"
        else:
            text = "Stop the vLLM server, then run nvidia-smi."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_run_pipeline_chains_models():
    client = _FakeClient()
    result = run_pipeline(
        client,
        "okey so like train fails maybe vram?",
        sharp_model="question-sharper",
        answer_model="me-assistant",
    )
    assert result["sharpened"].startswith("How do I")
    assert "vLLM" in result["answer"]
    assert [c[0] for c in client.chat.completions.calls] == [
        "question-sharper",
        "me-assistant",
    ]


def test_run_pipeline_aborts_on_empty_sharpen():
    class EmptySharp(_FakeCompletions):
        def create(self, *, model, messages):
            self.calls.append((model, messages[0]["content"]))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  "))])

    client = _FakeClient()
    client.chat.completions = EmptySharp()
    try:
        run_pipeline(client, "x", sharp_model="question-sharper", answer_model="me-assistant")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    assert len(client.chat.completions.calls) == 1
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/test_personal_pipeline.py -v
```

- [ ] **Step 3: Implement library + CLI**

`scripts/lib/personal_pipeline.py`:

```python
"""Chain question-sharper → me-assistant over an OpenAI-compatible client."""
from __future__ import annotations

from typing import Any


def run_pipeline(
    client: Any,
    raw: str,
    *,
    sharp_model: str,
    answer_model: str,
) -> dict[str, str]:
    sharp = client.chat.completions.create(
        model=sharp_model,
        messages=[{"role": "user", "content": raw}],
    )
    sharpened = (sharp.choices[0].message.content or "").strip()
    if not sharpened:
        raise ValueError("question-sharper returned an empty question")

    ans = client.chat.completions.create(
        model=answer_model,
        messages=[{"role": "user", "content": sharpened}],
    )
    answer = (ans.choices[0].message.content or "").strip()
    return {"raw": raw, "sharpened": sharpened, "answer": answer}
```

`scripts/personal_pipeline.py`:

```python
#!/usr/bin/env python3
"""Run personal tech pipeline: sharpen question, then answer as me-assistant."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from scripts.lib.env_config import load_env_file
from scripts.lib.personal_pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "output" / "personal_runs.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Messy tech thought / question")
    parser.add_argument("--sharp-model", default="question-sharper")
    parser.add_argument("--answer-model", default="me-assistant")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    config = load_env_file(REPO_ROOT / "config" / "model.env")
    port = config.get("PORT", "8000")
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

    try:
        result = run_pipeline(
            client,
            args.prompt,
            sharp_model=args.sharp_model,
            answer_model=args.answer_model,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Raw: {result['raw']}")
    print(f"Sharpened: {result['sharpened']}")
    print(f"Answer: {result['answer']}")

    if not args.no_log:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Logged → {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit tests — expect PASS**

```bash
python -m pytest tests/test_personal_pipeline.py tests/test_personal_extract.py -v
```

- [ ] **Step 5: Commit** (if requested)

```bash
git add scripts/lib/personal_pipeline.py scripts/personal_pipeline.py tests/test_personal_pipeline.py
git commit -m "feat: personal pipeline client with run logging"
```

---

### Task 7: README + end-to-end operator path

**Files:**
- Modify: `README.md`
- Optionally link the new design spec under Troubleshooting / docs list

**Interfaces:**
- Consumes: all prior tasks’ CLIs
- Produces: documented personal loop operators can follow

- [ ] **Step 1: Add README section** after the FAQ LoRA section

Insert a section titled `## Personal tech pipeline (question-sharper → me-assistant)` that documents:

1. `python scripts/extract_personal_candidates.py`
2. Review under `data/personal/candidates/`, then `python scripts/promote_personal_data.py --reviewed`
3. Validate both `data/personal/*.jsonl` files
4. Stop vLLM; train with `--data` / `--output` for each adapter path
5. Serve with `LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant"`
6. `python scripts/personal_pipeline.py "..."`

Link `config/personal_sources.env` and
`docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md`.

Also add the design link to the docs bullet list.

- [ ] **Step 2: Run full unit suite**

```bash
python -m pytest -v
```

Expected: all existing + new tests PASS.

- [ ] **Step 3: Commit** (if requested)

```bash
git add README.md
git commit -m "docs: document personal tech LoRA pipeline"
```

---

### Task 8: GPU smoke (manual / operator)

**Files:** none new

**Interfaces:** uses adapters from Task 4–5 and pipeline from Task 6

- [ ] **Step 1: Ensure reviewed train files have meaningful volume**

If promote produced &lt;20 pairs each, pause and expand/edit candidates before claiming personalization quality. Smoke can still run on small data.

- [ ] **Step 2: Train both adapters** (stop server first)

```bash
nvidia-smi
python scripts/train_lora.py --data data/personal/question_sharp.jsonl --output output/lora_question_sharper
python scripts/train_lora.py --data data/personal/me_assistant.jsonl --output output/lora_me_assistant
```

Expected: each ends with adapter saved under the given output dir.

- [ ] **Step 3: Serve + pipeline**

```bash
LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" \
  ./scripts/serve_with_lora.sh
# other terminal:
python scripts/personal_pipeline.py "okey can we like make the question clearer for vllm lora?"
```

Expected: prints Raw / Sharpened / Answer; appends `output/personal_runs.jsonl`.

- [ ] **Step 4: Mark plan complete** — no code commit required unless operator wants run logs ignored (already under `output/` if gitignored).

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| Two-adapter pipeline packaging | 5, 6 |
| Tech domain focus | 7 (docs), extraction sources |
| Mine transcripts + markdown | 2, 3 |
| Candidates then human review | 1, 3 (`--reviewed`) |
| Train on dense 27B QLoRA | 4 + existing train |
| Dual adapter outputs | 4, 8 |
| Multi LoRA serve | 5 |
| Pipeline client + logging | 6 |
| Privacy gitignore | 1 |
| Success smoke | 8 |
| Out of scope (tools, auto-promote, multi-task) | not implemented |

## Placeholder / consistency self-review

- No TBD/TODO left in tasks
- Model names consistent: `question-sharper`, `me-assistant`
- Paths consistent: `output/lora_question_sharper`, `output/lora_me_assistant`
- Candidate → promote → train data paths aligned
- `run_pipeline` signature shared by tests and CLI
