# Review package
BASE: 23b0d3f037b93d0c41eb61ece1cc195360fe37c2
HEAD: 484b8b77971a998cfb822adb80aa7d1c19a09893

## Commits
484b8b7 fix: load QLoRA model fully on cuda:0 for 24GB train
578bcdf docs: document personal tech LoRA pipeline
026f8c2 feat: personal pipeline client with run logging
0d44c08 feat: serve_with_lora supports multiple LORA_MODULES
e0ac82f fix: scope Task 4 to CLI --data/--output only
1874a6d feat: train_lora accepts --data and --output paths
e516098 feat: extract and promote personal train datasets
a96be0e feat: extract personal LoRA candidates from transcripts and markdown
f7223cf chore: scaffold personal pipeline data paths
92185ab docs: add personal tech pipeline implementation plan
04f567a docs: add personal tech pipeline design
21aa7e5 docs: document Qwen3.6-27B LoRA generate-train-serve loop
2569c3f feat: harden train_lora.py for Qwen3.6-27B QLoRA
a91dc3b feat: align serve_with_lora.sh with AWQ serve flags
4ce16cb feat: add TRAIN_MODEL for dense Qwen3.6-27B QLoRA

## Stat
 .gitignore                                         |  20 +-
 README.md                                          |  69 ++
 config/model.env                                   |   1 +
 config/personal_sources.env                        |   5 +
 data/personal/README.md                            |   6 +
 data/personal/candidates/.gitkeep                  |   0
 .../plans/2026-08-08-personal-tech-pipeline.md     | 986 +++++++++++++++++++++
 .../plans/2026-08-08-qwen36-27b-lora-train.md      | 325 +++++++
 .../2026-08-08-personal-tech-pipeline-design.md    | 136 +++
 .../2026-08-08-qwen36-27b-lora-train-design.md     | 136 +++
 scripts/extract_personal_candidates.py             |  77 ++
 scripts/lib/personal_extract.py                    | 145 +++
 scripts/lib/personal_pipeline.py                   |  27 +
 scripts/personal_pipeline.py                       |  65 ++
 scripts/promote_personal_data.py                   |  63 ++
 scripts/serve_with_lora.sh                         |  55 +-
 scripts/train_lora.py                              |  71 +-
 tests/test_personal_extract.py                     |  55 ++
 tests/test_personal_pipeline.py                    |  53 ++
 19 files changed, 2265 insertions(+), 30 deletions(-)

## Diff
diff --git a/.gitignore b/.gitignore
index 754052b..e794bd9 100644
--- a/.gitignore
+++ b/.gitignore
@@ -1,8 +1,12 @@
-.venv/
-__pycache__/
-*.pyc
-.pytest_cache/
-output/
-data/generated/*
-!data/generated/.gitkeep
-data/train.jsonl
+.venv/
+__pycache__/
+*.pyc
+.pytest_cache/
+output/
+data/generated/*
+!data/generated/.gitkeep
+data/train.jsonl
+
+# Personal pipeline datasets (may contain private chat text)
+data/personal/**/*.jsonl
+!data/personal/candidates/.gitkeep
diff --git a/README.md b/README.md
index f0ed735..746090b 100644
--- a/README.md
+++ b/README.md
@@ -78,27 +78,96 @@ To roll back to the small bf16 model for LoRA experiments, set in `config/model.
 
 ```env
 MODEL=Qwen/Qwen3-4B-Instruct-2507
 MAX_MODEL_LEN=32768
 MAX_NUM_SEQS=
 QUANTIZATION=none
 REASONING_PARSER=
 LANGUAGE_MODEL_ONLY=0
 ```
 
+## LoRA on Qwen3.6-27B (example FAQ)
+
+End-to-end generate → train → serve-with-LoRA using `data/source_docs/example_faq.md`.
+Generation uses the running AWQ server; training loads dense `Qwen/Qwen3.6-27B` in 4-bit
+(QLoRA cannot train on the AWQ checkpoint).
+
+```bash
+# 1) server already running with AWQ 27B
+python scripts/generate_training_data.py
+
+# 2) first-run promote (light review optional)
+mkdir -p data
+cp data/generated/raw_qa.jsonl data/train.jsonl
+python scripts/validate_dataset.py data/train.jsonl
+
+# 3) stop the vLLM server (Ctrl+C in its terminal), confirm VRAM free:
+nvidia-smi
+
+# 4) train (downloads dense Qwen/Qwen3.6-27B on first run — large)
+# Knobs live in config/model.env (batch=2, epochs=3, accum=8, ~70% free GPU/CPU).
+python scripts/train_lora.py
+
+# 5) serve base + adapter
+./scripts/serve_with_lora.sh
+# other terminal:
+python scripts/test_client.py --model support-adapter
+```
+
+## Personal tech pipeline (question-sharper → me-assistant)
+
+Two LoRA adapters on the same AWQ base: sharpen messy tech thoughts into clear
+questions, then answer in your preferred style. Sources and paths are in
+[`config/personal_sources.env`](config/personal_sources.env); full design in
+[Personal tech pipeline design](docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md).
+
+**WSL note:** `~/.cursor/...` in config resolves to the WSL home, not Windows
+Cursor data. When extracting from WSL, point at the Windows agent-transcripts
+folder:
+
+```bash
+export AGENT_TRANSCRIPTS_DIR=/mnt/c/Users/supre/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
+```
+
+```bash
+# 1) extract candidates from transcripts + configured markdown
+python scripts/extract_personal_candidates.py
+# review/edit under data/personal/candidates/
+
+# 2) promote after human review (strips metadata, validates)
+python scripts/promote_personal_data.py --reviewed
+python scripts/validate_dataset.py data/personal/question_sharp.jsonl
+python scripts/validate_dataset.py data/personal/me_assistant.jsonl
+
+# 3) stop vLLM; confirm VRAM free (nvidia-smi), then train each adapter
+python scripts/train_lora.py --data data/personal/question_sharp.jsonl --output output/lora_question_sharper
+python scripts/train_lora.py --data data/personal/me_assistant.jsonl --output output/lora_me_assistant
+
+# 4) serve base + both adapters
+LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" ./scripts/serve_with_lora.sh
+
+# 5) run the chained pipeline (optional run log: output/personal_runs.jsonl)
+python scripts/personal_pipeline.py "how do i fix oom when starting vllm on 24gb"
+```
+
+Personal train/candidate JSONL under `data/personal/` may contain private chat
+text — keep it local unless you explicitly version sanitized data.
+
 ## Troubleshooting
 
 Driver, CUDA, and out-of-memory issues are documented in:
 
 - [Design: Qwen + vLLM + LoRA setup](docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md)
 - [Design: easy onboard setup scripts](docs/superpowers/specs/2026-08-06-easy-onboard-setup-scripts-design.md)
 - [Design: Qwen3.6-27B AWQ serve](docs/superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md)
+- [Design: Qwen3.6-27B LoRA train](docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md)
+- [Design: personal tech pipeline](docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md)
 
 Tune model/port/context in `config/model.env` — setup does not rewrite it.
 
 ## Unit tests (no GPU required)
 
 `pytest` is not installed by `setup.sh`; install it into the venv first:
 
 ```bash
 pip install pytest
 python -m pytest -v
diff --git a/config/model.env b/config/model.env
index ed2d081..0af0dde 100644
--- a/config/model.env
+++ b/config/model.env
@@ -1,9 +1,10 @@
 MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP
 PORT=8000
 MAX_MODEL_LEN=4096
 GPU_MEM_UTIL=0.92
 MAX_NUM_SEQS=32
 QUANTIZATION=awq
 ADAPTER_NAME=support-adapter
 REASONING_PARSER=qwen3
 LANGUAGE_MODEL_ONLY=1
+TRAIN_MODEL=Qwen/Qwen3.6-27B
diff --git a/config/personal_sources.env b/config/personal_sources.env
new file mode 100644
index 0000000..99e07c5
--- /dev/null
+++ b/config/personal_sources.env
@@ -0,0 +1,5 @@
+# Absolute or ~ paths OK. Override with env vars of the same name.
+# WSL: ~ resolves to WSL $HOME, not Windows Cursor. Use /mnt/c/... path instead, e.g.:
+# AGENT_TRANSCRIPTS_DIR=/mnt/c/Users/supre/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
+AGENT_TRANSCRIPTS_DIR=~/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
+MARKDOWN_GLOBS=docs/superpowers/specs/*.md,docs/superpowers/plans/*.md,README.md
diff --git a/data/personal/README.md b/data/personal/README.md
new file mode 100644
index 0000000..3a12971
--- /dev/null
+++ b/data/personal/README.md
@@ -0,0 +1,6 @@
+# Personal tech datasets
+
+1. Run `python scripts/extract_personal_candidates.py`
+2. Review/edit files under `candidates/`
+3. Promote: `python scripts/promote_personal_data.py --reviewed`
+4. Train each adapter with `scripts/train_lora.py --data ... --output ...`
diff --git a/data/personal/candidates/.gitkeep b/data/personal/candidates/.gitkeep
new file mode 100644
index 0000000..e69de29
diff --git a/docs/superpowers/plans/2026-08-08-personal-tech-pipeline.md b/docs/superpowers/plans/2026-08-08-personal-tech-pipeline.md
new file mode 100644
index 0000000..d3083bc
--- /dev/null
+++ b/docs/superpowers/plans/2026-08-08-personal-tech-pipeline.md
@@ -0,0 +1,986 @@
+# Personal Tech Pipeline Implementation Plan
+
+> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
+
+**Goal:** Mine Cursor transcripts + repo markdown into two reviewable datasets, train two QLoRA adapters (question-sharper and me-assistant), serve both on AWQ, and chain them with a logging pipeline client.
+
+**Architecture:** Keep the existing FAQ LoRA loop untouched as the default. Add a parallel personal path under `data/personal/` and `output/lora_*`. Extraction writes **candidates** only; promotion into train JSONL is explicit after review. `train_lora.py` gains `--data` / `--output`. Serve registers both LoRA modules. `personal_pipeline.py` calls sharper then me-assistant and appends `output/personal_runs.jsonl`.
+
+**Tech Stack:** Python 3.12, existing `scripts.lib.dataset_validation`, transformers/PEFT/TRL QLoRA train path, vLLM `--enable-lora` multi-module serve, OpenAI-compatible client.
+
+## Global Constraints
+
+- Spec: `docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md`
+- Domain v1: tech / coding / ML setup only
+- Two adapters (pipeline), not one multi-task model
+- Train base: dense `TRAIN_MODEL=Qwen/Qwen3.6-27B`; never train on AWQ
+- Serve base: existing AWQ `MODEL` from `config/model.env`
+- LoRA rank/alpha stay 16/16
+- Personal JSONL under `data/personal/` must be gitignored (private chat text)
+- No auto-promote without review
+- Stop vLLM before training (VRAM)
+- Do not commit unless the user explicitly asks (or the chosen execution skill requires commits — then commit only plan-scoped files)
+
+## File Structure
+
+```
+.gitignore                                      # MODIFY — ignore data/personal train+candidates
+data/personal/candidates/.gitkeep               # CREATE
+data/personal/README.md                         # CREATE — review/promote instructions
+config/personal_sources.env                     # CREATE — transcript + markdown roots
+scripts/lib/personal_extract.py                 # CREATE — pure extract helpers
+scripts/extract_personal_candidates.py          # CREATE — CLI writer
+scripts/promote_personal_data.py                # CREATE — candidates → train JSONL after review flag
+scripts/train_lora.py                           # MODIFY — --data / --output
+scripts/serve_with_lora.sh                      # MODIFY — multi adapter via LORA_MODULES
+scripts/personal_pipeline.py                    # CREATE — sharper → me-assistant + log
+tests/test_personal_extract.py                  # CREATE
+tests/test_personal_pipeline.py                 # CREATE
+README.md                                       # MODIFY — personal pipeline section
+```
+
+---
+
+### Task 1: Privacy paths + personal data scaffold
+
+**Files:**
+- Modify: `.gitignore`
+- Create: `data/personal/candidates/.gitkeep`
+- Create: `data/personal/README.md`
+- Create: `config/personal_sources.env`
+
+**Interfaces:**
+- Consumes: design privacy rule (local-only personal JSONL)
+- Produces: ignored `data/personal/**/*.jsonl`; documented source config keys `AGENT_TRANSCRIPTS_DIR`, `MARKDOWN_GLOBS`
+
+- [ ] **Step 1: Update `.gitignore`**
+
+Append:
+
+```gitignore
+# Personal pipeline datasets (may contain private chat text)
+data/personal/**/*.jsonl
+!data/personal/candidates/.gitkeep
+```
+
+Keep existing `data/train.jsonl` / `data/generated/*` rules.
+
+- [ ] **Step 2: Create scaffold files**
+
+`data/personal/candidates/.gitkeep` — empty file.
+
+`data/personal/README.md`:
+
+```markdown
+# Personal tech datasets
+
+1. Run `python scripts/extract_personal_candidates.py`
+2. Review/edit files under `candidates/`
+3. Promote: `python scripts/promote_personal_data.py --reviewed`
+4. Train each adapter with `scripts/train_lora.py --data ... --output ...`
+```
+
+`config/personal_sources.env`:
+
+```env
+# Absolute or ~ paths OK. Override with env vars of the same name.
+AGENT_TRANSCRIPTS_DIR=~/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
+MARKDOWN_GLOBS=docs/superpowers/specs/*.md,docs/superpowers/plans/*.md,README.md
+```
+
+- [ ] **Step 3: Verify ignore**
+
+```bash
+git check-ignore -v data/personal/candidates/foo.jsonl
+```
+
+Expected: a matching `.gitignore` rule line.
+
+- [ ] **Step 4: Commit** (only if user/execution skill requests)
+
+```bash
+git add .gitignore data/personal/candidates/.gitkeep data/personal/README.md config/personal_sources.env
+git commit -m "chore: scaffold personal pipeline data paths"
+```
+
+---
+
+### Task 2: Personal extract library (TDD)
+
+**Files:**
+- Create: `scripts/lib/personal_extract.py`
+- Test: `tests/test_personal_extract.py`
+
+**Interfaces:**
+- Consumes: Cursor transcript JSONL lines with `role` + `message.content[].text`; markdown files as text
+- Produces:
+  - `extract_user_query(text: str) -> str | None`
+  - `iter_transcript_user_texts(path: Path) -> list[str]`
+  - `iter_transcript_qa_pairs(path: Path) -> list[tuple[str, str]]`
+  - `draft_sharpen(messy: str) -> str`
+  - `pairs_from_markdown(text: str, source: str) -> list[dict]`
+  - Candidate dict shape: `{"instruction": str, "response": str, "source": str, "kind": "sharpen"|"me_assistant"}`
+
+- [ ] **Step 1: Write failing tests** in `tests/test_personal_extract.py`
+
+```python
+from pathlib import Path
+
+from scripts.lib.personal_extract import (
+    draft_sharpen,
+    extract_user_query,
+    iter_transcript_qa_pairs,
+    iter_transcript_user_texts,
+    pairs_from_markdown,
+)
+
+
+def test_extract_user_query_from_wrapper():
+    raw = "<user_query>\nokey can we fix the train OOM?\n</user_query>"
+    assert extract_user_query(raw) == "okey can we fix the train OOM?"
+
+
+def test_extract_user_query_plain_fallback():
+    assert extract_user_query("plain question about LoRA") == "plain question about LoRA"
+
+
+def test_draft_sharpen_collapses_whitespace():
+    messy = "okey   so like\n\ncan we train  two adapters??"
+    sharp = draft_sharpen(messy)
+    assert "  " not in sharp
+    assert "?" in sharp or sharp.endswith("adapters")
+
+
+def test_iter_transcript_user_texts(tmp_path: Path):
+    p = tmp_path / "t.jsonl"
+    p.write_text(
+        '{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\\nfix download\\n</user_query>"}]}}\n'
+        '{"role":"assistant","message":{"content":[{"type":"text","text":"checking..."}]}}\n',
+        encoding="utf-8",
+    )
+    texts = iter_transcript_user_texts(p)
+    assert texts == ["fix download"]
+
+
+def test_iter_transcript_qa_pairs(tmp_path: Path):
+    p = tmp_path / "t.jsonl"
+    p.write_text(
+        '{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\\nhow do I serve LoRA?\\n</user_query>"}]}}\n'
+        '{"role":"assistant","message":{"content":[{"type":"text","text":"Use serve_with_lora.sh"}]}}\n',
+        encoding="utf-8",
+    )
+    pairs = iter_transcript_qa_pairs(p)
+    assert pairs == [("how do I serve LoRA?", "Use serve_with_lora.sh")]
+
+
+def test_pairs_from_markdown_heading_chunks():
+    md = "# Serve\n\nUse AWQ + adapter.\n\n# Train\n\nStop server first.\n"
+    pairs = pairs_from_markdown(md, source="docs/x.md")
+    assert len(pairs) >= 1
+    assert all(p["kind"] == "me_assistant" for p in pairs)
+    assert all(p["instruction"] and p["response"] for p in pairs)
+```
+
+- [ ] **Step 2: Run tests — expect FAIL**
+
+```bash
+python -m pytest tests/test_personal_extract.py -v
+```
+
+Expected: import / missing module failures.
+
+- [ ] **Step 3: Implement `scripts/lib/personal_extract.py`**
+
+```python
+"""Extract personal LoRA candidate pairs from Cursor transcripts and markdown."""
+from __future__ import annotations
+
+import json
+import re
+from pathlib import Path
+
+_USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL | re.IGNORECASE)
+
+
+def extract_user_query(text: str) -> str | None:
+    text = (text or "").strip()
+    if not text:
+        return None
+    m = _USER_QUERY_RE.search(text)
+    if m:
+        q = m.group(1).strip()
+        return q or None
+    # Skip obvious system/tool dumps
+    if text.startswith("{" ) and '"role"' in text:
+        return None
+    return text
+
+
+def draft_sharpen(messy: str) -> str:
+    """Heuristic draft only — human must review before train promote."""
+    cleaned = re.sub(r"\s+", " ", (messy or "").strip())
+    if not cleaned:
+        return ""
+    if cleaned[-1] not in ".?!":
+        cleaned += "?"
+    # Prefer a single question-shaped line
+    if len(cleaned) > 240:
+        cleaned = cleaned[:237].rstrip() + "..."
+    return cleaned[0].upper() + cleaned[1:] if cleaned else ""
+
+
+def _message_text(obj: dict) -> str:
+    msg = obj.get("message") or {}
+    content = msg.get("content")
+    if isinstance(content, str):
+        return content
+    if isinstance(content, list):
+        parts: list[str] = []
+        for block in content:
+            if isinstance(block, dict) and block.get("type") == "text":
+                parts.append(str(block.get("text") or ""))
+            elif isinstance(block, str):
+                parts.append(block)
+        return "\n".join(parts)
+    return ""
+
+
+def iter_transcript_user_texts(path: Path) -> list[str]:
+    out: list[str] = []
+    for line in Path(path).read_text(encoding="utf-8").splitlines():
+        if not line.strip():
+            continue
+        try:
+            obj = json.loads(line)
+        except json.JSONDecodeError:
+            continue
+        if obj.get("role") != "user":
+            continue
+        q = extract_user_query(_message_text(obj))
+        if q:
+            out.append(q)
+    return out
+
+
+def iter_transcript_qa_pairs(path: Path) -> list[tuple[str, str]]:
+    pairs: list[tuple[str, str]] = []
+    pending: str | None = None
+    for line in Path(path).read_text(encoding="utf-8").splitlines():
+        if not line.strip():
+            continue
+        try:
+            obj = json.loads(line)
+        except json.JSONDecodeError:
+            continue
+        role = obj.get("role")
+        text = _message_text(obj)
+        if role == "user":
+            pending = extract_user_query(text)
+        elif role == "assistant" and pending:
+            # First text paragraph only — drop huge tool dumps
+            reply = text.strip().split("\n\n")[0].strip()
+            if reply and len(reply) < 4000:
+                pairs.append((pending, reply))
+            pending = None
+    return pairs
+
+
+def pairs_from_markdown(text: str, source: str) -> list[dict]:
+    """Turn markdown H1/H2 sections into me_assistant candidates."""
+    sections = re.split(r"(?m)^#{1,2}\s+", text)
+    out: list[dict] = []
+    # sections[0] is preface before first heading
+    parts = re.findall(r"(?m)^(#{1,2}\s+.+)$(.*?)(?=^#{1,2}\s+|\Z)", text, flags=re.DOTALL)
+    if not parts:
+        body = text.strip()
+        if body:
+            title = Path(source).stem.replace("-", " ")
+            out.append(
+                {
+                    "instruction": f"What should I know about {title}?",
+                    "response": body[:2000],
+                    "source": source,
+                    "kind": "me_assistant",
+                }
+            )
+        return out
+    for heading_line, body in parts:
+        title = re.sub(r"^#{1,2}\s+", "", heading_line).strip()
+        body = body.strip()
+        if not title or not body:
+            continue
+        out.append(
+            {
+                "instruction": f"Explain: {title}",
+                "response": body[:2000],
+                "source": source,
+                "kind": "me_assistant",
+            }
+        )
+    return out
+
+
+def sharpen_candidates_from_texts(texts: list[str], source: str) -> list[dict]:
+    out: list[dict] = []
+    for t in texts:
+        sharp = draft_sharpen(t)
+        if not sharp or sharp == t:
+            # still keep if messy enough (length or newlines originally)
+            if len(t) < 20:
+                continue
+        out.append(
+            {
+                "instruction": t,
+                "response": sharp or draft_sharpen(t),
+                "source": source,
+                "kind": "sharpen",
+            }
+        )
+    return out
+```
+
+- [ ] **Step 4: Run tests — expect PASS**
+
+```bash
+python -m pytest tests/test_personal_extract.py -v
+```
+
+Expected: all PASSED. If `test_draft_sharpen_*` is brittle, adjust assertion to match `draft_sharpen` behavior without weakening the function.
+
+- [ ] **Step 5: Commit** (if requested)
+
+```bash
+git add scripts/lib/personal_extract.py tests/test_personal_extract.py
+git commit -m "feat: extract personal LoRA candidates from transcripts and markdown"
+```
+
+---
+
+### Task 3: Extract + promote CLIs
+
+**Files:**
+- Create: `scripts/extract_personal_candidates.py`
+- Create: `scripts/promote_personal_data.py`
+- Modify: none required in lib beyond Task 2
+
+**Interfaces:**
+- Consumes: `config/personal_sources.env`, `load_env_file`, extract helpers
+- Produces:
+  - `data/personal/candidates/question_sharp.jsonl`
+  - `data/personal/candidates/me_assistant.jsonl`
+  - On promote: `data/personal/question_sharp.jsonl`, `data/personal/me_assistant.jsonl` (instruction/response only)
+- Promote requires `--reviewed` flag (no silent promote)
+
+- [ ] **Step 1: Implement `scripts/extract_personal_candidates.py`**
+
+```python
+#!/usr/bin/env python3
+"""Mine transcripts + markdown into personal candidate JSONL files."""
+from __future__ import annotations
+
+import json
+import os
+import sys
+from pathlib import Path
+
+sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
+
+from scripts.lib.env_config import load_env_file
+from scripts.lib.personal_extract import (
+    iter_transcript_qa_pairs,
+    iter_transcript_user_texts,
+    pairs_from_markdown,
+    sharpen_candidates_from_texts,
+)
+
+REPO_ROOT = Path(__file__).resolve().parent.parent
+OUT_DIR = REPO_ROOT / "data" / "personal" / "candidates"
+
+
+def _expand(p: str) -> Path:
+    return Path(os.path.expanduser(p)).expanduser()
+
+
+def main() -> int:
+    cfg = load_env_file(REPO_ROOT / "config" / "personal_sources.env")
+    transcripts = _expand(
+        os.environ.get("AGENT_TRANSCRIPTS_DIR") or cfg.get("AGENT_TRANSCRIPTS_DIR", "")
+    )
+    globs = (
+        os.environ.get("MARKDOWN_GLOBS") or cfg.get("MARKDOWN_GLOBS", "README.md")
+    ).split(",")
+
+    sharpen: list[dict] = []
+    me: list[dict] = []
+
+    if transcripts.is_dir():
+        for path in sorted(transcripts.rglob("*.jsonl")):
+            src = str(path)
+            sharpen.extend(sharpen_candidates_from_texts(iter_transcript_user_texts(path), src))
+            for q, a in iter_transcript_qa_pairs(path):
+                me.append(
+                    {
+                        "instruction": q,
+                        "response": a,
+                        "source": src,
+                        "kind": "me_assistant",
+                    }
+                )
+    else:
+        print(f"WARNING: transcripts dir missing: {transcripts}", file=sys.stderr)
+
+    for pattern in globs:
+        pattern = pattern.strip()
+        if not pattern:
+            continue
+        for path in sorted(REPO_ROOT.glob(pattern)):
+            if not path.is_file():
+                continue
+            me.extend(pairs_from_markdown(path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))))
+
+    OUT_DIR.mkdir(parents=True, exist_ok=True)
+    sharp_path = OUT_DIR / "question_sharp.jsonl"
+    me_path = OUT_DIR / "me_assistant.jsonl"
+    sharp_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in sharpen) + ("\n" if sharpen else ""), encoding="utf-8")
+    me_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in me) + ("\n" if me else ""), encoding="utf-8")
+    print(f"Wrote {len(sharpen)} sharpen candidates → {sharp_path}")
+    print(f"Wrote {len(me)} me_assistant candidates → {me_path}")
+    print("Review/edit candidates, then: python scripts/promote_personal_data.py --reviewed")
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
+```
+
+- [ ] **Step 2: Implement `scripts/promote_personal_data.py`**
+
+```python
+#!/usr/bin/env python3
+"""Promote reviewed personal candidates to train JSONL (instruction/response only)."""
+from __future__ import annotations
+
+import argparse
+import json
+import sys
+from pathlib import Path
+
+sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
+
+from scripts.lib.dataset_validation import validate_dataset_file
+
+REPO_ROOT = Path(__file__).resolve().parent.parent
+CAND = REPO_ROOT / "data" / "personal" / "candidates"
+OUT = REPO_ROOT / "data" / "personal"
+
+
+def _strip_meta(path: Path, dest: Path) -> int:
+    rows: list[dict] = []
+    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
+        if not line.strip():
+            continue
+        obj = json.loads(line)
+        row = {"instruction": obj["instruction"], "response": obj["response"]}
+        rows.append(row)
+    dest.parent.mkdir(parents=True, exist_ok=True)
+    dest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
+    return len(rows)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument(
+        "--reviewed",
+        action="store_true",
+        help="Required confirmation that candidates were human-reviewed",
+    )
+    args = parser.parse_args()
+    if not args.reviewed:
+        print("ERROR: refusing to promote without --reviewed", file=sys.stderr)
+        return 1
+
+    mapping = [
+        (CAND / "question_sharp.jsonl", OUT / "question_sharp.jsonl"),
+        (CAND / "me_assistant.jsonl", OUT / "me_assistant.jsonl"),
+    ]
+    for src, dest in mapping:
+        if not src.exists():
+            print(f"ERROR: missing {src}", file=sys.stderr)
+            return 1
+        n = _strip_meta(src, dest)
+        errors = validate_dataset_file(dest)
+        if errors:
+            print(f"ERROR: {dest} invalid:", file=sys.stderr)
+            print("\n".join(errors[:20]), file=sys.stderr)
+            return 1
+        print(f"Promoted {n} rows → {dest}")
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
+```
+
+- [ ] **Step 3: Smoke extract (no GPU)**
+
+```bash
+python scripts/extract_personal_candidates.py
+python scripts/promote_personal_data.py
+# expect ERROR without --reviewed
+python scripts/promote_personal_data.py --reviewed
+python scripts/validate_dataset.py data/personal/question_sharp.jsonl
+python scripts/validate_dataset.py data/personal/me_assistant.jsonl
+```
+
+Expected: candidate counts printed; promote without flag fails; with flag validates clean (after any empty-file edge: ensure extract produced ≥1 row or skip validate with a clear message — if zero rows, print ERROR and exit 1 in promote).
+
+- [ ] **Step 4: Commit** (if requested)
+
+```bash
+git add scripts/extract_personal_candidates.py scripts/promote_personal_data.py
+git commit -m "feat: extract and promote personal train datasets"
+```
+
+---
+
+### Task 4: Train script `--data` / `--output`
+
+**Files:**
+- Modify: `scripts/train_lora.py`
+
+**Interfaces:**
+- Consumes: CLI `--data` (Path), `--output` (Path); env `TRAIN_DATA`, `TRAIN_OUTPUT` as overrides
+- Produces: adapter written to `--output` (default remains `output/lora_adapter`)
+
+- [ ] **Step 1: Add argparse at top of `main()`**
+
+Replace fixed `TRAIN_DATA_PATH` / `OUTPUT_DIR` usage with:
+
+```python
+import argparse
+
+def main() -> int:
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--data", default=None, help="Train JSONL path (default data/train.jsonl)")
+    parser.add_argument("--output", default=None, help="Adapter output dir (default output/lora_adapter)")
+    args = parser.parse_args()
+
+    train_data = Path(
+        args.data
+        or os.environ.get("TRAIN_DATA")
+        or (REPO_ROOT / "data" / "train.jsonl")
+    )
+    output_dir = Path(
+        args.output
+        or os.environ.get("TRAIN_OUTPUT")
+        or (REPO_ROOT / "output" / "lora_adapter")
+    )
+```
+
+Then replace every `TRAIN_DATA_PATH` → `train_data` and `OUTPUT_DIR` → `output_dir` in `main()`.
+
+Keep module-level constants for backward-compatible imports if tests reference them, or update tests if any break.
+
+- [ ] **Step 2: Verify help + dry path check**
+
+```bash
+python scripts/train_lora.py --help
+```
+
+Expected: shows `--data` and `--output`.
+
+```bash
+python scripts/train_lora.py --data /no/such.jsonl --output /tmp/x; echo EXIT:$?
+```
+
+Expected: ERROR about missing file, non-zero exit (no GPU load).
+
+- [ ] **Step 3: Commit** (if requested)
+
+```bash
+git add scripts/train_lora.py
+git commit -m "feat: train_lora accepts --data and --output paths"
+```
+
+---
+
+### Task 5: Multi-LoRA serve
+
+**Files:**
+- Modify: `scripts/serve_with_lora.sh`
+
+**Interfaces:**
+- Consumes: optional `LORA_MODULES` env (comma-separated `name=rel/or/abs/path`)
+- Produces: `vllm serve ... --lora-modules name=path [name2=path2 ...]`
+- Default when unset: keep today’s single `ADAPTER_NAME=output/lora_adapter` behavior
+
+- [ ] **Step 1: Replace adapter resolution block** in `scripts/serve_with_lora.sh`
+
+Replace the single `ADAPTER_PATH` / `--lora-modules` section with:
+
+```bash
+LORA_MODULE_ARGS=()
+if [ -n "${LORA_MODULES:-}" ]; then
+    IFS=',' read -r -a _mods <<< "$LORA_MODULES"
+    for spec in "${_mods[@]}"; do
+        spec_trimmed="$(echo "$spec" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
+        name="${spec_trimmed%%=*}"
+        path="${spec_trimmed#*=}"
+        if [[ "$path" != /* ]]; then
+            path="$REPO_ROOT/$path"
+        fi
+        if [ ! -d "$path" ]; then
+            echo "ERROR: LoRA path missing for $name: $path" >&2
+            exit 1
+        fi
+        LORA_MODULE_ARGS+=("${name}=${path}")
+    done
+else
+    ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
+    if [ ! -d "$ADAPTER_PATH" ]; then
+        echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
+        exit 1
+    fi
+    LORA_MODULE_ARGS+=("${ADAPTER_NAME}=${ADAPTER_PATH}")
+fi
+
+echo "Starting vLLM server with LoRA modules: ${LORA_MODULE_ARGS[*]} port=$PORT"
+
+vllm serve "$MODEL" \
+    --port "$PORT" \
+    --max-model-len "$MAX_MODEL_LEN" \
+    --gpu-memory-utilization "$GPU_MEM_UTIL" \
+    --enable-lora \
+    --max-lora-rank 16 \
+    --lora-modules "${LORA_MODULE_ARGS[@]}" \
+    "${QUANT_FLAG[@]}" \
+    "${REASONING_FLAG[@]}" \
+    "${LM_ONLY_FLAG[@]}" \
+    "${MAX_SEQS_FLAG[@]}" \
+    "${EXTRA_FLAGS[@]}"
+```
+
+- [ ] **Step 2: Syntax check**
+
+```bash
+bash -n scripts/serve_with_lora.sh
+```
+
+Expected: no output, exit 0.
+
+- [ ] **Step 3: Document personal invoke** (comment at top of script or README in Task 7)
+
+Personal serve example:
+
+```bash
+LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" \
+  ./scripts/serve_with_lora.sh
+```
+
+- [ ] **Step 4: Commit** (if requested)
+
+```bash
+git add scripts/serve_with_lora.sh
+git commit -m "feat: serve_with_lora supports multiple LORA_MODULES"
+```
+
+---
+
+### Task 6: Pipeline client (TDD)
+
+**Files:**
+- Create: `scripts/personal_pipeline.py`
+- Create: `scripts/lib/personal_pipeline.py` (pure functions for testability)
+- Test: `tests/test_personal_pipeline.py`
+
+**Interfaces:**
+- Consumes: OpenAI-compatible client; model names `question-sharper`, `me-assistant`; port from config
+- Produces:
+  - `run_pipeline(client, raw: str, *, sharp_model: str, answer_model: str) -> dict` with keys `raw`, `sharpened`, `answer`
+  - CLI prints both steps; appends JSON line to `output/personal_runs.jsonl` unless `--no-log`
+  - Empty sharpened → exit 1, no assistant call
+
+- [ ] **Step 1: Write failing tests**
+
+```python
+from types import SimpleNamespace
+
+from scripts.lib.personal_pipeline import run_pipeline
+
+
+class _FakeCompletions:
+    def __init__(self):
+        self.calls = []
+
+    def create(self, *, model, messages):
+        self.calls.append((model, messages[0]["content"]))
+        if model == "question-sharper":
+            text = "How do I free VRAM before QLoRA training?"
+        else:
+            text = "Stop the vLLM server, then run nvidia-smi."
+        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
+
+
+class _FakeClient:
+    def __init__(self):
+        self.chat = SimpleNamespace(completions=_FakeCompletions())
+
+
+def test_run_pipeline_chains_models():
+    client = _FakeClient()
+    result = run_pipeline(
+        client,
+        "okey so like train fails maybe vram?",
+        sharp_model="question-sharper",
+        answer_model="me-assistant",
+    )
+    assert result["sharpened"].startswith("How do I")
+    assert "vLLM" in result["answer"]
+    assert [c[0] for c in client.chat.completions.calls] == [
+        "question-sharper",
+        "me-assistant",
+    ]
+
+
+def test_run_pipeline_aborts_on_empty_sharpen():
+    class EmptySharp(_FakeCompletions):
+        def create(self, *, model, messages):
+            self.calls.append((model, messages[0]["content"]))
+            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  "))])
+
+    client = _FakeClient()
+    client.chat.completions = EmptySharp()
+    try:
+        run_pipeline(client, "x", sharp_model="question-sharper", answer_model="me-assistant")
+        assert False, "expected ValueError"
+    except ValueError as exc:
+        assert "empty" in str(exc).lower()
+    assert len(client.chat.completions.calls) == 1
+```
+
+- [ ] **Step 2: Run — expect FAIL**
+
+```bash
+python -m pytest tests/test_personal_pipeline.py -v
+```
+
+- [ ] **Step 3: Implement library + CLI**
+
+`scripts/lib/personal_pipeline.py`:
+
+```python
+"""Chain question-sharper → me-assistant over an OpenAI-compatible client."""
+from __future__ import annotations
+
+from typing import Any
+
+
+def run_pipeline(
+    client: Any,
+    raw: str,
+    *,
+    sharp_model: str,
+    answer_model: str,
+) -> dict[str, str]:
+    sharp = client.chat.completions.create(
+        model=sharp_model,
+        messages=[{"role": "user", "content": raw}],
+    )
+    sharpened = (sharp.choices[0].message.content or "").strip()
+    if not sharpened:
+        raise ValueError("question-sharper returned an empty question")
+
+    ans = client.chat.completions.create(
+        model=answer_model,
+        messages=[{"role": "user", "content": sharpened}],
+    )
+    answer = (ans.choices[0].message.content or "").strip()
+    return {"raw": raw, "sharpened": sharpened, "answer": answer}
+```
+
+`scripts/personal_pipeline.py`:
+
+```python
+#!/usr/bin/env python3
+"""Run personal tech pipeline: sharpen question, then answer as me-assistant."""
+from __future__ import annotations
+
+import argparse
+import json
+import sys
+from datetime import datetime, timezone
+from pathlib import Path
+
+sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
+
+from openai import OpenAI
+
+from scripts.lib.env_config import load_env_file
+from scripts.lib.personal_pipeline import run_pipeline
+
+REPO_ROOT = Path(__file__).resolve().parent.parent
+LOG_PATH = REPO_ROOT / "output" / "personal_runs.jsonl"
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("prompt", help="Messy tech thought / question")
+    parser.add_argument("--sharp-model", default="question-sharper")
+    parser.add_argument("--answer-model", default="me-assistant")
+    parser.add_argument("--no-log", action="store_true")
+    args = parser.parse_args()
+
+    config = load_env_file(REPO_ROOT / "config" / "model.env")
+    port = config.get("PORT", "8000")
+    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")
+
+    try:
+        result = run_pipeline(
+            client,
+            args.prompt,
+            sharp_model=args.sharp_model,
+            answer_model=args.answer_model,
+        )
+    except ValueError as exc:
+        print(f"ERROR: {exc}", file=sys.stderr)
+        return 1
+    except Exception as exc:
+        print(f"FAILED: {exc}", file=sys.stderr)
+        return 1
+
+    print(f"Raw: {result['raw']}")
+    print(f"Sharpened: {result['sharpened']}")
+    print(f"Answer: {result['answer']}")
+
+    if not args.no_log:
+        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
+        row = {
+            "ts": datetime.now(timezone.utc).isoformat(),
+            **result,
+        }
+        with LOG_PATH.open("a", encoding="utf-8") as fh:
+            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
+        print(f"Logged → {LOG_PATH}")
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
+```
+
+- [ ] **Step 4: Run unit tests — expect PASS**
+
+```bash
+python -m pytest tests/test_personal_pipeline.py tests/test_personal_extract.py -v
+```
+
+- [ ] **Step 5: Commit** (if requested)
+
+```bash
+git add scripts/lib/personal_pipeline.py scripts/personal_pipeline.py tests/test_personal_pipeline.py
+git commit -m "feat: personal pipeline client with run logging"
+```
+
+---
+
+### Task 7: README + end-to-end operator path
+
+**Files:**
+- Modify: `README.md`
+- Optionally link the new design spec under Troubleshooting / docs list
+
+**Interfaces:**
+- Consumes: all prior tasks’ CLIs
+- Produces: documented personal loop operators can follow
+
+- [ ] **Step 1: Add README section** after the FAQ LoRA section
+
+Insert a section titled `## Personal tech pipeline (question-sharper → me-assistant)` that documents:
+
+1. `python scripts/extract_personal_candidates.py`
+2. Review under `data/personal/candidates/`, then `python scripts/promote_personal_data.py --reviewed`
+3. Validate both `data/personal/*.jsonl` files
+4. Stop vLLM; train with `--data` / `--output` for each adapter path
+5. Serve with `LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant"`
+6. `python scripts/personal_pipeline.py "..."`
+
+Link `config/personal_sources.env` and
+`docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md`.
+
+Also add the design link to the docs bullet list.
+
+- [ ] **Step 2: Run full unit suite**
+
+```bash
+python -m pytest -v
+```
+
+Expected: all existing + new tests PASS.
+
+- [ ] **Step 3: Commit** (if requested)
+
+```bash
+git add README.md
+git commit -m "docs: document personal tech LoRA pipeline"
+```
+
+---
+
+### Task 8: GPU smoke (manual / operator)
+
+**Files:** none new
+
+**Interfaces:** uses adapters from Task 4–5 and pipeline from Task 6
+
+- [ ] **Step 1: Ensure reviewed train files have meaningful volume**
+
+If promote produced &lt;20 pairs each, pause and expand/edit candidates before claiming personalization quality. Smoke can still run on small data.
+
+- [ ] **Step 2: Train both adapters** (stop server first)
+
+```bash
+nvidia-smi
+python scripts/train_lora.py --data data/personal/question_sharp.jsonl --output output/lora_question_sharper
+python scripts/train_lora.py --data data/personal/me_assistant.jsonl --output output/lora_me_assistant
+```
+
+Expected: each ends with adapter saved under the given output dir.
+
+- [ ] **Step 3: Serve + pipeline**
+
+```bash
+LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" \
+  ./scripts/serve_with_lora.sh
+# other terminal:
+python scripts/personal_pipeline.py "okey can we like make the question clearer for vllm lora?"
+```
+
+Expected: prints Raw / Sharpened / Answer; appends `output/personal_runs.jsonl`.
+
+- [ ] **Step 4: Mark plan complete** — no code commit required unless operator wants run logs ignored (already under `output/` if gitignored).
+
+---
+
+## Spec coverage checklist
+
+| Spec requirement | Task |
+|---|---|
+| Two-adapter pipeline packaging | 5, 6 |
+| Tech domain focus | 7 (docs), extraction sources |
+| Mine transcripts + markdown | 2, 3 |
+| Candidates then human review | 1, 3 (`--reviewed`) |
+| Train on dense 27B QLoRA | 4 + existing train |
+| Dual adapter outputs | 4, 8 |
+| Multi LoRA serve | 5 |
+| Pipeline client + logging | 6 |
+| Privacy gitignore | 1 |
+| Success smoke | 8 |
+| Out of scope (tools, auto-promote, multi-task) | not implemented |
+
+## Placeholder / consistency self-review
+
+- No TBD/TODO left in tasks
+- Model names consistent: `question-sharper`, `me-assistant`
+- Paths consistent: `output/lora_question_sharper`, `output/lora_me_assistant`
+- Candidate → promote → train data paths aligned
+- `run_pipeline` signature shared by tests and CLI
diff --git a/docs/superpowers/plans/2026-08-08-qwen36-27b-lora-train.md b/docs/superpowers/plans/2026-08-08-qwen36-27b-lora-train.md
new file mode 100644
index 0000000..e010238
--- /dev/null
+++ b/docs/superpowers/plans/2026-08-08-qwen36-27b-lora-train.md
@@ -0,0 +1,325 @@
+# Qwen3.6-27B LoRA Train Implementation Plan
+
+> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
+
+**Goal:** Wire config + scripts so operators can generate FAQ Q&A on the AWQ 27B server, QLoRA-train an adapter on dense `Qwen/Qwen3.6-27B`, and serve AWQ + adapter.
+
+**Architecture:** Keep serving on AWQ (`MODEL`) and training on dense (`TRAIN_MODEL`). Reuse existing `generate_training_data.py` / `validate_dataset.py` / `train_lora.py`. Align `serve_with_lora.sh` with the flag pattern already used by `start_server.sh`. First-run data promotion is a documented `cp` (no new promote script unless validation forces it).
+
+**Tech Stack:** vLLM serve, transformers + PEFT + bitsandbytes + TRL QLoRA, existing dataset helpers.
+
+## Global Constraints
+
+- Train base: `TRAIN_MODEL=Qwen/Qwen3.6-27B` (dense); never train on the AWQ checkpoint.
+- Serve base stays `MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP`.
+- First-run data: auto-copy `data/generated/raw_qa.jsonl` → `data/train.jsonl`.
+- Smoke train knobs: `MAX_SEQ_LENGTH=1024`, `BATCH_SIZE=1`, `NUM_EPOCHS=1`.
+- LoRA rank/alpha remain 16/16; adapter path `output/lora_adapter/`.
+- Do not Unsloth-rewrite training in this plan.
+- Stop the vLLM server before training (VRAM).
+- Do not commit unless the user explicitly asks (or the chosen execution skill requires commits on a feature branch — then commit only plan-scoped files).
+
+## File Structure
+
+```
+config/model.env           # MODIFY — add TRAIN_MODEL
+scripts/train_lora.py      # MODIFY — safer 27B defaults / target-module fallback
+scripts/serve_with_lora.sh # MODIFY — wsl_runtime_env + reasoning/lm-only/max-num-seqs flags
+README.md                  # MODIFY — generate → train → serve-with-lora
+```
+
+No new Python modules required for v1.
+
+---
+
+### Task 1: Add `TRAIN_MODEL` to config
+
+**Files:**
+- Modify: `config/model.env`
+- Test: grep keys
+
+**Interfaces:**
+- Consumes: design decision `TRAIN_MODEL=Qwen/Qwen3.6-27B`
+- Produces: `TRAIN_MODEL` for `train_lora.py` (`os.environ` override still wins)
+
+- [ ] **Step 1: Append / set in `config/model.env`**
+
+Ensure these keys exist (keep existing serve keys):
+
+```env
+TRAIN_MODEL=Qwen/Qwen3.6-27B
+```
+
+Full expected file after edit (serve knobs unchanged from current master):
+
+```env
+MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP
+PORT=8000
+MAX_MODEL_LEN=4096
+GPU_MEM_UTIL=0.92
+MAX_NUM_SEQS=32
+QUANTIZATION=awq
+ADAPTER_NAME=support-adapter
+REASONING_PARSER=qwen3
+LANGUAGE_MODEL_ONLY=1
+TRAIN_MODEL=Qwen/Qwen3.6-27B
+```
+
+- [ ] **Step 2: Verify**
+
+```bash
+grep -E '^(MODEL|TRAIN_MODEL|QUANTIZATION)=' config/model.env
+```
+
+Expected:
+
+```
+MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP
+QUANTIZATION=awq
+TRAIN_MODEL=Qwen/Qwen3.6-27B
+```
+
+---
+
+### Task 2: Align `serve_with_lora.sh` with serve flags + WSL env
+
+**Files:**
+- Modify: `scripts/serve_with_lora.sh`
+- Test: `bash -n scripts/serve_with_lora.sh` + dry-run argv assembly
+
+**Interfaces:**
+- Consumes: same optional keys as `start_server.sh` (`REASONING_PARSER`, `LANGUAGE_MODEL_ONLY`, `MAX_NUM_SEQS`, `QUANTIZATION`, `EXTRA_ARGS`) plus LoRA adapter dir
+- Produces: `vllm serve` with `--enable-lora` and matching optional flags; sources `scripts/wsl_runtime_env.sh` when present
+
+- [ ] **Step 1: Replace `scripts/serve_with_lora.sh` with**
+
+```bash
+#!/usr/bin/env bash
+set -euo pipefail
+
+REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
+# shellcheck disable=SC1091
+if [ -f "$REPO_ROOT/scripts/wsl_runtime_env.sh" ]; then
+    source "$REPO_ROOT/scripts/wsl_runtime_env.sh"
+fi
+# shellcheck disable=SC1091
+source "$REPO_ROOT/config/model.env"
+
+QUANT_FLAG=()
+if [ "${QUANTIZATION:-none}" != "none" ]; then
+    QUANT_FLAG=(--quantization "$QUANTIZATION")
+fi
+
+REASONING_FLAG=()
+if [ -n "${REASONING_PARSER:-}" ]; then
+    REASONING_FLAG=(--reasoning-parser "$REASONING_PARSER")
+fi
+
+LM_ONLY_FLAG=()
+if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ] || [ "${LANGUAGE_MODEL_ONLY:-}" = "true" ]; then
+    LM_ONLY_FLAG=(--language-model-only)
+fi
+
+MAX_SEQS_FLAG=()
+if [ -n "${MAX_NUM_SEQS:-}" ]; then
+    MAX_SEQS_FLAG=(--max-num-seqs "$MAX_NUM_SEQS")
+fi
+
+EXTRA_FLAGS=()
+if [ -n "${EXTRA_ARGS:-}" ]; then
+    # shellcheck disable=SC2206
+    EXTRA_FLAGS=($EXTRA_ARGS)
+fi
+
+ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
+if [ ! -d "$ADAPTER_PATH" ]; then
+    echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
+    exit 1
+fi
+
+echo "Starting vLLM server with LoRA: model=$MODEL adapter=$ADAPTER_NAME port=$PORT"
+
+vllm serve "$MODEL" \
+    --port "$PORT" \
+    --max-model-len "$MAX_MODEL_LEN" \
+    --gpu-memory-utilization "$GPU_MEM_UTIL" \
+    --enable-lora \
+    --max-lora-rank 16 \
+    --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
+    "${QUANT_FLAG[@]}" \
+    "${REASONING_FLAG[@]}" \
+    "${LM_ONLY_FLAG[@]}" \
+    "${MAX_SEQS_FLAG[@]}" \
+    "${EXTRA_FLAGS[@]}"
+```
+
+Ensure LF line endings.
+
+- [ ] **Step 2: Syntax-check**
+
+```bash
+bash -n scripts/serve_with_lora.sh && echo "serve_with_lora.sh: OK"
+```
+
+---
+
+### Task 3: Harden `train_lora.py` for 27B QLoRA
+
+**Files:**
+- Modify: `scripts/train_lora.py`
+- Test: unit-level — prefer a small pure helper for target-module resolution if extracted; otherwise smoke via import + dry path that fails on missing `train.jsonl` (existing behavior)
+
+**Interfaces:**
+- Consumes: `TRAIN_MODEL` from env/config; `data/train.jsonl`
+- Produces: adapter at `output/lora_adapter/`; default smoke-friendly seq/batch when env unset **only if** documented — prefer keeping code defaults but document env overrides in README; optionally set code defaults to `1024`/`1` when `TRAIN_MODEL` contains `27B`
+
+- [ ] **Step 1: Update module docstring** to mention Qwen3.6-27B + `TRAIN_MODEL`
+
+- [ ] **Step 2: After loading the model, if PEFT target modules fail or to be proactive, resolve targets**
+
+Implement this behavior in `main()` before `get_peft_model`:
+
+```python
+    target_modules = list(LORA_TARGET_MODULES)
+    # If none of the named modules exist (architecture drift), fall back to all Linear names except embeddings/lm_head.
+    named = {n.split(".")[-1] for n, _ in model.named_modules()}
+    if not any(t in named for t in target_modules):
+        import torch.nn as nn
+        target_modules = sorted(
+            {
+                name.split(".")[-1]
+                for name, module in model.named_modules()
+                if isinstance(module, nn.Linear)
+                and name.split(".")[-1] not in {"lm_head"}
+            }
+        )
+        print(f"WARNING: default LoRA targets missing; using Linear modules: {target_modules}")
+```
+
+Use `target_modules=target_modules` in `LoraConfig`.
+
+- [ ] **Step 3: When `TRAIN_MODEL` / resolved base contains `27B`, print a reminder to stop the vLLM server and recommend smoke env knobs** (do not hard-fail).
+
+- [ ] **Step 4: Confirm missing-data path still works**
+
+```bash
+# from repo root with venv — expect exit 1 if train.jsonl absent
+python scripts/train_lora.py ; echo exit:$?
+```
+
+Expected: error about missing `data/train.jsonl` (or validation) without traceback noise from unrelated imports if possible — current script imports torch only after validation, keep that order.
+
+---
+
+### Task 4: Document the 27B LoRA loop in README
+
+**Files:**
+- Modify: `README.md`
+- Test: visual read + link to design exists
+
+**Interfaces:**
+- Consumes: Tasks 1–3 operator flow
+- Produces: section “LoRA on Qwen3.6-27B (example FAQ)”
+
+- [ ] **Step 1: Add a section after Default model** covering:
+
+1. Start server, generate from example FAQ  
+2. Auto-copy promote + validate  
+3. Stop server  
+4. Train with env knobs  
+5. Serve with LoRA + test client  
+
+Exact commands:
+
+```bash
+# 1) server already running with AWQ 27B
+python scripts/generate_training_data.py
+
+# 2) first-run promote (light review optional)
+mkdir -p data
+cp data/generated/raw_qa.jsonl data/train.jsonl
+python scripts/validate_dataset.py data/train.jsonl
+
+# 3) stop the vLLM server (Ctrl+C in its terminal), confirm VRAM free:
+nvidia-smi
+
+# 4) train (downloads dense Qwen/Qwen3.6-27B on first run — large)
+TRAIN_MODEL=Qwen/Qwen3.6-27B MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
+  python scripts/train_lora.py
+
+# 5) serve base + adapter
+./scripts/serve_with_lora.sh
+# other terminal:
+python scripts/test_client.py --model support-adapter
+```
+
+Also link:
+`docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md`
+
+- [ ] **Step 2: Confirm design file exists**
+
+```bash
+test -f docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md && echo LORA_DOC_OK
+```
+
+---
+
+### Task 5: End-to-end GPU run (manual)
+
+**Files:** none (runtime)
+
+**Interfaces:**
+- Consumes: Tasks 1–4
+- Produces: trained adapter + successful adapter chat reply
+
+- [ ] **Step 1: Ensure AWQ server is running** (`./scripts/start_server.sh`). If `data/generated/raw_qa.jsonl` already exists and is non-empty, move it aside first (generator refuses overwrite).
+
+- [ ] **Step 2: Generate + promote + validate**
+
+```bash
+source .venv/bin/activate
+python scripts/generate_training_data.py
+cp data/generated/raw_qa.jsonl data/train.jsonl
+python scripts/validate_dataset.py data/train.jsonl
+```
+
+Expected: validate prints success / line count > 0.
+
+- [ ] **Step 3: Stop server; train smoke**
+
+```bash
+TRAIN_MODEL=Qwen/Qwen3.6-27B MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
+  python scripts/train_lora.py
+```
+
+Expected: adapter files under `output/lora_adapter/`; final loss printed.
+
+If OOM: set `MAX_SEQ_LENGTH=512` and retry.
+
+- [ ] **Step 4: Serve with LoRA + test**
+
+```bash
+./scripts/serve_with_lora.sh
+# other shell:
+python scripts/test_client.py --model support-adapter
+```
+
+Expected: non-empty response, exit 0.
+
+- [ ] **Step 5: If AWQ+adapter load fails**, capture the error in the task report; do not silently switch training bases — escalate (design risk: dense-trained adapter on AWQ serve).
+
+---
+
+## Spec coverage self-check
+
+| Spec requirement | Task |
+|---|---|
+| `TRAIN_MODEL=Qwen/Qwen3.6-27B` | Task 1 |
+| Generate from example FAQ via running server | Task 5 (uses existing generator) |
+| Auto-copy promote + validate | Task 4 docs + Task 5 |
+| Stop server before train | Task 4/5 |
+| QLoRA train_lora.py / target fallback | Task 3 |
+| serve_with_lora aligned flags | Task 2 |
+| test_client on `support-adapter` | Task 5 |
+| No Unsloth rewrite | (no task) |
+| README | Task 4 |
diff --git a/docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md b/docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md
new file mode 100644
index 0000000..948ccd3
--- /dev/null
+++ b/docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md
@@ -0,0 +1,136 @@
+# Design: Personal tech pipeline (question-sharper → me-assistant)
+
+**Date:** 2026-08-08  
+**Status:** approved  
+**Depends on:** [2026-08-08-qwen36-27b-lora-train-design.md](2026-08-08-qwen36-27b-lora-train-design.md),
+[2026-08-08-qwen36-27b-awq-serve-design.md](2026-08-08-qwen36-27b-awq-serve-design.md)
+
+## Goal
+
+Build a **personal, trackable tech pipeline** on the existing Qwen3.6-27B AWQ +
+QLoRA stack:
+
+1. **Question sharper** — messy / vague tech thought → clear, focused question  
+2. **Me-assistant** — clear tech question → answer in the user’s preferred style  
+3. Chain them automatically (pipeline), with optional run logging for learning
+
+Domain for v1: **tech / coding / this ML setup** (not general life advice).
+
+## Decisions (locked)
+
+| Item | Choice |
+|---|---|
+| Packaging | Pipeline of **two adapters** (not one multi-task model) |
+| Domain | Tech / coding / ML setup |
+| Data source | Mine **Cursor agent transcripts** + **repo markdown** |
+| Bootstrap | Extract candidates → human review → train JSONL |
+| Train base | Dense `Qwen/Qwen3.6-27B` QLoRA (same as current loop) |
+| Serve base | Existing AWQ `MODEL` with both LoRA modules loaded |
+| Privacy | Personal JSONL stays local; gitignore private paths |
+
+## Architecture
+
+```
+messy thought
+    → LoRA: question-sharper
+    → clear tech question
+    → LoRA: me-assistant
+    → answer in user’s style
+    → optional log: output/personal_runs.jsonl
+```
+
+**Components**
+
+1. **Extract** — scripts turn transcripts + selected markdown into candidate pairs  
+2. **Review** — human keep / edit / reject; promote into train files  
+3. **Train** — two QLoRA adapters on dense 27B  
+4. **Serve** — one AWQ vLLM process registering both modules  
+5. **Pipeline client** — one command that chains sharper → assistant and prints both steps  
+
+## Data
+
+Same schema as the existing FAQ loop: one JSON object per line with
+`instruction` and `response`.
+
+| File | `instruction` | `response` |
+|---|---|---|
+| `data/personal/question_sharp.jsonl` | Messy / vague tech thought | Clear, focused question |
+| `data/personal/me_assistant.jsonl` | Clear tech question | Answer in preferred “me” style |
+
+**Staging**
+
+- Raw candidates: `data/personal/candidates/`  
+- Promote to the train JSONLs only after review  
+- First useful train target: **~50–100 pairs per adapter** (far more than the FAQ smoke test)
+
+**Sources**
+
+- Cursor agent transcripts for this project (user turns + endorsed assistant turns)  
+- Repo markdown (`docs/`, design notes, and other paths listed in config)
+
+**Extraction rules (v1)**
+
+- **Sharper:** user messages that look messy / multi-thought → one sharpened question (draft helper allowed; human edits)  
+- **Me-assistant:** clear questions paired with answers the user accepts as “how I want this answered”  
+- No auto-promote without review  
+
+**Privacy**
+
+- Do not commit personal train/candidate JSONL if it contains private chat text  
+- Add `data/personal/` (or at least candidates + train JSONL) to `.gitignore` unless the user explicitly opts into versioning sanitized data  
+
+## Train
+
+- Reuse `scripts/train_lora.py` with a small extension for train data path + output dir  
+- Outputs:
+  - `output/lora_question_sharper/`
+  - `output/lora_me_assistant/`
+- Same operational rule: stop the vLLM server before training to free VRAM  
+- Train each adapter separately from its JSONL  
+
+## Serve
+
+- Extend `scripts/serve_with_lora.sh` (or a sibling) to register both modules, e.g.:
+  - `question-sharper=<path>`
+  - `me-assistant=<path>`
+- Keep one AWQ base process; max LoRA rank unchanged (16) unless training changes rank  
+
+## Pipeline client
+
+New script (e.g. `scripts/personal_pipeline.py`):
+
+1. Call chat API with `model=question-sharper` and the raw user text  
+2. Abort with a clear error if the sharpened question is empty  
+3. Call chat API with `model=me-assistant` and the sharpened question  
+4. Print **both** steps (sharpened question + final answer)  
+5. Optionally append a JSON line to `output/personal_runs.jsonl`:
+   - timestamp, raw input, sharpened question, final answer  
+
+Missing adapters or server-down errors should fail clearly (same spirit as
+`test_client.py`).
+
+## Out of scope (v1)
+
+- Web search / tool use for factual grounding  
+- Non-tech domains  
+- Auto-train without human review  
+- Guaranteeing correctness beyond data quality  
+- Single multi-task adapter with `[SHARPEN]` / `[ANSWER]` tags  
+
+## Success criteria
+
+- Extract script produces candidates from transcripts + configured markdown  
+- After review, both train JSONLs pass `validate_dataset.py`  
+- Both adapters train and load together on one server  
+- Pipeline turns one messy tech thought into a clearer question + a useful tech answer  
+- Runs can be logged for later improvement  
+
+**Smoke test:** three messy prompts; sharpened questions are clearer than the
+raw input; answers feel closer to the user’s preferred style than base AWQ alone.
+
+## Risks / notes
+
+- Thin data → weak personalization; volume and review matter more than epochs  
+- Two adapters mean two train runs and slightly more VRAM for LoRA slots at serve time  
+- Transcript mining will be noisy; review is mandatory for quality  
+- Factual errors still possible; treat this as style + focus, not a truth oracle  
diff --git a/docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md b/docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md
new file mode 100644
index 0000000..ee60379
--- /dev/null
+++ b/docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md
@@ -0,0 +1,136 @@
+# Design: Qwen3.6-27B LoRA train (from example FAQ)
+
+**Date:** 2026-08-08  
+**Status:** approved  
+**Depends on:** [2026-08-08-qwen36-27b-awq-serve-design.md](2026-08-08-qwen36-27b-awq-serve-design.md)
+
+## Goal
+
+Run the existing customer-support LoRA loop against **Qwen3.6-27B** on this
+machine (24GB RTX 5090 Laptop / WSL2):
+
+1. Generate draft Q&A from `data/source_docs/example_faq.md` using the running
+   AWQ vLLM server.
+2. Promote data into `data/train.jsonl` (light review; first run may auto-copy).
+3. Stop the server, QLoRA-train a LoRA adapter on the **dense** base
+   `Qwen/Qwen3.6-27B`.
+4. Serve AWQ base + adapter via `serve_with_lora.sh`.
+
+## Context
+
+- Serving uses `shawnw3i/Qwen3.6-27B-AWQ-MTP` (compressed). QLoRA cannot train
+  on that checkpoint; training must load dense `Qwen/Qwen3.6-27B` in 4-bit.
+- `scripts/train_lora.py` already supports `TRAIN_MODEL` override and QLoRA via
+  transformers + PEFT + bitsandbytes + TRL.
+- There is no `data/train.jsonl` yet; only `example_faq.md`.
+- User chose: generate from the example FAQ (option A), Approach 1 (full 27B
+  QLoRA), design accepted 2026-08-08.
+
+## Decision
+
+| Item | Choice |
+|---|---|
+| Generation model | Running AWQ serve model (`MODEL` in `config/model.env`) |
+| Train base | `TRAIN_MODEL=Qwen/Qwen3.6-27B` (dense) |
+| Method | QLoRA 4-bit NF4, existing `train_lora.py` stack (not Unsloth-first) |
+| First-run data | Auto-copy `data/generated/raw_qa.jsonl` → `data/train.jsonl` after validate; keep manual review as documented optional step |
+| Train memory knobs | `MAX_SEQ_LENGTH=1024`, `BATCH_SIZE=1`, `GRADIENT_ACCUMULATION_STEPS` keep/raise as needed, `NUM_EPOCHS=1` for first smoke then 3 for real |
+| Adapter out | `output/lora_adapter/` (unchanged) |
+| Serve after train | AWQ `MODEL` + `--enable-lora` via `serve_with_lora.sh` |
+
+## Pipeline
+
+```
+start_server.sh (AWQ 27B)
+        │
+        ▼
+generate_training_data.py  →  data/generated/raw_qa.jsonl
+        │
+        ▼
+promote (auto-copy first run)  →  data/train.jsonl
+        │
+        ▼
+validate_dataset.py
+        │
+        ▼
+stop server (free ~20GB VRAM)
+        │
+        ▼
+TRAIN_MODEL=Qwen/Qwen3.6-27B train_lora.py  →  output/lora_adapter/
+        │
+        ▼
+serve_with_lora.sh  (AWQ base + adapter)
+        │
+        ▼
+test_client.py --model support-adapter
+```
+
+## File / config changes (expected)
+
+```
+config/model.env              # add TRAIN_MODEL=Qwen/Qwen3.6-27B
+scripts/train_lora.py         # 27B-safe defaults / target modules if needed
+scripts/serve_with_lora.sh    # reuse start_server flag pattern (reasoning, etc.) if missing
+README.md                     # document generate → train → serve-with-lora for 27B
+docs/... (this design)
+```
+
+Optional small helper (only if needed): `scripts/promote_generated_data.py` or a
+documented `cp` step — prefer documented copy over new code unless validation
+requires it.
+
+### `config/model.env` additions
+
+```env
+TRAIN_MODEL=Qwen/Qwen3.6-27B
+```
+
+Serving keys stay on the AWQ checkpoint. Training always prefers `TRAIN_MODEL`.
+
+### Training defaults for 24GB + 27B
+
+| Knob | First smoke | Follow-up |
+|---|---|---|
+| `MAX_SEQ_LENGTH` | 1024 | 2048 if VRAM allows |
+| `BATCH_SIZE` | 1 | 1–2 |
+| `NUM_EPOCHS` | 1 | 3 |
+| LoRA rank/alpha | 16/16 (existing) | unchanged unless OOM |
+
+If Qwen3.6 module names differ from the current
+`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` list, discover
+trainable Linear names once and update `LORA_TARGET_MODULES` (or use a PEFT
+regex / `all-linear` policy) so GDN-only layers are not required targets.
+
+## Verification
+
+1. Server up → `python scripts/generate_training_data.py` writes non-empty
+   `raw_qa.jsonl`.
+2. Promote → `python scripts/validate_dataset.py` passes on `train.jsonl`.
+3. Server stopped; `nvidia-smi` shows free VRAM.
+4. `TRAIN_MODEL=Qwen/Qwen3.6-27B MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 python scripts/train_lora.py`
+   completes and writes `output/lora_adapter/`.
+5. `./scripts/serve_with_lora.sh` starts; `python scripts/test_client.py --model support-adapter`
+   returns a non-empty reply.
+
+## Risks
+
+- **VRAM:** 27B QLoRA on 24GB is tight; OOM → shorten seq length, ensure server
+  is fully dead, enable more aggressive checkpointing if needed.
+- **Architecture:** Qwen3.6 hybrid (GDN + attention) may not match Unsloth; stay
+  on transformers/PEFT. Some modules may not accept LoRA — target attention/MLP
+  projections that exist.
+- **Download:** Dense `Qwen/Qwen3.6-27B` is a large first-time download (~50GB+
+  before 4-bit runtime load).
+- **Data quality:** Auto-copy is fine for a first end-to-end proof; real support
+  use still wants human review of `raw_qa.jsonl`.
+- **Serve+LoRA:** Adapter trained on dense base must load against AWQ serve
+  weights in vLLM; if incompatible, document fallback (serve dense quantized or
+  merge — out of scope for v1 unless smoke fails).
+
+## Out of scope
+
+- Unsloth-primary rewrite
+- Multi-GPU / DeepSpeed
+- Training directly on the AWQ weights
+- Replacing the example FAQ with real product corpora (later)
+- MTP speculative decoding during serve-with-lora
diff --git a/scripts/extract_personal_candidates.py b/scripts/extract_personal_candidates.py
new file mode 100644
index 0000000..c1783bb
--- /dev/null
+++ b/scripts/extract_personal_candidates.py
@@ -0,0 +1,77 @@
+#!/usr/bin/env python3
+"""Mine transcripts + markdown into personal candidate JSONL files."""
+from __future__ import annotations
+
+import json
+import os
+import sys
+from pathlib import Path
+
+sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
+
+from scripts.lib.env_config import load_env_file
+from scripts.lib.personal_extract import (
+    iter_transcript_qa_pairs,
+    iter_transcript_user_texts,
+    pairs_from_markdown,
+    sharpen_candidates_from_texts,
+)
+
+REPO_ROOT = Path(__file__).resolve().parent.parent
+OUT_DIR = REPO_ROOT / "data" / "personal" / "candidates"
+
+
+def _expand(p: str) -> Path:
+    return Path(os.path.expanduser(p)).expanduser()
+
+
+def main() -> int:
+    cfg = load_env_file(REPO_ROOT / "config" / "personal_sources.env")
+    transcripts = _expand(
+        os.environ.get("AGENT_TRANSCRIPTS_DIR") or cfg.get("AGENT_TRANSCRIPTS_DIR", "")
+    )
+    globs = (
+        os.environ.get("MARKDOWN_GLOBS") or cfg.get("MARKDOWN_GLOBS", "README.md")
+    ).split(",")
+
+    sharpen: list[dict] = []
+    me: list[dict] = []
+
+    if transcripts.is_dir():
+        for path in sorted(transcripts.rglob("*.jsonl")):
+            src = str(path)
+            sharpen.extend(sharpen_candidates_from_texts(iter_transcript_user_texts(path), src))
+            for q, a in iter_transcript_qa_pairs(path):
+                me.append(
+                    {
+                        "instruction": q,
+                        "response": a,
+                        "source": src,
+                        "kind": "me_assistant",
+                    }
+                )
+    else:
+        print(f"WARNING: transcripts dir missing: {transcripts}", file=sys.stderr)
+
+    for pattern in globs:
+        pattern = pattern.strip()
+        if not pattern:
+            continue
+        for path in sorted(REPO_ROOT.glob(pattern)):
+            if not path.is_file():
+                continue
+            me.extend(pairs_from_markdown(path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))))
+
+    OUT_DIR.mkdir(parents=True, exist_ok=True)
+    sharp_path = OUT_DIR / "question_sharp.jsonl"
+    me_path = OUT_DIR / "me_assistant.jsonl"
+    sharp_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in sharpen) + ("\n" if sharpen else ""), encoding="utf-8")
+    me_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in me) + ("\n" if me else ""), encoding="utf-8")
+    print(f"Wrote {len(sharpen)} sharpen candidates → {sharp_path}")
+    print(f"Wrote {len(me)} me_assistant candidates → {me_path}")
+    print("Review/edit candidates, then: python scripts/promote_personal_data.py --reviewed")
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/scripts/lib/personal_extract.py b/scripts/lib/personal_extract.py
new file mode 100644
index 0000000..fce6658
--- /dev/null
+++ b/scripts/lib/personal_extract.py
@@ -0,0 +1,145 @@
+"""Extract personal LoRA candidate pairs from Cursor transcripts and markdown."""
+from __future__ import annotations
+
+import json
+import re
+from pathlib import Path
+
+_USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL | re.IGNORECASE)
+
+
+def extract_user_query(text: str) -> str | None:
+    text = (text or "").strip()
+    if not text:
+        return None
+    m = _USER_QUERY_RE.search(text)
+    if m:
+        q = m.group(1).strip()
+        return q or None
+    # Skip obvious system/tool dumps
+    if text.startswith("{") and '"role"' in text:
+        return None
+    return text
+
+
+def draft_sharpen(messy: str) -> str:
+    """Heuristic draft only — human must review before train promote."""
+    cleaned = re.sub(r"\s+", " ", (messy or "").strip())
+    if not cleaned:
+        return ""
+    if cleaned[-1] not in ".?!":
+        cleaned += "?"
+    # Prefer a single question-shaped line
+    if len(cleaned) > 240:
+        cleaned = cleaned[:237].rstrip() + "..."
+    return cleaned[0].upper() + cleaned[1:] if cleaned else ""
+
+
+def _message_text(obj: dict) -> str:
+    msg = obj.get("message") or {}
+    content = msg.get("content")
+    if isinstance(content, str):
+        return content
+    if isinstance(content, list):
+        parts: list[str] = []
+        for block in content:
+            if isinstance(block, dict) and block.get("type") == "text":
+                parts.append(str(block.get("text") or ""))
+            elif isinstance(block, str):
+                parts.append(block)
+        return "\n".join(parts)
+    return ""
+
+
+def iter_transcript_user_texts(path: Path) -> list[str]:
+    out: list[str] = []
+    for line in Path(path).read_text(encoding="utf-8").splitlines():
+        if not line.strip():
+            continue
+        try:
+            obj = json.loads(line)
+        except json.JSONDecodeError:
+            continue
+        if obj.get("role") != "user":
+            continue
+        q = extract_user_query(_message_text(obj))
+        if q:
+            out.append(q)
+    return out
+
+
+def iter_transcript_qa_pairs(path: Path) -> list[tuple[str, str]]:
+    pairs: list[tuple[str, str]] = []
+    pending: str | None = None
+    for line in Path(path).read_text(encoding="utf-8").splitlines():
+        if not line.strip():
+            continue
+        try:
+            obj = json.loads(line)
+        except json.JSONDecodeError:
+            continue
+        role = obj.get("role")
+        text = _message_text(obj)
+        if role == "user":
+            pending = extract_user_query(text)
+        elif role == "assistant" and pending:
+            # First text paragraph only — drop huge tool dumps
+            reply = text.strip().split("\n\n")[0].strip()
+            if reply and len(reply) < 4000:
+                pairs.append((pending, reply))
+            pending = None
+    return pairs
+
+
+def pairs_from_markdown(text: str, source: str) -> list[dict]:
+    """Turn markdown H1/H2 sections into me_assistant candidates."""
+    out: list[dict] = []
+    parts = re.findall(
+        r"(?m)^(#{1,2}\s+[^\n]+)$" r"(.*?)(?=^#{1,2}\s+|\Z)", text, flags=re.DOTALL
+    )
+    if not parts:
+        body = text.strip()
+        if body:
+            title = Path(source).stem.replace("-", " ")
+            out.append(
+                {
+                    "instruction": f"What should I know about {title}?",
+                    "response": body[:2000],
+                    "source": source,
+                    "kind": "me_assistant",
+                }
+            )
+        return out
+    for heading_line, body in parts:
+        title = re.sub(r"^#{1,2}\s+", "", heading_line).strip()
+        body = body.strip()
+        if not title or not body:
+            continue
+        out.append(
+            {
+                "instruction": f"Explain: {title}",
+                "response": body[:2000],
+                "source": source,
+                "kind": "me_assistant",
+            }
+        )
+    return out
+
+
+def sharpen_candidates_from_texts(texts: list[str], source: str) -> list[dict]:
+    out: list[dict] = []
+    for t in texts:
+        sharp = draft_sharpen(t)
+        if not sharp or sharp == t:
+            # still keep if messy enough (length or newlines originally)
+            if len(t) < 20:
+                continue
+        out.append(
+            {
+                "instruction": t,
+                "response": sharp or draft_sharpen(t),
+                "source": source,
+                "kind": "sharpen",
+            }
+        )
+    return out
diff --git a/scripts/lib/personal_pipeline.py b/scripts/lib/personal_pipeline.py
new file mode 100644
index 0000000..b27ba35
--- /dev/null
+++ b/scripts/lib/personal_pipeline.py
@@ -0,0 +1,27 @@
+"""Chain question-sharper → me-assistant over an OpenAI-compatible client."""
+from __future__ import annotations
+
+from typing import Any
+
+
+def run_pipeline(
+    client: Any,
+    raw: str,
+    *,
+    sharp_model: str,
+    answer_model: str,
+) -> dict[str, str]:
+    sharp = client.chat.completions.create(
+        model=sharp_model,
+        messages=[{"role": "user", "content": raw}],
+    )
+    sharpened = (sharp.choices[0].message.content or "").strip()
+    if not sharpened:
+        raise ValueError("question-sharper returned an empty question")
+
+    ans = client.chat.completions.create(
+        model=answer_model,
+        messages=[{"role": "user", "content": sharpened}],
+    )
+    answer = (ans.choices[0].message.content or "").strip()
+    return {"raw": raw, "sharpened": sharpened, "answer": answer}
diff --git a/scripts/personal_pipeline.py b/scripts/personal_pipeline.py
new file mode 100644
index 0000000..1f35165
--- /dev/null
+++ b/scripts/personal_pipeline.py
@@ -0,0 +1,65 @@
+#!/usr/bin/env python3
+"""Run personal tech pipeline: sharpen question, then answer as me-assistant."""
+from __future__ import annotations
+
+import argparse
+import json
+import sys
+from datetime import datetime, timezone
+from pathlib import Path
+
+sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
+
+from openai import OpenAI
+
+from scripts.lib.env_config import load_env_file
+from scripts.lib.personal_pipeline import run_pipeline
+
+REPO_ROOT = Path(__file__).resolve().parent.parent
+LOG_PATH = REPO_ROOT / "output" / "personal_runs.jsonl"
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("prompt", help="Messy tech thought / question")
+    parser.add_argument("--sharp-model", default="question-sharper")
+    parser.add_argument("--answer-model", default="me-assistant")
+    parser.add_argument("--no-log", action="store_true")
+    args = parser.parse_args()
+
+    config = load_env_file(REPO_ROOT / "config" / "model.env")
+    port = config.get("PORT", "8000")
+    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")
+
+    try:
+        result = run_pipeline(
+            client,
+            args.prompt,
+            sharp_model=args.sharp_model,
+            answer_model=args.answer_model,
+        )
+    except ValueError as exc:
+        print(f"ERROR: {exc}", file=sys.stderr)
+        return 1
+    except Exception as exc:
+        print(f"FAILED: {exc}", file=sys.stderr)
+        return 1
+
+    print(f"Raw: {result['raw']}")
+    print(f"Sharpened: {result['sharpened']}")
+    print(f"Answer: {result['answer']}")
+
+    if not args.no_log:
+        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
+        row = {
+            "ts": datetime.now(timezone.utc).isoformat(),
+            **result,
+        }
+        with LOG_PATH.open("a", encoding="utf-8") as fh:
+            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
+        print(f"Logged → {LOG_PATH}")
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/scripts/promote_personal_data.py b/scripts/promote_personal_data.py
new file mode 100644
index 0000000..9ffabb9
--- /dev/null
+++ b/scripts/promote_personal_data.py
@@ -0,0 +1,63 @@
+#!/usr/bin/env python3
+"""Promote reviewed personal candidates to train JSONL (instruction/response only)."""
+from __future__ import annotations
+
+import argparse
+import json
+import sys
+from pathlib import Path
+
+sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
+
+from scripts.lib.dataset_validation import validate_dataset_file
+
+REPO_ROOT = Path(__file__).resolve().parent.parent
+CAND = REPO_ROOT / "data" / "personal" / "candidates"
+OUT = REPO_ROOT / "data" / "personal"
+
+
+def _strip_meta(path: Path, dest: Path) -> int:
+    rows: list[dict] = []
+    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
+        if not line.strip():
+            continue
+        obj = json.loads(line)
+        row = {"instruction": obj["instruction"], "response": obj["response"]}
+        rows.append(row)
+    dest.parent.mkdir(parents=True, exist_ok=True)
+    dest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
+    return len(rows)
+
+
+def main() -> int:
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument(
+        "--reviewed",
+        action="store_true",
+        help="Required confirmation that candidates were human-reviewed",
+    )
+    args = parser.parse_args()
+    if not args.reviewed:
+        print("ERROR: refusing to promote without --reviewed", file=sys.stderr)
+        return 1
+
+    mapping = [
+        (CAND / "question_sharp.jsonl", OUT / "question_sharp.jsonl"),
+        (CAND / "me_assistant.jsonl", OUT / "me_assistant.jsonl"),
+    ]
+    for src, dest in mapping:
+        if not src.exists():
+            print(f"ERROR: missing {src}", file=sys.stderr)
+            return 1
+        n = _strip_meta(src, dest)
+        errors = validate_dataset_file(dest)
+        if errors:
+            print(f"ERROR: {dest} invalid:", file=sys.stderr)
+            print("\n".join(errors[:20]), file=sys.stderr)
+            return 1
+        print(f"Promoted {n} rows → {dest}")
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/scripts/serve_with_lora.sh b/scripts/serve_with_lora.sh
index 3e658a8..90c48e4 100644
--- a/scripts/serve_with_lora.sh
+++ b/scripts/serve_with_lora.sh
@@ -1,34 +1,77 @@
 #!/usr/bin/env bash
+# Multi-LoRA: LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" ./scripts/serve_with_lora.sh
 set -euo pipefail
 
 REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
+# shellcheck disable=SC1091
+if [ -f "$REPO_ROOT/scripts/wsl_runtime_env.sh" ]; then
+    source "$REPO_ROOT/scripts/wsl_runtime_env.sh"
+fi
+# shellcheck disable=SC1091
 source "$REPO_ROOT/config/model.env"
 
 QUANT_FLAG=()
 if [ "${QUANTIZATION:-none}" != "none" ]; then
     QUANT_FLAG=(--quantization "$QUANTIZATION")
 fi
 
+REASONING_FLAG=()
+if [ -n "${REASONING_PARSER:-}" ]; then
+    REASONING_FLAG=(--reasoning-parser "$REASONING_PARSER")
+fi
+
+LM_ONLY_FLAG=()
+if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ] || [ "${LANGUAGE_MODEL_ONLY:-}" = "true" ]; then
+    LM_ONLY_FLAG=(--language-model-only)
+fi
+
+MAX_SEQS_FLAG=()
+if [ -n "${MAX_NUM_SEQS:-}" ]; then
+    MAX_SEQS_FLAG=(--max-num-seqs "$MAX_NUM_SEQS")
+fi
+
 EXTRA_FLAGS=()
 if [ -n "${EXTRA_ARGS:-}" ]; then
     # shellcheck disable=SC2206
     EXTRA_FLAGS=($EXTRA_ARGS)
 fi
 
-ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
-if [ ! -d "$ADAPTER_PATH" ]; then
-    echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
-    exit 1
+LORA_MODULE_ARGS=()
+if [ -n "${LORA_MODULES:-}" ]; then
+    IFS=',' read -r -a _mods <<< "$LORA_MODULES"
+    for spec in "${_mods[@]}"; do
+        spec_trimmed="$(echo "$spec" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
+        name="${spec_trimmed%%=*}"
+        path="${spec_trimmed#*=}"
+        if [[ "$path" != /* ]]; then
+            path="$REPO_ROOT/$path"
+        fi
+        if [ ! -d "$path" ]; then
+            echo "ERROR: LoRA path missing for $name: $path" >&2
+            exit 1
+        fi
+        LORA_MODULE_ARGS+=("${name}=${path}")
+    done
+else
+    ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
+    if [ ! -d "$ADAPTER_PATH" ]; then
+        echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
+        exit 1
+    fi
+    LORA_MODULE_ARGS+=("${ADAPTER_NAME}=${ADAPTER_PATH}")
 fi
 
-echo "Starting vLLM server with LoRA: model=$MODEL adapter=$ADAPTER_NAME port=$PORT"
+echo "Starting vLLM server with LoRA modules: ${LORA_MODULE_ARGS[*]} port=$PORT"
 
 vllm serve "$MODEL" \
     --port "$PORT" \
     --max-model-len "$MAX_MODEL_LEN" \
     --gpu-memory-utilization "$GPU_MEM_UTIL" \
     --enable-lora \
     --max-lora-rank 16 \
-    --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
+    --lora-modules "${LORA_MODULE_ARGS[@]}" \
     "${QUANT_FLAG[@]}" \
+    "${REASONING_FLAG[@]}" \
+    "${LM_ONLY_FLAG[@]}" \
+    "${MAX_SEQS_FLAG[@]}" \
     "${EXTRA_FLAGS[@]}"
diff --git a/scripts/train_lora.py b/scripts/train_lora.py
index dca556b..2fdad5e 100644
--- a/scripts/train_lora.py
+++ b/scripts/train_lora.py
@@ -1,25 +1,30 @@
 #!/usr/bin/env python3
-"""Fine-tune a LoRA adapter for Qwen3-4B on customer-support Q&A data.
+"""Fine-tune a LoRA adapter for Qwen3 / Qwen3.6-27B on customer-support Q&A data.
 
 Uses 4-bit QLoRA via transformers + PEFT + bitsandbytes + TRL, sized for
 consumer GPUs. Reads data/train.jsonl (validate first with
 scripts/validate_dataset.py) and writes the trained adapter to
 output/lora_adapter/.
 
 Usage:
   python scripts/train_lora.py
 
-On 6GB cards / when serving uses an AWQ checkpoint:
-  TRAIN_MODEL=Qwen/Qwen3-4B-Instruct-2507 MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 \\
+Serving may use an AWQ checkpoint while training needs the dense base. Set
+TRAIN_MODEL in config/model.env or the environment (e.g.
+TRAIN_MODEL=Qwen/Qwen3.6-27B for 27B QLoRA).
+
+On 6GB cards / tight VRAM (especially 27B):
+  TRAIN_MODEL=Qwen/Qwen3.6-27B MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \\
     python scripts/train_lora.py
 """
+import argparse
 import os
 import sys
 from pathlib import Path
 
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 from scripts.lib.dataset_validation import validate_dataset_file
 from scripts.lib.env_config import load_env_file
 
 REPO_ROOT = Path(__file__).resolve().parent.parent
@@ -31,39 +36,60 @@ MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
 LORA_RANK = 16
 LORA_ALPHA = 16
 LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
 BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
 GRADIENT_ACCUMULATION_STEPS = 4
 NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
 LEARNING_RATE = 2e-4
 
 
 def main() -> int:
-    if not TRAIN_DATA_PATH.exists():
-        print(f"ERROR: {TRAIN_DATA_PATH} not found. Create it first (see README).", file=sys.stderr)
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--data", default=None, help="Train JSONL path (default data/train.jsonl)")
+    parser.add_argument("--output", default=None, help="Adapter output dir (default output/lora_adapter)")
+    args = parser.parse_args()
+
+    train_data = Path(
+        args.data
+        or os.environ.get("TRAIN_DATA")
+        or (REPO_ROOT / "data" / "train.jsonl")
+    )
+    output_dir = Path(
+        args.output
+        or os.environ.get("TRAIN_OUTPUT")
+        or (REPO_ROOT / "output" / "lora_adapter")
+    )
+
+    if not train_data.exists():
+        print(f"ERROR: {train_data} not found. Create it first (see README).", file=sys.stderr)
         return 1
 
-    errors = validate_dataset_file(TRAIN_DATA_PATH)
+    errors = validate_dataset_file(train_data)
     if errors:
         print(
-            f"ERROR: {TRAIN_DATA_PATH} has {len(errors)} invalid line(s). "
+            f"ERROR: {train_data} has {len(errors)} invalid line(s). "
             "Run scripts/validate_dataset.py for details.",
             file=sys.stderr,
         )
         return 1
 
     config = load_env_file(REPO_ROOT / "config" / "model.env")
     # Serving may use an AWQ/compressed checkpoint; QLoRA training needs the
     # dense base instruct model. Override with TRAIN_MODEL when they differ.
     base_model = os.environ.get("TRAIN_MODEL") or config.get("TRAIN_MODEL") or config["MODEL"]
     print(f"Training LoRA on base model: {base_model}")
-    print(f"Examples: {TRAIN_DATA_PATH} | seq={MAX_SEQ_LENGTH} batch={BATCH_SIZE} epochs={NUM_EPOCHS}")
+    print(f"Examples: {train_data} | seq={MAX_SEQ_LENGTH} batch={BATCH_SIZE} epochs={NUM_EPOCHS}")
+    if "27B" in base_model.upper():
+        print(
+            "NOTE: Stop the vLLM server before training to free VRAM. "
+            "Recommended: MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1"
+        )
 
     import torch
     from datasets import load_dataset
     from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
     from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
     from trl import SFTConfig, SFTTrainer
 
     if not torch.cuda.is_available():
         print("ERROR: CUDA is not available to torch.", file=sys.stderr)
         return 1
@@ -75,38 +101,51 @@ def main() -> int:
         bnb_4bit_use_double_quant=True,
     )
 
     tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
     if tokenizer.pad_token is None:
         tokenizer.pad_token = tokenizer.eos_token
 
     model = AutoModelForCausalLM.from_pretrained(
         base_model,
         quantization_config=bnb_config,
-        device_map="auto",
+        # Single-GPU QLoRA: keep the whole 4-bit model on cuda:0 (avoid CPU offload rejection).
+        device_map={"": 0},
         trust_remote_code=True,
     )
     model = prepare_model_for_kbit_training(model)
+    target_modules = list(LORA_TARGET_MODULES)
+    named = {n.split(".")[-1] for n, _ in model.named_modules()}
+    if not any(t in named for t in target_modules):
+        target_modules = sorted(
+            {
+                name.split(".")[-1]
+                for name, module in model.named_modules()
+                if isinstance(module, torch.nn.Linear)
+                and name.split(".")[-1] not in {"lm_head"}
+            }
+        )
+        print(f"WARNING: default LoRA targets missing; using Linear modules: {target_modules}")
     model = get_peft_model(
         model,
         LoraConfig(
             r=LORA_RANK,
             lora_alpha=LORA_ALPHA,
-            target_modules=LORA_TARGET_MODULES,
+            target_modules=target_modules,
             lora_dropout=0.0,
             bias="none",
             task_type="CAUSAL_LM",
         ),
     )
     model.print_trainable_parameters()
 
-    dataset = load_dataset("json", data_files=str(TRAIN_DATA_PATH), split="train")
+    dataset = load_dataset("json", data_files=str(train_data), split="train")
 
     def format_example(example: dict) -> dict:
         messages = [
             {"role": "user", "content": example["instruction"]},
             {"role": "assistant", "content": example["response"]},
         ]
         return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
 
     dataset = dataset.map(format_example)
 
@@ -114,31 +153,31 @@ def main() -> int:
         model=model,
         processing_class=tokenizer,
         train_dataset=dataset,
         args=SFTConfig(
             dataset_text_field="text",
             max_length=MAX_SEQ_LENGTH,
             per_device_train_batch_size=BATCH_SIZE,
             gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
             num_train_epochs=NUM_EPOCHS,
             learning_rate=LEARNING_RATE,
-            output_dir=str(OUTPUT_DIR / "checkpoints"),
+            output_dir=str(output_dir / "checkpoints"),
             logging_steps=1,
             save_strategy="no",
             report_to="none",
             bf16=True,
             gradient_checkpointing=True,
         ),
     )
 
     result = trainer.train()
     print(f"Final training loss: {result.training_loss:.4f}")
 
-    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
-    model.save_pretrained(str(OUTPUT_DIR))
-    tokenizer.save_pretrained(str(OUTPUT_DIR))
-    print(f"Adapter saved to {OUTPUT_DIR}")
+    output_dir.mkdir(parents=True, exist_ok=True)
+    model.save_pretrained(str(output_dir))
+    tokenizer.save_pretrained(str(output_dir))
+    print(f"Adapter saved to {output_dir}")
     return 0
 
 
 if __name__ == "__main__":
     raise SystemExit(main())
diff --git a/tests/test_personal_extract.py b/tests/test_personal_extract.py
new file mode 100644
index 0000000..ee5ff3e
--- /dev/null
+++ b/tests/test_personal_extract.py
@@ -0,0 +1,55 @@
+from pathlib import Path
+
+from scripts.lib.personal_extract import (
+    draft_sharpen,
+    extract_user_query,
+    iter_transcript_qa_pairs,
+    iter_transcript_user_texts,
+    pairs_from_markdown,
+)
+
+
+def test_extract_user_query_from_wrapper():
+    raw = "<user_query>\nokey can we fix the train OOM?\n</user_query>"
+    assert extract_user_query(raw) == "okey can we fix the train OOM?"
+
+
+def test_extract_user_query_plain_fallback():
+    assert extract_user_query("plain question about LoRA") == "plain question about LoRA"
+
+
+def test_draft_sharpen_collapses_whitespace():
+    messy = "okey   so like\n\ncan we train  two adapters??"
+    sharp = draft_sharpen(messy)
+    assert "  " not in sharp
+    assert "?" in sharp or sharp.endswith("adapters")
+
+
+def test_iter_transcript_user_texts(tmp_path: Path):
+    p = tmp_path / "t.jsonl"
+    p.write_text(
+        '{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\\nfix download\\n</user_query>"}]}}\n'
+        '{"role":"assistant","message":{"content":[{"type":"text","text":"checking..."}]}}\n',
+        encoding="utf-8",
+    )
+    texts = iter_transcript_user_texts(p)
+    assert texts == ["fix download"]
+
+
+def test_iter_transcript_qa_pairs(tmp_path: Path):
+    p = tmp_path / "t.jsonl"
+    p.write_text(
+        '{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\\nhow do I serve LoRA?\\n</user_query>"}]}}\n'
+        '{"role":"assistant","message":{"content":[{"type":"text","text":"Use serve_with_lora.sh"}]}}\n',
+        encoding="utf-8",
+    )
+    pairs = iter_transcript_qa_pairs(p)
+    assert pairs == [("how do I serve LoRA?", "Use serve_with_lora.sh")]
+
+
+def test_pairs_from_markdown_heading_chunks():
+    md = "# Serve\n\nUse AWQ + adapter.\n\n# Train\n\nStop server first.\n"
+    pairs = pairs_from_markdown(md, source="docs/x.md")
+    assert len(pairs) >= 1
+    assert all(p["kind"] == "me_assistant" for p in pairs)
+    assert all(p["instruction"] and p["response"] for p in pairs)
diff --git a/tests/test_personal_pipeline.py b/tests/test_personal_pipeline.py
new file mode 100644
index 0000000..b14a2d4
--- /dev/null
+++ b/tests/test_personal_pipeline.py
@@ -0,0 +1,53 @@
+from types import SimpleNamespace
+
+from scripts.lib.personal_pipeline import run_pipeline
+
+
+class _FakeCompletions:
+    def __init__(self):
+        self.calls = []
+
+    def create(self, *, model, messages):
+        self.calls.append((model, messages[0]["content"]))
+        if model == "question-sharper":
+            text = "How do I free VRAM before QLoRA training?"
+        else:
+            text = "Stop the vLLM server, then run nvidia-smi."
+        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
+
+
+class _FakeClient:
+    def __init__(self):
+        self.chat = SimpleNamespace(completions=_FakeCompletions())
+
+
+def test_run_pipeline_chains_models():
+    client = _FakeClient()
+    result = run_pipeline(
+        client,
+        "okey so like train fails maybe vram?",
+        sharp_model="question-sharper",
+        answer_model="me-assistant",
+    )
+    assert result["sharpened"].startswith("How do I")
+    assert "vLLM" in result["answer"]
+    assert [c[0] for c in client.chat.completions.calls] == [
+        "question-sharper",
+        "me-assistant",
+    ]
+
+
+def test_run_pipeline_aborts_on_empty_sharpen():
+    class EmptySharp(_FakeCompletions):
+        def create(self, *, model, messages):
+            self.calls.append((model, messages[0]["content"]))
+            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  "))])
+
+    client = _FakeClient()
+    client.chat.completions = EmptySharp()
+    try:
+        run_pipeline(client, "x", sharp_model="question-sharper", answer_model="me-assistant")
+        assert False, "expected ValueError"
+    except ValueError as exc:
+        assert "empty" in str(exc).lower()
+    assert len(client.chat.completions.calls) == 1
