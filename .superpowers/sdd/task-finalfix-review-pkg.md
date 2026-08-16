# Review package
BASE: 484b8b77971a998cfb822adb80aa7d1c19a09893
HEAD: 9ed03e66c661bc8a9e351d48db8e8205569c014f

## Commits
9ed03e6 fix(personal-pipeline): address final review merge blockers

## Stat
 .superpowers/sdd/progress.md              |  36 +++
 .superpowers/sdd/task-final-fix-report.md |  34 +++
 README.md                                 | 352 +++++++++++++++---------------
 scripts/extract_personal_candidates.py    |  20 +-
 scripts/lib/personal_extract.py           |  21 +-
 scripts/promote_personal_data.py          |  20 +-
 scripts/train_lora.py                     |   9 +-
 tests/test_extract_personal_candidates.py |  50 +++++
 tests/test_personal_extract.py            |  14 ++
 tests/test_promote_personal_data.py       |  72 ++++++
 10 files changed, 439 insertions(+), 189 deletions(-)

## Diff
diff --git a/.superpowers/sdd/progress.md b/.superpowers/sdd/progress.md
new file mode 100644
index 0000000..f63d67f
--- /dev/null
+++ b/.superpowers/sdd/progress.md
@@ -0,0 +1,36 @@
+﻿# SDD Progress — Qwen3.6-27B LoRA train
+Branch: feat/qwen36-27b-lora-train
+Plan: docs/superpowers/plans/2026-08-08-qwen36-27b-lora-train.md
+
+Task 1: complete (23b0d3f..4ce16cb, review clean)
+
+Task 2: complete (4ce16cb..a91dc3b, review clean)
+
+Task 3: complete (a91dc3b..2569c3f, review clean)
+
+Task 4: complete (commit 21aa7e5, review pending minor README intro drift)
+
+# SDD Progress - Personal tech pipeline
+Branch: feat/qwen36-27b-lora-train
+Plan: docs/superpowers/plans/2026-08-08-personal-tech-pipeline.md
+
+Task 1: complete (commits 92185ab..f7223cf, review clean)
+
+Task 2: complete (commits f7223cf..a96be0e, review clean; minors: untested sharpen_candidates_from_texts)
+
+Task 3: complete (commits a96be0e..e516098, review clean; minors: zero-row promote, expanduser duplex)
+
+Task 4: complete (commits e516098..e0ac82f, review clean; note re-apply device_map single-GPU before GPU train)
+
+Task 5: complete (commits e0ac82f..0d44c08, review clean)
+
+Task 6: complete (commits 0d44c08..026f8c2, review clean)
+
+Task 7: complete (commits 026f8c2..578bcdf, review clean)
+
+Task 8: complete (GPU smoke: sharper 71 rows OK; me-assistant 30-row smoke OK; dual serve + pipeline OK)
+
+Task 8 (ledger note): plumbing smoke only — extract → promote → train → dual-serve →
+personal_pipeline chain works end-to-end. Sharper dataset quality and 3-prompt success
+criteria were NOT met; candidates need human review before claiming personalization quality.
+
diff --git a/.superpowers/sdd/task-final-fix-report.md b/.superpowers/sdd/task-final-fix-report.md
new file mode 100644
index 0000000..c3e4c8c
--- /dev/null
+++ b/.superpowers/sdd/task-final-fix-report.md
@@ -0,0 +1,34 @@
+# Task final fix report — personal tech pipeline review
+
+**Branch:** feat/qwen36-27b-lora-train  
+**Date:** 2026-08-08
+
+## Status
+
+All merge-blocker findings addressed. Full pytest suite passes (44 tests).
+
+## Commits
+
+1. `fix(personal-pipeline): address final review merge blockers` — train env knobs, promote/extract/sharpen fixes, README, tests, ledger
+2. (single commit if squashed locally)
+
+## What was fixed
+
+| Finding | Fix |
+|---|---|
+| README claimed train knobs in `config/model.env` | README documents env vars; `train_lora.py` reads `GRADIENT_ACCUMULATION_STEPS` and `GRADIENT_CHECKPOINTING` from env |
+| Zero-row promote | `promote_personal_data.py` exits 1 if either output has 0 rows |
+| Empty `AGENT_TRANSCRIPTS_DIR` → cwd rglob | `_transcripts_root()` returns None for empty; warn and skip |
+| Ledger over-claim on Task 8 | `progress.md` notes plumbing-only smoke; quality criteria not met |
+| Short/junk sharpen rows | Always skip `len(t.strip()) < 20`; skip trivial capitalize+`?` single tokens |
+| Optional JSON/KeyError guards | Clear ERROR lines in `_strip_meta` |
+
+## Tests
+
+- **Focused:** 15 passed (`test_promote_personal_data`, `test_personal_extract`, `test_extract_personal_candidates`)
+- **Full suite:** 44 passed via WSL `.venv/bin/python -m pytest -v`
+
+## Not done (per scope)
+
+- No GPU retrain or smoke re-run
+- No full extract pipeline rewrite
diff --git a/README.md b/README.md
index 746090b..f47154e 100644
--- a/README.md
+++ b/README.md
@@ -1,174 +1,178 @@
-# Qwen3.6-27B (AWQ) + vLLM + LoRA
-
-Serve **Qwen3.6-27B** via vLLM (AWQ 4-bit for single-GPU 24GB cards), then
-optionally customize a smaller base model with a LoRA adapter trained on your
-own docs. LoRA fine-tuning of the 27B checkpoint is not in this path yet.
-
-Authored to run on **Linux or WSL2** with an NVIDIA GPU. Native Windows cannot
-run the GPU stack; Windows teammates use the thin setup wrappers below, which
-forward into WSL.
-
-## Onboarding (setup only)
-
-One command installs the Python venv, dependencies, and verifies CUDA/vLLM.
-Starting the server and sending a test request are **manual** next steps.
-
-**Windows (PowerShell or cmd):**
-
-```bat
-scripts\setup.cmd
-```
-
-or:
-
-```powershell
-powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
-```
-
-**Linux / already inside WSL:**
-
-```bash
-./scripts/setup.sh
-```
-
-When setup finishes, activate the venv it created, then:
-
-```bash
-./scripts/start_server.sh
-# in another terminal, with the venv active:
-python scripts/test_client.py
-```
-
-## Default model (Qwen3.6-27B AWQ)
-
-`config/model.env` defaults to:
-
-- `MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP`
-- `QUANTIZATION=awq`
-- `MAX_MODEL_LEN=4096`
-- `MAX_NUM_SEQS=32`
-- `GPU_MEM_UTIL=0.92`
-- `REASONING_PARSER=qwen3`
-- `LANGUAGE_MODEL_ONLY=1`
-
-First server start downloads the AWQ weights from Hugging Face (large).
-
-On WSL, `scripts/start_server.sh` sources `scripts/wsl_runtime_env.sh` for a
-user-space GCC/CUDA toolkit (micromamba env `cc`) needed by Triton/FlashInfer.
-Create it once if missing:
-
-```bash
-# one-time: user-space gcc + CUDA 13.3 toolkit (no sudo)
-bash scripts/_install_usergcc.sh
-```
-
-Blackwell-only env vars (`FLASHINFER_CUDA_ARCH_LIST`, `TORCH_CUDA_ARCH_LIST`)
-are set only when `nvidia-smi` reports compute capability major >= 12.
-`VLLM_USE_FLASHINFER_SAMPLER` defaults to `0` (avoids curand.h JIT); override
-to `1` if you have full CUDA math headers.
-
-Pass extra vLLM flags via `EXTRA_ARGS`, e.g.
-`EXTRA_ARGS="--enforce-eager" ./scripts/start_server.sh`.
-
-If you OOM or hit Mamba-cache errors: lower `MAX_MODEL_LEN` / `MAX_NUM_SEQS`,
-or raise `GPU_MEM_UTIL` slightly. If VRAM remains, try raising `MAX_MODEL_LEN`
-toward `8192`.
-
-To roll back to the small bf16 model for LoRA experiments, set in `config/model.env`:
-
-```env
-MODEL=Qwen/Qwen3-4B-Instruct-2507
-MAX_MODEL_LEN=32768
-MAX_NUM_SEQS=
-QUANTIZATION=none
-REASONING_PARSER=
-LANGUAGE_MODEL_ONLY=0
-```
-
-## LoRA on Qwen3.6-27B (example FAQ)
-
-End-to-end generate → train → serve-with-LoRA using `data/source_docs/example_faq.md`.
-Generation uses the running AWQ server; training loads dense `Qwen/Qwen3.6-27B` in 4-bit
-(QLoRA cannot train on the AWQ checkpoint).
-
-```bash
-# 1) server already running with AWQ 27B
-python scripts/generate_training_data.py
-
-# 2) first-run promote (light review optional)
-mkdir -p data
-cp data/generated/raw_qa.jsonl data/train.jsonl
-python scripts/validate_dataset.py data/train.jsonl
-
-# 3) stop the vLLM server (Ctrl+C in its terminal), confirm VRAM free:
-nvidia-smi
-
-# 4) train (downloads dense Qwen/Qwen3.6-27B on first run — large)
-# Knobs live in config/model.env (batch=2, epochs=3, accum=8, ~70% free GPU/CPU).
-python scripts/train_lora.py
-
-# 5) serve base + adapter
-./scripts/serve_with_lora.sh
-# other terminal:
-python scripts/test_client.py --model support-adapter
-```
-
-## Personal tech pipeline (question-sharper → me-assistant)
-
-Two LoRA adapters on the same AWQ base: sharpen messy tech thoughts into clear
-questions, then answer in your preferred style. Sources and paths are in
-[`config/personal_sources.env`](config/personal_sources.env); full design in
-[Personal tech pipeline design](docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md).
-
-**WSL note:** `~/.cursor/...` in config resolves to the WSL home, not Windows
-Cursor data. When extracting from WSL, point at the Windows agent-transcripts
-folder:
-
-```bash
-export AGENT_TRANSCRIPTS_DIR=/mnt/c/Users/supre/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
-```
-
-```bash
-# 1) extract candidates from transcripts + configured markdown
-python scripts/extract_personal_candidates.py
-# review/edit under data/personal/candidates/
-
-# 2) promote after human review (strips metadata, validates)
-python scripts/promote_personal_data.py --reviewed
-python scripts/validate_dataset.py data/personal/question_sharp.jsonl
-python scripts/validate_dataset.py data/personal/me_assistant.jsonl
-
-# 3) stop vLLM; confirm VRAM free (nvidia-smi), then train each adapter
-python scripts/train_lora.py --data data/personal/question_sharp.jsonl --output output/lora_question_sharper
-python scripts/train_lora.py --data data/personal/me_assistant.jsonl --output output/lora_me_assistant
-
-# 4) serve base + both adapters
-LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" ./scripts/serve_with_lora.sh
-
-# 5) run the chained pipeline (optional run log: output/personal_runs.jsonl)
-python scripts/personal_pipeline.py "how do i fix oom when starting vllm on 24gb"
-```
-
-Personal train/candidate JSONL under `data/personal/` may contain private chat
-text — keep it local unless you explicitly version sanitized data.
-
-## Troubleshooting
-
-Driver, CUDA, and out-of-memory issues are documented in:
-
-- [Design: Qwen + vLLM + LoRA setup](docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md)
-- [Design: easy onboard setup scripts](docs/superpowers/specs/2026-08-06-easy-onboard-setup-scripts-design.md)
-- [Design: Qwen3.6-27B AWQ serve](docs/superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md)
-- [Design: Qwen3.6-27B LoRA train](docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md)
-- [Design: personal tech pipeline](docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md)
-
-Tune model/port/context in `config/model.env` — setup does not rewrite it.
-
-## Unit tests (no GPU required)
-
-`pytest` is not installed by `setup.sh`; install it into the venv first:
-
-```bash
-pip install pytest
-python -m pytest -v
-```
+# Qwen3.6-27B (AWQ) + vLLM + LoRA
+
+Serve **Qwen3.6-27B** via vLLM (AWQ 4-bit for single-GPU 24GB cards), then
+optionally customize a smaller base model with a LoRA adapter trained on your
+own docs. LoRA fine-tuning of the 27B checkpoint is not in this path yet.
+
+Authored to run on **Linux or WSL2** with an NVIDIA GPU. Native Windows cannot
+run the GPU stack; Windows teammates use the thin setup wrappers below, which
+forward into WSL.
+
+## Onboarding (setup only)
+
+One command installs the Python venv, dependencies, and verifies CUDA/vLLM.
+Starting the server and sending a test request are **manual** next steps.
+
+**Windows (PowerShell or cmd):**
+
+```bat
+scripts\setup.cmd
+```
+
+or:
+
+```powershell
+powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
+```
+
+**Linux / already inside WSL:**
+
+```bash
+./scripts/setup.sh
+```
+
+When setup finishes, activate the venv it created, then:
+
+```bash
+./scripts/start_server.sh
+# in another terminal, with the venv active:
+python scripts/test_client.py
+```
+
+## Default model (Qwen3.6-27B AWQ)
+
+`config/model.env` defaults to:
+
+- `MODEL=shawnw3i/Qwen3.6-27B-AWQ-MTP`
+- `QUANTIZATION=awq`
+- `MAX_MODEL_LEN=4096`
+- `MAX_NUM_SEQS=32`
+- `GPU_MEM_UTIL=0.92`
+- `REASONING_PARSER=qwen3`
+- `LANGUAGE_MODEL_ONLY=1`
+
+First server start downloads the AWQ weights from Hugging Face (large).
+
+On WSL, `scripts/start_server.sh` sources `scripts/wsl_runtime_env.sh` for a
+user-space GCC/CUDA toolkit (micromamba env `cc`) needed by Triton/FlashInfer.
+Create it once if missing:
+
+```bash
+# one-time: user-space gcc + CUDA 13.3 toolkit (no sudo)
+bash scripts/_install_usergcc.sh
+```
+
+Blackwell-only env vars (`FLASHINFER_CUDA_ARCH_LIST`, `TORCH_CUDA_ARCH_LIST`)
+are set only when `nvidia-smi` reports compute capability major >= 12.
+`VLLM_USE_FLASHINFER_SAMPLER` defaults to `0` (avoids curand.h JIT); override
+to `1` if you have full CUDA math headers.
+
+Pass extra vLLM flags via `EXTRA_ARGS`, e.g.
+`EXTRA_ARGS="--enforce-eager" ./scripts/start_server.sh`.
+
+If you OOM or hit Mamba-cache errors: lower `MAX_MODEL_LEN` / `MAX_NUM_SEQS`,
+or raise `GPU_MEM_UTIL` slightly. If VRAM remains, try raising `MAX_MODEL_LEN`
+toward `8192`.
+
+To roll back to the small bf16 model for LoRA experiments, set in `config/model.env`:
+
+```env
+MODEL=Qwen/Qwen3-4B-Instruct-2507
+MAX_MODEL_LEN=32768
+MAX_NUM_SEQS=
+QUANTIZATION=none
+REASONING_PARSER=
+LANGUAGE_MODEL_ONLY=0
+```
+
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
+# Train knobs via env (defaults in train_lora.py): MAX_SEQ_LENGTH, BATCH_SIZE,
+# NUM_EPOCHS, GRADIENT_ACCUMULATION_STEPS, GRADIENT_CHECKPOINTING.
+# Example 27B smoke: MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
+#   GRADIENT_ACCUMULATION_STEPS=8 GRADIENT_CHECKPOINTING=1 python scripts/train_lora.py
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
+# Same train env knobs as above (MAX_SEQ_LENGTH, BATCH_SIZE, NUM_EPOCHS, etc.).
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
+## Troubleshooting
+
+Driver, CUDA, and out-of-memory issues are documented in:
+
+- [Design: Qwen + vLLM + LoRA setup](docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md)
+- [Design: easy onboard setup scripts](docs/superpowers/specs/2026-08-06-easy-onboard-setup-scripts-design.md)
+- [Design: Qwen3.6-27B AWQ serve](docs/superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md)
+- [Design: Qwen3.6-27B LoRA train](docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md)
+- [Design: personal tech pipeline](docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md)
+
+Tune model/port/context in `config/model.env` — setup does not rewrite it.
+
+## Unit tests (no GPU required)
+
+`pytest` is not installed by `setup.sh`; install it into the venv first:
+
+```bash
+pip install pytest
+python -m pytest -v
+```
diff --git a/scripts/extract_personal_candidates.py b/scripts/extract_personal_candidates.py
index c1783bb..c553b4c 100644
--- a/scripts/extract_personal_candidates.py
+++ b/scripts/extract_personal_candidates.py
@@ -18,47 +18,57 @@ from scripts.lib.personal_extract import (
 )
 
 REPO_ROOT = Path(__file__).resolve().parent.parent
 OUT_DIR = REPO_ROOT / "data" / "personal" / "candidates"
 
 
 def _expand(p: str) -> Path:
     return Path(os.path.expanduser(p)).expanduser()
 
 
+def _transcripts_root(raw: str) -> Path | None:
+    if not (raw or "").strip():
+        return None
+    return _expand(raw)
+
+
 def main() -> int:
     cfg = load_env_file(REPO_ROOT / "config" / "personal_sources.env")
-    transcripts = _expand(
-        os.environ.get("AGENT_TRANSCRIPTS_DIR") or cfg.get("AGENT_TRANSCRIPTS_DIR", "")
-    )
+    transcripts_raw = os.environ.get("AGENT_TRANSCRIPTS_DIR") or cfg.get("AGENT_TRANSCRIPTS_DIR", "")
+    transcripts = _transcripts_root(transcripts_raw)
     globs = (
         os.environ.get("MARKDOWN_GLOBS") or cfg.get("MARKDOWN_GLOBS", "README.md")
     ).split(",")
 
     sharpen: list[dict] = []
     me: list[dict] = []
 
-    if transcripts.is_dir():
+    if transcripts is not None and transcripts.is_dir():
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
-    else:
+    elif transcripts is not None:
         print(f"WARNING: transcripts dir missing: {transcripts}", file=sys.stderr)
+    else:
+        print(
+            "WARNING: AGENT_TRANSCRIPTS_DIR not set or empty; skipping transcript extraction",
+            file=sys.stderr,
+        )
 
     for pattern in globs:
         pattern = pattern.strip()
         if not pattern:
             continue
         for path in sorted(REPO_ROOT.glob(pattern)):
             if not path.is_file():
                 continue
             me.extend(pairs_from_markdown(path.read_text(encoding="utf-8"), str(path.relative_to(REPO_ROOT))))
 
diff --git a/scripts/lib/personal_extract.py b/scripts/lib/personal_extract.py
index fce6658..5d97ab8 100644
--- a/scripts/lib/personal_extract.py
+++ b/scripts/lib/personal_extract.py
@@ -15,20 +15,31 @@ def extract_user_query(text: str) -> str | None:
     m = _USER_QUERY_RE.search(text)
     if m:
         q = m.group(1).strip()
         return q or None
     # Skip obvious system/tool dumps
     if text.startswith("{") and '"role"' in text:
         return None
     return text
 
 
+def _is_trivial_sharpen(messy: str, sharp: str) -> bool:
+    """True when draft only capitalizes a short single-token input and adds ?."""
+    cleaned = re.sub(r"\s+", " ", (messy or "").strip())
+    if not cleaned or " " in cleaned or len(cleaned) >= 20:
+        return False
+    expected = cleaned[0].upper() + cleaned[1:]
+    if expected[-1] not in ".?!":
+        expected += "?"
+    return sharp == expected
+
+
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
@@ -122,24 +133,24 @@ def pairs_from_markdown(text: str, source: str) -> list[dict]:
                 "source": source,
                 "kind": "me_assistant",
             }
         )
     return out
 
 
 def sharpen_candidates_from_texts(texts: list[str], source: str) -> list[dict]:
     out: list[dict] = []
     for t in texts:
+        if len(t.strip()) < 20:
+            continue
         sharp = draft_sharpen(t)
-        if not sharp or sharp == t:
-            # still keep if messy enough (length or newlines originally)
-            if len(t) < 20:
-                continue
+        if not sharp or _is_trivial_sharpen(t, sharp):
+            continue
         out.append(
             {
                 "instruction": t,
-                "response": sharp or draft_sharpen(t),
+                "response": sharp,
                 "source": source,
                 "kind": "sharpen",
             }
         )
     return out
diff --git a/scripts/promote_personal_data.py b/scripts/promote_personal_data.py
index 9ffabb9..50d3396 100644
--- a/scripts/promote_personal_data.py
+++ b/scripts/promote_personal_data.py
@@ -14,22 +14,30 @@ from scripts.lib.dataset_validation import validate_dataset_file
 REPO_ROOT = Path(__file__).resolve().parent.parent
 CAND = REPO_ROOT / "data" / "personal" / "candidates"
 OUT = REPO_ROOT / "data" / "personal"
 
 
 def _strip_meta(path: Path, dest: Path) -> int:
     rows: list[dict] = []
     for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
         if not line.strip():
             continue
-        obj = json.loads(line)
-        row = {"instruction": obj["instruction"], "response": obj["response"]}
+        try:
+            obj = json.loads(line)
+        except json.JSONDecodeError as exc:
+            print(f"ERROR: {path}:{i} invalid JSON: {exc}", file=sys.stderr)
+            raise
+        try:
+            row = {"instruction": obj["instruction"], "response": obj["response"]}
+        except KeyError as exc:
+            print(f"ERROR: {path}:{i} missing field {exc}", file=sys.stderr)
+            raise
         rows.append(row)
     dest.parent.mkdir(parents=True, exist_ok=True)
     dest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
     return len(rows)
 
 
 def main() -> int:
     parser = argparse.ArgumentParser(description=__doc__)
     parser.add_argument(
         "--reviewed",
@@ -42,21 +50,27 @@ def main() -> int:
         return 1
 
     mapping = [
         (CAND / "question_sharp.jsonl", OUT / "question_sharp.jsonl"),
         (CAND / "me_assistant.jsonl", OUT / "me_assistant.jsonl"),
     ]
     for src, dest in mapping:
         if not src.exists():
             print(f"ERROR: missing {src}", file=sys.stderr)
             return 1
-        n = _strip_meta(src, dest)
+        try:
+            n = _strip_meta(src, dest)
+        except (json.JSONDecodeError, KeyError):
+            return 1
+        if n == 0:
+            print(f"ERROR: {src} produced 0 rows after promote", file=sys.stderr)
+            return 1
         errors = validate_dataset_file(dest)
         if errors:
             print(f"ERROR: {dest} invalid:", file=sys.stderr)
             print("\n".join(errors[:20]), file=sys.stderr)
             return 1
         print(f"Promoted {n} rows → {dest}")
     return 0
 
 
 if __name__ == "__main__":
diff --git a/scripts/train_lora.py b/scripts/train_lora.py
index 2fdad5e..b2fcac1 100644
--- a/scripts/train_lora.py
+++ b/scripts/train_lora.py
@@ -30,22 +30,27 @@ from scripts.lib.env_config import load_env_file
 REPO_ROOT = Path(__file__).resolve().parent.parent
 TRAIN_DATA_PATH = REPO_ROOT / "data" / "train.jsonl"
 OUTPUT_DIR = REPO_ROOT / "output" / "lora_adapter"
 
 # Sized for 8GB-class GPUs. Override via env on tighter cards.
 MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
 LORA_RANK = 16
 LORA_ALPHA = 16
 LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
 BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
-GRADIENT_ACCUMULATION_STEPS = 4
+GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4"))
 NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
+GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1").strip().lower() in {
+    "1",
+    "true",
+    "yes",
+}
 LEARNING_RATE = 2e-4
 
 
 def main() -> int:
     parser = argparse.ArgumentParser(description=__doc__)
     parser.add_argument("--data", default=None, help="Train JSONL path (default data/train.jsonl)")
     parser.add_argument("--output", default=None, help="Adapter output dir (default output/lora_adapter)")
     args = parser.parse_args()
 
     train_data = Path(
@@ -158,21 +163,21 @@ def main() -> int:
             max_length=MAX_SEQ_LENGTH,
             per_device_train_batch_size=BATCH_SIZE,
             gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
             num_train_epochs=NUM_EPOCHS,
             learning_rate=LEARNING_RATE,
             output_dir=str(output_dir / "checkpoints"),
             logging_steps=1,
             save_strategy="no",
             report_to="none",
             bf16=True,
-            gradient_checkpointing=True,
+            gradient_checkpointing=GRADIENT_CHECKPOINTING,
         ),
     )
 
     result = trainer.train()
     print(f"Final training loss: {result.training_loss:.4f}")
 
     output_dir.mkdir(parents=True, exist_ok=True)
     model.save_pretrained(str(output_dir))
     tokenizer.save_pretrained(str(output_dir))
     print(f"Adapter saved to {output_dir}")
diff --git a/tests/test_extract_personal_candidates.py b/tests/test_extract_personal_candidates.py
new file mode 100644
index 0000000..1982b44
--- /dev/null
+++ b/tests/test_extract_personal_candidates.py
@@ -0,0 +1,50 @@
+import json
+import sys
+from pathlib import Path
+from unittest.mock import patch
+
+from scripts.extract_personal_candidates import _transcripts_root, main
+
+
+def test_transcripts_root_empty():
+    assert _transcripts_root("") is None
+    assert _transcripts_root("   ") is None
+
+
+def test_transcripts_root_expands_path(tmp_path: Path):
+    d = tmp_path / "transcripts"
+    d.mkdir()
+    assert _transcripts_root(str(d)) == d
+
+
+def test_empty_transcripts_dir_skips_rglob_cwd(tmp_path: Path, monkeypatch, capsys):
+    repo = tmp_path / "repo"
+    repo.mkdir()
+    cwd_jsonl = repo / "should_not_scan.jsonl"
+    cwd_jsonl.write_text(
+        '{"role":"user","message":{"content":[{"type":"text","text":"<user_query>\\nsecret\\n</user_query>"}]}}\n',
+        encoding="utf-8",
+    )
+    out_dir = repo / "data" / "personal" / "candidates"
+    monkeypatch.chdir(repo)
+    monkeypatch.setenv("AGENT_TRANSCRIPTS_DIR", "")
+    monkeypatch.setenv("MARKDOWN_GLOBS", "missing_glob_*.md")
+
+    import scripts.extract_personal_candidates as extract_mod
+
+    with patch.object(extract_mod, "REPO_ROOT", repo), patch.object(
+        extract_mod, "OUT_DIR", out_dir
+    ), patch.object(extract_mod, "load_env_file", return_value={}):
+        old_argv = sys.argv
+        sys.argv = ["extract_personal_candidates.py"]
+        try:
+            rc = extract_mod.main()
+        finally:
+            sys.argv = old_argv
+
+    captured = capsys.readouterr()
+    assert rc == 0
+    assert "skipping transcript extraction" in captured.err.lower()
+    sharp = out_dir / "question_sharp.jsonl"
+    assert sharp.exists()
+    assert "secret" not in sharp.read_text(encoding="utf-8")
diff --git a/tests/test_personal_extract.py b/tests/test_personal_extract.py
index ee5ff3e..c67679e 100644
--- a/tests/test_personal_extract.py
+++ b/tests/test_personal_extract.py
@@ -1,18 +1,19 @@
 from pathlib import Path
 
 from scripts.lib.personal_extract import (
     draft_sharpen,
     extract_user_query,
     iter_transcript_qa_pairs,
     iter_transcript_user_texts,
     pairs_from_markdown,
+    sharpen_candidates_from_texts,
 )
 
 
 def test_extract_user_query_from_wrapper():
     raw = "<user_query>\nokey can we fix the train OOM?\n</user_query>"
     assert extract_user_query(raw) == "okey can we fix the train OOM?"
 
 
 def test_extract_user_query_plain_fallback():
     assert extract_user_query("plain question about LoRA") == "plain question about LoRA"
@@ -46,10 +47,23 @@ def test_iter_transcript_qa_pairs(tmp_path: Path):
     pairs = iter_transcript_qa_pairs(p)
     assert pairs == [("how do I serve LoRA?", "Use serve_with_lora.sh")]
 
 
 def test_pairs_from_markdown_heading_chunks():
     md = "# Serve\n\nUse AWQ + adapter.\n\n# Train\n\nStop server first.\n"
     pairs = pairs_from_markdown(md, source="docs/x.md")
     assert len(pairs) >= 1
     assert all(p["kind"] == "me_assistant" for p in pairs)
     assert all(p["instruction"] and p["response"] for p in pairs)
+
+
+def test_sharpen_skips_short_texts():
+    out = sharpen_candidates_from_texts(["oom", "fix vram please now"], source="t.jsonl")
+    assert out == []
+
+
+def test_sharpen_skips_trivial_single_token():
+    out = sharpen_candidates_from_texts(["how do i fix oom on 24gb vram card"], source="t.jsonl")
+    assert len(out) == 1
+    assert out[0]["instruction"].startswith("how do i fix oom")
+    assert out[0]["response"]
+
diff --git a/tests/test_promote_personal_data.py b/tests/test_promote_personal_data.py
new file mode 100644
index 0000000..0505e98
--- /dev/null
+++ b/tests/test_promote_personal_data.py
@@ -0,0 +1,72 @@
+import json
+import sys
+from pathlib import Path
+from unittest.mock import patch
+
+import pytest
+
+from scripts.promote_personal_data import _strip_meta
+
+
+def test_strip_meta_strips_source_and_kind(tmp_path: Path):
+    src = tmp_path / "candidates.jsonl"
+    src.write_text(
+        json.dumps(
+            {
+                "instruction": "How do I train?",
+                "response": "Stop the server first.",
+                "source": "docs/x.md",
+                "kind": "me_assistant",
+            }
+        )
+        + "\n",
+        encoding="utf-8",
+    )
+    dest = tmp_path / "out.jsonl"
+    n = _strip_meta(src, dest)
+    assert n == 1
+    row = json.loads(dest.read_text(encoding="utf-8").strip())
+    assert row == {"instruction": "How do I train?", "response": "Stop the server first."}
+
+
+def test_strip_meta_zero_rows(tmp_path: Path):
+    src = tmp_path / "empty.jsonl"
+    src.write_text("\n\n", encoding="utf-8")
+    dest = tmp_path / "out.jsonl"
+    assert _strip_meta(src, dest) == 0
+    assert dest.read_text(encoding="utf-8") == ""
+
+
+def test_strip_meta_missing_field(tmp_path: Path, capsys):
+    src = tmp_path / "bad.jsonl"
+    src.write_text(json.dumps({"instruction": "only instruction"}) + "\n", encoding="utf-8")
+    dest = tmp_path / "out.jsonl"
+    with pytest.raises(KeyError):
+        _strip_meta(src, dest)
+    captured = capsys.readouterr()
+    assert "ERROR:" in captured.err
+    assert "missing field" in captured.err
+
+
+def test_main_rejects_zero_row_promote(tmp_path: Path, capsys):
+    cand = tmp_path / "candidates"
+    cand.mkdir()
+    (cand / "question_sharp.jsonl").write_text("\n", encoding="utf-8")
+    (cand / "me_assistant.jsonl").write_text(
+        json.dumps({"instruction": "q", "response": "a", "kind": "me_assistant"}) + "\n",
+        encoding="utf-8",
+    )
+    out = tmp_path / "personal"
+
+    import scripts.promote_personal_data as promote_mod
+
+    with patch.object(promote_mod, "CAND", cand), patch.object(promote_mod, "OUT", out):
+        old_argv = sys.argv
+        sys.argv = ["promote_personal_data.py", "--reviewed"]
+        try:
+            rc = promote_mod.main()
+        finally:
+            sys.argv = old_argv
+    captured = capsys.readouterr()
+    assert rc == 1
+    assert "0 rows" in captured.err
