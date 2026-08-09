# Review package
BASE: 026f8c20419d5a771c553a12a28f605580ff4e6c
HEAD: 578bcdf432dd3562789b9196068ff1d3a39b66bd

## Commits
578bcdf docs: document personal tech LoRA pipeline

## Stat
 README.md                   | 44 ++++++++++++++++++++++++++++++++++++++++++--
 config/personal_sources.env |  2 ++
 2 files changed, 44 insertions(+), 2 deletions(-)

## Diff
diff --git a/README.md b/README.md
index 957e337..746090b 100644
--- a/README.md
+++ b/README.md
@@ -97,37 +97,77 @@ python scripts/generate_training_data.py
 
 # 2) first-run promote (light review optional)
 mkdir -p data
 cp data/generated/raw_qa.jsonl data/train.jsonl
 python scripts/validate_dataset.py data/train.jsonl
 
 # 3) stop the vLLM server (Ctrl+C in its terminal), confirm VRAM free:
 nvidia-smi
 
 # 4) train (downloads dense Qwen/Qwen3.6-27B on first run — large)
-TRAIN_MODEL=Qwen/Qwen3.6-27B MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
-  python scripts/train_lora.py
+# Knobs live in config/model.env (batch=2, epochs=3, accum=8, ~70% free GPU/CPU).
+python scripts/train_lora.py
 
 # 5) serve base + adapter
 ./scripts/serve_with_lora.sh
 # other terminal:
 python scripts/test_client.py --model support-adapter
 ```
 
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
 - [Design: Qwen3.6-27B LoRA train](docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md)
+- [Design: personal tech pipeline](docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md)
 
 Tune model/port/context in `config/model.env` — setup does not rewrite it.
 
 ## Unit tests (no GPU required)
 
 `pytest` is not installed by `setup.sh`; install it into the venv first:
 
 ```bash
 pip install pytest
 python -m pytest -v
diff --git a/config/personal_sources.env b/config/personal_sources.env
index 999bde9..99e07c5 100644
--- a/config/personal_sources.env
+++ b/config/personal_sources.env
@@ -1,3 +1,5 @@
 # Absolute or ~ paths OK. Override with env vars of the same name.
+# WSL: ~ resolves to WSL $HOME, not Windows Cursor. Use /mnt/c/... path instead, e.g.:
+# AGENT_TRANSCRIPTS_DIR=/mnt/c/Users/supre/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
 AGENT_TRANSCRIPTS_DIR=~/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
 MARKDOWN_GLOBS=docs/superpowers/specs/*.md,docs/superpowers/plans/*.md,README.md
