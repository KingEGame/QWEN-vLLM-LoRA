BASE dc05aa0a18dd4d646a146f8da41bc5fac3069751
HEAD ed39be0
ed39be0 docs: link learning guides from README
95cf6af docs: add operator cheatsheet
df724b3 docs: add architecture learning guide
 README.md                            |   7 ++
 docs/guides/architecture-learning.md | 115 +++++++++++++++++++++++++++++++++
 docs/guides/operator-cheatsheet.md   | 120 +++++++++++++++++++++++++++++++++++
 3 files changed, 242 insertions(+)
diff --git a/README.md b/README.md
index f47154e..7492f24 100644
--- a/README.md
+++ b/README.md
@@ -6,10 +6,15 @@ own docs. LoRA fine-tuning of the 27B checkpoint is not in this path yet.
 
 Authored to run on **Linux or WSL2** with an NVIDIA GPU. Native Windows cannot
 run the GPU stack; Windows teammates use the thin setup wrappers below, which
 forward into WSL.
 
+## Guides
+
+- [Architecture learning](docs/guides/architecture-learning.md) — what we built, vLLM vs LoRA vs Qwen, limits
+- [Operator cheatsheet](docs/guides/operator-cheatsheet.md) — commands by scenario
+
 ## Onboarding (setup only)
 
 One command installs the Python venv, dependencies, and verifies CUDA/vLLM.
 Starting the server and sending a test request are **manual** next steps.
 
@@ -158,10 +163,12 @@ text — keep it local unless you explicitly version sanitized data.
 
 ## Troubleshooting
 
 Driver, CUDA, and out-of-memory issues are documented in:
 
+- [Architecture learning](docs/guides/architecture-learning.md)
+- [Operator cheatsheet](docs/guides/operator-cheatsheet.md)
 - [Design: Qwen + vLLM + LoRA setup](docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md)
 - [Design: easy onboard setup scripts](docs/superpowers/specs/2026-08-06-easy-onboard-setup-scripts-design.md)
 - [Design: Qwen3.6-27B AWQ serve](docs/superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md)
 - [Design: Qwen3.6-27B LoRA train](docs/superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md)
 - [Design: personal tech pipeline](docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md)
diff --git a/docs/guides/architecture-learning.md b/docs/guides/architecture-learning.md
new file mode 100644
index 0000000..a00f4de
--- /dev/null
+++ b/docs/guides/architecture-learning.md
@@ -0,0 +1,115 @@
+# System architecture learning
+
+Companion: [Operator cheatsheet](operator-cheatsheet.md)
+
+Pre-push learning note for this repo’s Qwen3.6-27B + vLLM + LoRA work (local
+`master` may be ahead of `origin/master`).
+
+## What we prepared to achieve
+
+1. Serve **Qwen3.6-27B** on a **24GB** laptop GPU via **AWQ** + vLLM (WSL2).
+2. Train **LoRA** adapters with **QLoRA** on dense `Qwen/Qwen3.6-27B` (cannot train on AWQ).
+3. Build a **personal tech pipeline**: messy thought → `question-sharper` → clear question → `me-assistant` → answer.
+
+Design trail:
+
+- [AWQ serve](../superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md)
+- [LoRA train](../superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md)
+- [Personal pipeline](../superpowers/specs/2026-08-08-personal-tech-pipeline-design.md)
+
+## What we achieved
+
+| Track | Status |
+|---|---|
+| AWQ serve on 24GB | Working (tuned `MAX_MODEL_LEN` / `MAX_NUM_SEQS`) |
+| FAQ LoRA smoke | Working end-to-end on tiny `data/train.jsonl` |
+| Personal extract / promote / train / multi-LoRA serve / pipeline client | **Plumbing works** |
+| Personalization quality (“understands me”) | **Not yet** — candidates need real human edit; smoke used noisy / thin data |
+
+## Are the results good?
+
+**Serve / train infrastructure: yes.**  
+**Personal adapters as “your model”: not yet.**
+
+Evidence from the personal smoke: the pipeline returns Raw / Sharpened / Answer and logs to `output/personal_runs.jsonl`, but sharpened text was often long lecture-style output rather than a tighter question — a data + review problem, not a missing script.
+
+LoRA trainable share observed in training logs was on the order of **~0.3%** of parameters (`print_trainable_parameters` in [`scripts/train_lora.py`](../../scripts/train_lora.py#L144)).
+
+## Who does what (responsibilities)
+
+```
+User prompt
+   │
+   ▼
+vLLM (runtime)  ── loads AWQ base, KV cache, OpenAI HTTP API, optional LoRA slots
+   │
+   ├── Qwen3.6-27B base weights  ≈ ~99%+ of knowledge / reasoning / default style
+   └── LoRA adapter(s)           ≈ small delta (rank 16) steering task/style
+```
+
+| Piece | Responsibility | Rough “% of behavior” |
+|---|---|---|
+| **vLLM** | Serve fast, manage VRAM, expose `/v1/chat/completions`, attach LoRAs | 0% of *language knowledge*; 100% of *how inference is run* |
+| **Qwen3.6-27B (AWQ at serve / dense at train)** | Almost all understanding and generation skill | **~99%+** |
+| **LoRA** | Thin task/style overlay learned from your JSONL | **~0.3% params**; influence can be large *on the trained task* if data is good, tiny if data is bad |
+
+Code anchors:
+
+- Base serve: [`scripts/start_server.sh`](../../scripts/start_server.sh#L40) (`vllm serve`)
+- LoRA serve: [`scripts/serve_with_lora.sh`](../../scripts/serve_with_lora.sh#L40-L72) (`LORA_MODULES` / `--lora-modules`)
+- Train rank / load: [`scripts/train_lora.py`](../../scripts/train_lora.py#L36-L37), [`#L117`](../../scripts/train_lora.py#L117) (`device_map={"": 0}`)
+- Pipeline chain: [`scripts/lib/personal_pipeline.py`](../../scripts/lib/personal_pipeline.py#L7-L27)
+
+## Script map (what / why)
+
+| Script | Why it exists |
+|---|---|
+| [`scripts/setup.sh`](../../scripts/setup.sh) | Create `.venv`, install deps, CUDA check |
+| [`scripts/wsl_runtime_env.sh`](../../scripts/wsl_runtime_env.sh) | User-space GCC/CUDA for Triton on WSL |
+| [`scripts/start_server.sh`](../../scripts/start_server.sh) | AWQ base server from [`config/model.env`](../../config/model.env) |
+| [`scripts/test_client.py`](../../scripts/test_client.py) | One-shot OpenAI client smoke |
+| [`scripts/generate_training_data.py`](../../scripts/generate_training_data.py) | FAQ → draft Q&A via running server |
+| [`scripts/validate_dataset.py`](../../scripts/validate_dataset.py) | JSONL `instruction`/`response` gate |
+| [`scripts/train_lora.py`](../../scripts/train_lora.py) | QLoRA train; `--data` / `--output` |
+| [`scripts/serve_with_lora.sh`](../../scripts/serve_with_lora.sh) | AWQ + one or many LoRAs |
+| [`scripts/extract_personal_candidates.py`](../../scripts/extract_personal_candidates.py) | Mine transcripts + markdown → candidates |
+| [`scripts/promote_personal_data.py`](../../scripts/promote_personal_data.py) | Promote only with `--reviewed` |
+| [`scripts/personal_pipeline.py`](../../scripts/personal_pipeline.py) | Sharper → assistant + optional log |
+
+Helpers named `scripts/_*.sh` are **local ops** (prefetch, progress, SDD); prefer not to treat them as product API.
+
+## Dependency map (depends on what)
+
+```
+config/model.env ──► start_server.sh / serve_with_lora.sh / test_client.py / train (TRAIN_MODEL)
+config/personal_sources.env ──► extract_personal_candidates.py
+data/personal/candidates/*.jsonl ──(review)──► promote --reviewed ──► data/personal/*.jsonl
+data/personal/*.jsonl ──► train_lora.py --data/--output ──► output/lora_*
+output/lora_* + AWQ MODEL ──► serve_with_lora.sh (LORA_MODULES)
+running server ──► personal_pipeline.py / test_client.py
+```
+
+Privacy: [`data/personal/**/*.jsonl`](../../.gitignore) is gitignored.
+
+## Alternatives (what might be better)
+
+| Option | Pros | Cons on this 24GB box |
+|---|---|---|
+| Keep **27B dense AWQ + LoRA** (current) | Strong base; LoRA fits serve | Dense train download ~65G; slow train; data quality dominates |
+| Smaller instruct (e.g. 4B–8B) | Faster iterate; easier train | Weaker base answers |
+| **Qwen3.6 MoE** | Potentially better quality/efficiency *if* experts fit | MoE serve/train memory patterns differ; may not fit AWQ+multi-LoRA as cleanly on 24GB without more tuning |
+| Better **data** (same stack) | Biggest lever for “feels like me” | Requires your time reviewing candidates |
+
+**Recommendation:** keep current stack; invest in **reviewed** personal JSONL before chasing MoE.
+
+## What you can do right now
+
+1. Read [operator cheatsheet](operator-cheatsheet.md) and run base serve smoke.
+2. Edit `data/personal/candidates/*.jsonl` (delete junk like `"C"`→`"C?"`).
+3. Re-promote, retrain, re-serve, run three messy prompts through `personal_pipeline.py`.
+4. Push `master` only when you are happy with the learning docs + want remote backup.
+
+## Accuracy to “our questions” today
+
+The **base AWQ model** already answers general tech questions well.  
+The **personal LoRAs** do not yet reliably make questions clearer or answers more “you” — that waits on data quality, not on missing vLLM features.
diff --git a/docs/guides/operator-cheatsheet.md b/docs/guides/operator-cheatsheet.md
new file mode 100644
index 0000000..701758d
--- /dev/null
+++ b/docs/guides/operator-cheatsheet.md
@@ -0,0 +1,120 @@
+# Operator cheatsheet
+
+Companion: [Architecture learning](architecture-learning.md)
+
+All commands assume repo root and an activated `.venv` (from setup).
+
+## 1) First-time setup
+
+**Windows:**
+
+```bat
+scripts\setup.cmd
+```
+
+**WSL / Linux:**
+
+```bash
+./scripts/setup.sh
+source .venv/bin/activate
+```
+
+Optional WSL toolchain (Triton/FlashInfer): `bash scripts/_install_usergcc.sh` then rely on [`scripts/wsl_runtime_env.sh`](../../scripts/wsl_runtime_env.sh).
+
+## 2) Serve base AWQ (no LoRA)
+
+Config: [`config/model.env`](../../config/model.env) (`MODEL`, `QUANTIZATION=awq`, …).
+
+```bash
+./scripts/start_server.sh
+# other terminal:
+python scripts/test_client.py
+```
+
+When: you want the stock 27B assistant. First run downloads AWQ weights.
+
+## 3) FAQ LoRA loop
+
+```bash
+# server up:
+python scripts/generate_training_data.py
+cp data/generated/raw_qa.jsonl data/train.jsonl
+python scripts/validate_dataset.py data/train.jsonl
+
+# STOP server (Ctrl+C), check VRAM:
+nvidia-smi
+
+# train dense base (TRAIN_MODEL in config; override with env as needed)
+MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
+  GRADIENT_ACCUMULATION_STEPS=8 GRADIENT_CHECKPOINTING=1 \
+  python scripts/train_lora.py
+
+./scripts/serve_with_lora.sh
+python scripts/test_client.py --model support-adapter
+```
+
+Train entry: [`scripts/train_lora.py`](../../scripts/train_lora.py#L50) (`--data` / `--output` optional).
+
+## 4) Personal pipeline (sharper → me-assistant)
+
+Sources: [`config/personal_sources.env`](../../config/personal_sources.env).
+
+**WSL transcript path** (Windows Cursor data):
+
+```bash
+export AGENT_TRANSCRIPTS_DIR=/mnt/c/Users/supre/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
+```
+
+```bash
+python scripts/extract_personal_candidates.py
+# EDIT data/personal/candidates/*.jsonl  ← required for quality
+
+python scripts/promote_personal_data.py --reviewed
+python scripts/validate_dataset.py data/personal/question_sharp.jsonl
+python scripts/validate_dataset.py data/personal/me_assistant.jsonl
+
+# STOP server; train each adapter
+MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
+  GRADIENT_ACCUMULATION_STEPS=8 GRADIENT_CHECKPOINTING=1 \
+  python scripts/train_lora.py \
+    --data data/personal/question_sharp.jsonl \
+    --output output/lora_question_sharper
+
+MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \
+  GRADIENT_ACCUMULATION_STEPS=8 GRADIENT_CHECKPOINTING=1 \
+  python scripts/train_lora.py \
+    --data data/personal/me_assistant.jsonl \
+    --output output/lora_me_assistant
+
+LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" \
+  ./scripts/serve_with_lora.sh
+
+python scripts/personal_pipeline.py "okey so like why train OOM on 24gb?"
+```
+
+Promote gate: [`scripts/promote_personal_data.py`](../../scripts/promote_personal_data.py#L43-L49).  
+Pipeline: [`scripts/personal_pipeline.py`](../../scripts/personal_pipeline.py#L22-L60).  
+Multi-LoRA: [`scripts/serve_with_lora.sh`](../../scripts/serve_with_lora.sh#L2), [`#L40-L72`](../../scripts/serve_with_lora.sh#L40-L72).
+
+## 5) Always before training
+
+1. Stop vLLM.  
+2. `nvidia-smi` → nearly free VRAM.  
+3. Then run `train_lora.py`.
+
+## Train knobs (env)
+
+Read by [`scripts/train_lora.py`](../../scripts/train_lora.py):
+
+- `MAX_SEQ_LENGTH`, `BATCH_SIZE`, `NUM_EPOCHS`
+- `GRADIENT_ACCUMULATION_STEPS` (default 4)
+- `GRADIENT_CHECKPOINTING` (default on)
+- `TRAIN_MODEL` / `TRAIN_DATA` / `TRAIN_OUTPUT` (or CLI `--data` / `--output`)
+
+Do **not** assume every `TRAIN_*` key in `config/model.env` is wired — prefer the env names above.
+
+## Privacy / push checklist
+
+- Do not commit HF tokens.
+- `data/personal/**/*.jsonl` is gitignored — keep it that way.
+- Push when docs + code on `master` are what you want remote; personal datasets stay local.
