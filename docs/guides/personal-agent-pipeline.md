# Persistent Personal-Agent Pipeline

This pipeline prepares a small local assistant that behaves like an engineering
agent rather than a chat-only model: it interprets noisy voice input, inspects
the workspace, performs authorized work, verifies edits, retains unfinished
goals, and reports concise outcomes.

## Architecture

1. **Dataset** — Cursor/chat imports are sanitized, teacher-curated, reviewed,
   and triaged. Only `human_approved` records enter training.
2. **Environment** — `config/agent.env` defines the student/teacher models,
   private state paths, runtime limits, and edge-export settings.
3. **Scripts** — workspace tools enforce containment and read-before-edit;
   task and memory stores persist locally under ignored `output/` paths.
4. **Model** — Qwen3-1.7B is the student; Qwen3-0.6B is an optional smaller
   fallback; the existing Qwen3.8-27B server remains the dataset teacher.
5. **Pipeline** — inspect → act → verify → update task/memory → concise final.

The model choice is intentionally conservative. Benchmark the quantized model
on Raspberry Pi 5 and Pixel 7. Treat Arduino UNO Q support as unverified until
its RAM, runtime, and accelerator path are measured; use the 0.6B fallback or a
remote/local-network inference endpoint if 1.7B is not viable.

## 1. Prepare the dataset

```bash
python data/personal/qwen_dataset_toolkit/triage_review_queue.py
python scripts/prepare_agent_dataset.py
python scripts/build_agent_eval.py
```

Training is blocked below `AGENT_MIN_TRAIN_EXAMPLES` (20 by default). For a
format-only smoke test, use `--allow-small`; do not treat that adapter as useful.

## 2. Check the environment

```bash
python scripts/check_agent_environment.py
python scripts/check_agent_environment.py --require-training-ready
```

The second command fails until the approved dataset, CUDA, and training
packages are ready.

## 3. Train and serve

Stop the 27B teacher server before training, then run:

```bash
bash scripts/_train_agent.sh
bash scripts/_serve_agent.sh
```

The runtime server exposes the LoRA as `personal-agent` and enables Qwen tool
calling. The base model can be used for a runtime smoke before training.

## 4. Run the persistent tool agent

Read-only/default:

```bash
python scripts/agent_pipeline.py run --base-only "check why the tests fail"
```

Permit scoped edits and verification commands:

```bash
python scripts/agent_pipeline.py run --allow-write --allow-command \
  "fix the failing test and verify the goal"
```

The command tools are constrained to search, tests/builds, and read-only Git
inspection. File replacement is disabled unless `--allow-write` is provided,
and a file must be read before replacement. An edit cannot finish without a
subsequent successful verification or an explicit blocker.

Check persistent state:

```bash
python scripts/agent_pipeline.py status
```

Resolve or block an unfinished task explicitly when necessary:

```bash
python scripts/agent_pipeline.py task TASK_ID --status complete --note "Verified and superseded"
```

Tasks, runtime memory, run logs, and backups remain under `output/agent/` and
are ignored by Git. The default sync provider is `local`. Notion or another
cloud task store requires a separate connector, credentials, and authorization;
the agent must never claim cloud synchronization without an actual result.

## 5. Export for Raspberry Pi or Android

Install/build llama.cpp, set `LLAMA_CPP_DIR`, then:

```bash
bash scripts/_export_agent_gguf.sh
```

This merges the adapter and creates a `Q4_K_M` GGUF. Benchmark latency, memory,
context length, and tool-call reliability on each target before deployment.
