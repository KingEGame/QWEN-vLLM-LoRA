# System architecture learning

Companions: [Operator cheatsheet](operator-cheatsheet.md) and
[LoRA, QLoRA, and AWQ learning](lora-qlora-learning.md).

Pre-push learning note for this repo’s Qwen3.6-27B + vLLM + LoRA work (local
`master` may be ahead of `origin/master`).

## What we prepared to achieve

1. Serve **Qwen3.6-27B** on a **24GB** laptop GPU via **AWQ** + vLLM (WSL2).
2. Train **LoRA** adapters with **QLoRA** on dense `Qwen/Qwen3.6-27B` (cannot train on AWQ).
3. Build a **personal tech pipeline**: messy thought → `question-sharper` → clear question → `me-assistant` → answer.

Design trail:

- [AWQ serve](../superpowers/specs/2026-08-08-qwen36-27b-awq-serve-design.md)
- [LoRA train](../superpowers/specs/2026-08-08-qwen36-27b-lora-train-design.md)
- [Personal pipeline](../superpowers/specs/2026-08-08-personal-tech-pipeline-design.md)

## What we achieved

| Track | Status |
|---|---|
| AWQ serve on 24GB | Working (tuned `MAX_MODEL_LEN` / `MAX_NUM_SEQS`) |
| FAQ LoRA smoke | Working end-to-end on tiny `data/train.jsonl` |
| Personal extract / promote / train / multi-LoRA serve / pipeline client | **Plumbing works** |
| Personalization quality (“understands me”) | **Not yet** — candidates need real human edit; smoke used noisy / thin data |

## Are the results good?

**Serve / train infrastructure: yes.**  
**Personal adapters as “your model”: not yet.**

Evidence from the personal smoke: the pipeline returns Raw / Sharpened / Answer and logs to `output/personal_runs.jsonl`, but sharpened text was often long lecture-style output rather than a tighter question — a data + review problem, not a missing script.

LoRA trainable share observed in training logs was on the order of **~0.3%** of parameters (`print_trainable_parameters` in [`scripts/train_lora.py`](../../scripts/train_lora.py#L144)).

## Who does what (responsibilities)

```
User prompt
   │
   ▼
vLLM (runtime)  ── loads AWQ base, KV cache, OpenAI HTTP API, optional LoRA slots
   │
   ├── Qwen3.6-27B base weights  ≈ ~99%+ of knowledge / reasoning / default style
   └── LoRA adapter(s)           ≈ small delta (rank 16) steering task/style
```

| Piece | Responsibility | Rough “% of behavior” |
|---|---|---|
| **vLLM** | Serve fast, manage VRAM, expose `/v1/chat/completions`, attach LoRAs | 0% of *language knowledge*; 100% of *how inference is run* |
| **Qwen3.6-27B (AWQ at serve / dense at train)** | Almost all understanding and generation skill | **~99%+** |
| **LoRA** | Thin task/style overlay learned from your JSONL | **~0.3% params**; influence can be large *on the trained task* if data is good, tiny if data is bad |

Code anchors:

- Base serve: [`scripts/start_server.sh`](../../scripts/start_server.sh#L40) (`vllm serve`)
- LoRA serve: [`scripts/serve_with_lora.sh`](../../scripts/serve_with_lora.sh#L40-L72) (`LORA_MODULES` / `--lora-modules`)
- Train rank / load: [`scripts/train_lora.py`](../../scripts/train_lora.py#L36-L37), [`#L117`](../../scripts/train_lora.py#L117) (`device_map={"": 0}`)
- Pipeline chain: [`scripts/lib/personal_pipeline.py`](../../scripts/lib/personal_pipeline.py#L7-L27)

## Script map (what / why)

| Script | Why it exists |
|---|---|
| [`scripts/setup.sh`](../../scripts/setup.sh) | Create `.venv`, install deps, CUDA check |
| [`scripts/wsl_runtime_env.sh`](../../scripts/wsl_runtime_env.sh) | User-space GCC/CUDA for Triton on WSL |
| [`scripts/start_server.sh`](../../scripts/start_server.sh) | AWQ base server from [`config/model.env`](../../config/model.env) |
| [`scripts/test_client.py`](../../scripts/test_client.py) | One-shot OpenAI client smoke |
| [`scripts/generate_training_data.py`](../../scripts/generate_training_data.py) | FAQ → draft Q&A via running server |
| [`scripts/validate_dataset.py`](../../scripts/validate_dataset.py) | JSONL `instruction`/`response` gate |
| [`scripts/train_lora.py`](../../scripts/train_lora.py) | QLoRA train; `--data` / `--output` |
| [`scripts/serve_with_lora.sh`](../../scripts/serve_with_lora.sh) | AWQ + one or many LoRAs |
| [`scripts/extract_personal_candidates.py`](../../scripts/extract_personal_candidates.py) | Mine transcripts + markdown → candidates |
| [`scripts/promote_personal_data.py`](../../scripts/promote_personal_data.py) | Promote only with `--reviewed` |
| [`scripts/personal_pipeline.py`](../../scripts/personal_pipeline.py) | Sharper → assistant + optional log |

Helpers named `scripts/_*.sh` are **local ops** (prefetch, progress, SDD); prefer not to treat them as product API.

## Dependency map (depends on what)

```
config/model.env ──► start_server.sh / serve_with_lora.sh / test_client.py / train (TRAIN_MODEL)
config/personal_sources.env ──► extract_personal_candidates.py
data/personal/candidates/*.jsonl ──(review)──► promote --reviewed ──► data/personal/*.jsonl
data/personal/*.jsonl ──► train_lora.py --data/--output ──► output/lora_*
output/lora_* + AWQ MODEL ──► serve_with_lora.sh (LORA_MODULES)
running server ──► personal_pipeline.py / test_client.py
```

Privacy: [`data/personal/**/*.jsonl`](../../.gitignore) is gitignored.

## Alternatives (what might be better)

| Option | Pros | Cons on this 24GB box |
|---|---|---|
| Keep **27B dense AWQ + LoRA** (current) | Strong base; LoRA fits serve | Dense train download ~65G; slow train; data quality dominates |
| Smaller instruct (e.g. 4B–8B) | Faster iterate; easier train | Weaker base answers |
| **Qwen3.6 MoE** | Potentially better quality/efficiency *if* experts fit | MoE serve/train memory patterns differ; may not fit AWQ+multi-LoRA as cleanly on 24GB without more tuning |
| Better **data** (same stack) | Biggest lever for “feels like me” | Requires your time reviewing candidates |

**Recommendation:** keep current stack; invest in **reviewed** personal JSONL before chasing MoE.

## What you can do right now

1. Read [operator cheatsheet](operator-cheatsheet.md) and run base serve smoke.
2. Edit `data/personal/candidates/*.jsonl` (delete junk like `"C"`→`"C?"`).
3. Re-promote, retrain, re-serve, run three messy prompts through `personal_pipeline.py`.
4. Push `master` only when you are happy with the learning docs + want remote backup.

## Accuracy to “our questions” today

The **base AWQ model** already answers general tech questions well.  
The **personal LoRAs** do not yet reliably make questions clearer or answers more “you” — that waits on data quality, not on missing vLLM features.
