# Design: Personal tech pipeline (question-sharper → me-assistant)

**Date:** 2026-08-08  
**Status:** approved  
**Depends on:** [2026-08-08-qwen36-27b-lora-train-design.md](2026-08-08-qwen36-27b-lora-train-design.md),
[2026-08-08-qwen36-27b-awq-serve-design.md](2026-08-08-qwen36-27b-awq-serve-design.md)

## Goal

Build a **personal, trackable tech pipeline** on the existing Qwen3.6-27B AWQ +
QLoRA stack:

1. **Question sharper** — messy / vague tech thought → clear, focused question  
2. **Me-assistant** — clear tech question → answer in the user’s preferred style  
3. Chain them automatically (pipeline), with optional run logging for learning

Domain for v1: **tech / coding / this ML setup** (not general life advice).

## Decisions (locked)

| Item | Choice |
|---|---|
| Packaging | Pipeline of **two adapters** (not one multi-task model) |
| Domain | Tech / coding / ML setup |
| Data source | Mine **Cursor agent transcripts** + **repo markdown** |
| Bootstrap | Extract candidates → human review → train JSONL |
| Train base | Dense `Qwen/Qwen3.6-27B` QLoRA (same as current loop) |
| Serve base | Existing AWQ `MODEL` with both LoRA modules loaded |
| Privacy | Personal JSONL stays local; gitignore private paths |

## Architecture

```
messy thought
    → LoRA: question-sharper
    → clear tech question
    → LoRA: me-assistant
    → answer in user’s style
    → optional log: output/personal_runs.jsonl
```

**Components**

1. **Extract** — scripts turn transcripts + selected markdown into candidate pairs  
2. **Review** — human keep / edit / reject; promote into train files  
3. **Train** — two QLoRA adapters on dense 27B  
4. **Serve** — one AWQ vLLM process registering both modules  
5. **Pipeline client** — one command that chains sharper → assistant and prints both steps  

## Data

Same schema as the existing FAQ loop: one JSON object per line with
`instruction` and `response`.

| File | `instruction` | `response` |
|---|---|---|
| `data/personal/question_sharp.jsonl` | Messy / vague tech thought | Clear, focused question |
| `data/personal/me_assistant.jsonl` | Clear tech question | Answer in preferred “me” style |

**Staging**

- Raw candidates: `data/personal/candidates/`  
- Promote to the train JSONLs only after review  
- First useful train target: **~50–100 pairs per adapter** (far more than the FAQ smoke test)

**Sources**

- Cursor agent transcripts for this project (user turns + endorsed assistant turns)  
- Repo markdown (`docs/`, design notes, and other paths listed in config)

**Extraction rules (v1)**

- **Sharper:** user messages that look messy / multi-thought → one sharpened question (draft helper allowed; human edits)  
- **Me-assistant:** clear questions paired with answers the user accepts as “how I want this answered”  
- No auto-promote without review  

**Privacy**

- Do not commit personal train/candidate JSONL if it contains private chat text  
- Add `data/personal/` (or at least candidates + train JSONL) to `.gitignore` unless the user explicitly opts into versioning sanitized data  

## Train

- Reuse `scripts/train_lora.py` with a small extension for train data path + output dir  
- Outputs:
  - `output/lora_question_sharper/`
  - `output/lora_me_assistant/`
- Same operational rule: stop the vLLM server before training to free VRAM  
- Train each adapter separately from its JSONL  

## Serve

- Extend `scripts/serve_with_lora.sh` (or a sibling) to register both modules, e.g.:
  - `question-sharper=<path>`
  - `me-assistant=<path>`
- Keep one AWQ base process; max LoRA rank unchanged (16) unless training changes rank  

## Pipeline client

New script (e.g. `scripts/personal_pipeline.py`):

1. Call chat API with `model=question-sharper` and the raw user text  
2. Abort with a clear error if the sharpened question is empty  
3. Call chat API with `model=me-assistant` and the sharpened question  
4. Print **both** steps (sharpened question + final answer)  
5. Optionally append a JSON line to `output/personal_runs.jsonl`:
   - timestamp, raw input, sharpened question, final answer  

Missing adapters or server-down errors should fail clearly (same spirit as
`test_client.py`).

## Out of scope (v1)

- Web search / tool use for factual grounding  
- Non-tech domains  
- Auto-train without human review  
- Guaranteeing correctness beyond data quality  
- Single multi-task adapter with `[SHARPEN]` / `[ANSWER]` tags  

## Success criteria

- Extract script produces candidates from transcripts + configured markdown  
- After review, both train JSONLs pass `validate_dataset.py`  
- Both adapters train and load together on one server  
- Pipeline turns one messy tech thought into a clearer question + a useful tech answer  
- Runs can be logged for later improvement  

**Smoke test:** three messy prompts; sharpened questions are clearer than the
raw input; answers feel closer to the user’s preferred style than base AWQ alone.

## Risks / notes

- Thin data → weak personalization; volume and review matter more than epochs  
- Two adapters mean two train runs and slightly more VRAM for LoRA slots at serve time  
- Transcript mining will be noisy; review is mandatory for quality  
- Factual errors still possible; treat this as style + focus, not a truth oracle  
