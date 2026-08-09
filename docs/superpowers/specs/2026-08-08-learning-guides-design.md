# Design: Learning guides (architecture + operator cheatsheet)

**Date:** 2026-08-08  
**Status:** approved  
**Depends on:** existing AWQ serve, LoRA train, and personal pipeline work on `master`

## Goal

Before pushing the ~23 local commits ahead of `origin/master`, produce two
**easy-to-open learning guides** that explain what was built, why, how to run
it, limits/responsibilities, and dependencies — with links to real `path:line`
code anchors.

## Decisions (locked)

| Item | Choice |
|---|---|
| Packaging | **Two files** (Approach 3) |
| Location | `docs/guides/` (easy path, not buried under `docs/superpowers/specs/`) |
| Architecture / learning | `docs/guides/architecture-learning.md` |
| Operator commands | `docs/guides/operator-cheatsheet.md` |
| README | Add a short “Guides” section linking both |
| Code citations | Use repo-relative markdown links to `file` with line anchors where useful (Cursor/GitHub-style `#Lstart-Lend`) |
| Out of scope | Pushing remotes, retraining adapters, rewriting personal datasets |

## File layout

```
docs/guides/architecture-learning.md
docs/guides/operator-cheatsheet.md
README.md                          # MODIFY — Guides links only
```

This design note stays at:

`docs/superpowers/specs/2026-08-08-learning-guides-design.md`

## architecture-learning.md contents

1. **What we prepared to achieve** — 27B AWQ on 24GB WSL; QLoRA train path; personal sharper → me-assistant pipeline  
2. **What we achieved** — serve works; FAQ LoRA smoke; personal extract/promote/train/serve/pipeline plumbing  
3. **Honest quality** — personal adapters = infrastructure OK; data quality not yet “personal”; smoke ≠ success criteria  
4. **Responsibilities & approximate %**
   - **vLLM** — inference runtime (load AWQ, KV cache, OpenAI API, LoRA slots)
   - **Qwen3.6-27B (base)** — ~99%+ of knowledge / reasoning / style baseline
   - **LoRA adapters** — small trainable overlay (~0.3% params in our train logs); shifts style/task, does not replace the base
5. **Script map** — table of important scripts with `path:line` anchors  
6. **Dependency map** — `config/model.env` → serve/train; data JSONL → adapters → `serve_with_lora` / `personal_pipeline`  
7. **Alternatives** — smaller dense instruct; MoE Qwen3.6 (tradeoffs for 24GB); better data vs more epochs  
8. **What to do next** — review candidates, re-promote, retrain; then push when ready  

## operator-cheatsheet.md contents

Command-first scenarios:

| Scenario | Entry commands | Notes |
|---|---|---|
| Setup | `scripts/setup.cmd` / `./scripts/setup.sh` | Then activate `.venv` |
| Serve base AWQ | `./scripts/start_server.sh` + `python scripts/test_client.py` | Downloads AWQ on first run |
| FAQ LoRA loop | generate → validate → stop server → `train_lora.py` → `serve_with_lora.sh` | Dense train base via `TRAIN_MODEL` |
| Personal pipeline | extract → edit → `promote --reviewed` → train ×2 → `LORA_MODULES=...` → `personal_pipeline.py` | Human review required |
| Free VRAM for train | stop vLLM; `nvidia-smi` | Non-negotiable on 24GB |

Also document:

- WSL transcript path: `AGENT_TRANSCRIPTS_DIR=/mnt/c/Users/.../agent-transcripts`
- Train env knobs: `MAX_SEQ_LENGTH`, `BATCH_SIZE`, `NUM_EPOCHS`, `GRADIENT_ACCUMULATION_STEPS`, `GRADIENT_CHECKPOINTING`
- Privacy: do not commit tokens or `data/personal/**/*.jsonl`

## Success criteria

- Both guides exist under `docs/guides/`
- README links both
- Architecture guide answers vLLM vs LoRA vs base % honestly
- Cheatsheet is copy-paste usable for the four scenarios above
- Citations point at real files/lines in this repo (verified during implementation)

## Out of scope

- Remote `git push` / PR creation (operator chooses later)
- Auto-cleaning personal candidate quality
- Changing default serve/train models in this doc-only work
