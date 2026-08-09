# Task 7 Report: README + end-to-end operator path

## Status

**DONE**

## Commits

| SHA | Subject |
|-----|---------|
| `578bcdf` | docs: document personal tech LoRA pipeline |

## Changes

- Added `## Personal tech pipeline (question-sharper → me-assistant)` to `README.md` after the FAQ LoRA section: extract → review → promote → validate → train (both adapters) → multi-LoRA serve → `personal_pipeline.py`.
- Linked `config/personal_sources.env` and `docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md` in the new section and Troubleshooting docs list.
- Documented WSL `AGENT_TRANSCRIPTS_DIR=/mnt/c/...` override in README and as comments in `config/personal_sources.env`.

## Pytest summary

```
35 passed in 1.11s  (WSL .venv, python -m pytest -v)
```

## Concerns

- None blocking. `config/model.env` has unrelated unstaged changes (not part of this task).
- Operators must still stop vLLM before training and set the Windows transcript path when extracting from WSL.

## Branch

`feat/qwen36-27b-lora-train`
