# Task final fix report — personal tech pipeline review

**Branch:** feat/qwen36-27b-lora-train  
**Date:** 2026-08-08

## Status

All merge-blocker findings addressed. Full pytest suite passes (44 tests).

## Commits

1. `fix(personal-pipeline): address final review merge blockers` — train env knobs, promote/extract/sharpen fixes, README, tests, ledger
2. (single commit if squashed locally)

## What was fixed

| Finding | Fix |
|---|---|
| README claimed train knobs in `config/model.env` | README documents env vars; `train_lora.py` reads `GRADIENT_ACCUMULATION_STEPS` and `GRADIENT_CHECKPOINTING` from env |
| Zero-row promote | `promote_personal_data.py` exits 1 if either output has 0 rows |
| Empty `AGENT_TRANSCRIPTS_DIR` → cwd rglob | `_transcripts_root()` returns None for empty; warn and skip |
| Ledger over-claim on Task 8 | `progress.md` notes plumbing-only smoke; quality criteria not met |
| Short/junk sharpen rows | Always skip `len(t.strip()) < 20`; skip trivial capitalize+`?` single tokens |
| Optional JSON/KeyError guards | Clear ERROR lines in `_strip_meta` |

## Tests

- **Focused:** 15 passed (`test_promote_personal_data`, `test_personal_extract`, `test_extract_personal_candidates`)
- **Full suite:** 44 passed via WSL `.venv/bin/python -m pytest -v`

## Not done (per scope)

- No GPU retrain or smoke re-run
- No full extract pipeline rewrite
