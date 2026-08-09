# Task 3 Report: Extract + promote CLIs

## Status

**DONE**

## Commits

| SHA | Subject |
|-----|---------|
| `e516098` | feat: extract and promote personal train datasets |

## Files created

| File | Purpose |
|------|---------|
| `scripts/extract_personal_candidates.py` | Mine transcripts + markdown → candidate JSONL |
| `scripts/promote_personal_data.py` | Promote reviewed candidates (instruction/response only) |

## Implementation summary

### `extract_personal_candidates.py`

- Loads `config/personal_sources.env` via `load_env_file`; env vars override config.
- Expands `AGENT_TRANSCRIPTS_DIR` with `os.path.expanduser` (WSL-friendly for `~/.cursor/...`).
- Transcripts: `sharpen_candidates_from_texts` + `iter_transcript_qa_pairs` per `*.jsonl`.
- Markdown: `pairs_from_markdown` for each glob in `MARKDOWN_GLOBS`.
- Writes `data/personal/candidates/question_sharp.jsonl` and `me_assistant.jsonl`.

### `promote_personal_data.py`

- Requires `--reviewed`; exits 1 with clear error otherwise.
- Strips `source`/`kind` metadata; keeps `instruction`/`response` only.
- Validates promoted files with `validate_dataset_file` before reporting success.

## Smoke test evidence

Run via WSL with Windows transcripts path (default `~` resolves to WSL home, not Windows `.cursor`):

```bash
AGENT_TRANSCRIPTS_DIR=/mnt/c/Users/supre/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts \
  .venv/bin/python scripts/extract_personal_candidates.py
# Wrote 71 sharpen candidates, 174 me_assistant candidates

.venv/bin/python scripts/promote_personal_data.py
# ERROR: refusing to promote without --reviewed  (exit 1)

.venv/bin/python scripts/promote_personal_data.py --reviewed
# Promoted 71 rows → question_sharp.jsonl
# Promoted 174 rows → me_assistant.jsonl

.venv/bin/python scripts/validate_dataset.py data/personal/question_sharp.jsonl
# OK: valid (71 training examples)

.venv/bin/python scripts/validate_dataset.py data/personal/me_assistant.jsonl
# OK: valid (174 training examples)
```

Without transcripts override (default config only): 0 sharpen / 124 me_assistant from markdown; promote succeeds but `validate_dataset.py` fails on empty `question_sharp.jsonl` — expected when no transcript dir.

## Concerns

- **Transcript path**: `~/.cursor/...` in config resolves to WSL `$HOME`, not Windows `%USERPROFILE%`. Set `AGENT_TRANSCRIPTS_DIR` to the `/mnt/c/...` path (or export before extract) when running from WSL against Windows Cursor data.
- **Empty sharpen file**: Brief promote code allows zero-row promote; `validate_dataset.py` correctly rejects empty train files afterward.

## Branch

`feat/qwen36-27b-lora-train`
