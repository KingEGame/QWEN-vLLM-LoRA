# Task 4 Report: Train script `--data` / `--output`

**Status:** complete  
**Branch:** feat/qwen36-27b-lora-train  
**Commit:** `1874a6d` — feat: train_lora accepts --data and --output paths

## Changes

- Added `argparse` to `scripts/train_lora.py` with `--data` and `--output` flags.
- Resolved paths via CLI → `TRAIN_DATA` / `TRAIN_OUTPUT` env → defaults (`data/train.jsonl`, `output/lora_adapter`).
- Replaced all `main()` uses of `TRAIN_DATA_PATH` / `OUTPUT_DIR` with `train_data` / `output_dir`.
- Kept module-level `TRAIN_DATA_PATH` and `OUTPUT_DIR` constants for backward-compatible imports.

## Verification

- `python scripts/train_lora.py --help` → shows `--data` and `--output` options.
- `python scripts/train_lora.py --data /no/such.jsonl --output /tmp/x` → `ERROR: ... not found`, exit 1 (no GPU load).
- `python -m py_compile scripts/train_lora.py` → OK.

## Tests

- Manual CLI checks only (no GPU training run).

## Concerns

- None for this task scope. File also contains prior Task 3 27B QLoRA hardening in the same working diff if not yet committed separately on branch.

---

## Review fix (2026-08-08)

**Status:** complete (scoped to CLI only)  
**Fix commit:** `e0ac82f` — fix: scope Task 4 to CLI --data/--output only  
**Prior commit:** `1874a6d` (bundled unrelated 27B hardening; superseded by fix)

### Fix actions

- Restored `scripts/train_lora.py` non-CLI behavior from parent `e516098`.
- Re-applied only Task 4: `argparse` `--data`/`--output`, env `TRAIN_DATA`/`TRAIN_OUTPUT`, `train_data`/`output_dir` in `main()`.
- Removed bundled changes: `_cfg_get`, `_resolve_int`, `device_map={"":0}`, resource fraction, dataloader workers, 27B default knobs, docstring rewrite.

### Verification (post-fix)

```bash
python scripts/train_lora.py --help
# shows --data and --output options

python scripts/train_lora.py --data /no/such.jsonl --output /tmp/x; echo EXIT:$?
# ERROR: \no\such.jsonl not found. Create it first (see README).
# EXIT:1

python -m py_compile scripts/train_lora.py
# OK

git diff e516098 -- scripts/train_lora.py
# CLI-only diff (+argparse, path resolution, train_data/output_dir replacements)
```
