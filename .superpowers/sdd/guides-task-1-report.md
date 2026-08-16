# Task 1 Report: architecture-learning.md

## Status

**Complete.** Created and committed `docs/guides/architecture-learning.md` per brief.

## Commit

- **Hash:** `df724b3`
- **Message:** `docs: add architecture learning guide`
- **Files committed:** `docs/guides/architecture-learning.md` only (115 lines)
- **Branch:** `master` (ahead of `origin/master` by 27 commits)

## Anchor verification (`rg -n`)

All `#L` anchors in the guide match live code; no adjustments required.

| Anchor | Expected | Verified line |
|---|---|---|
| `scripts/start_server.sh#L40` | `vllm serve` | 40 |
| `scripts/serve_with_lora.sh#L40-L72` | `LORA_MODULES` / `--lora-modules` | 40–72 |
| `scripts/train_lora.py#L36-L37` | `LORA_RANK` / `LORA_ALPHA` | 36–37 |
| `scripts/train_lora.py#L117` | `device_map={"": 0}` | 117 |
| `scripts/train_lora.py#L144` | `print_trainable_parameters` | 144 |
| `scripts/lib/personal_pipeline.py#L7-L27` | `def run_pipeline` … return | 7–27 |

Supporting checks:

- `rg -n "vllm serve" scripts/start_server.sh scripts/serve_with_lora.sh` → 40, 66
- `rg -n "def run_pipeline|device_map|LORA_RANK|print_trainable|--reviewed" scripts/` → all expected hits

## Content summary

Guide covers: goals, achieved status table, quality assessment, vLLM/base/LoRA responsibility split, script map, dependency flow, alternatives table, next steps, and accuracy note. Links to three design specs and operator cheatsheet (placeholder until Task 2).

## Excluded from commit (as instructed)

- `.superpowers/sdd/*` scratch files
- `scripts/_*.sh` local ops helpers
- `.superpowers/sdd/progress.md` (modified, not staged)

## Notes

- `docs/guides/` directory created implicitly by file write.
- Operator cheatsheet link points to `operator-cheatsheet.md` (to be added in a later task).
