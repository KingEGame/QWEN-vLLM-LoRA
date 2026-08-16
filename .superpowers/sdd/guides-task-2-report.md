# Task 2 Report: operator-cheatsheet.md

**Status:** Complete

**Commit:** `docs: add operator cheatsheet` — `docs/guides/operator-cheatsheet.md` only.

## Deliverable

Created `docs/guides/operator-cheatsheet.md` per design section 3 and README command patterns. Sections: setup, base AWQ serve, FAQ LoRA loop, personal pipeline, pre-train checklist, train env knobs, privacy/push checklist. Links to companion `architecture-learning.md`.

## Anchor verification (rg)

| Link | Target | Result |
|------|--------|--------|
| `train_lora.py#L50` | `def main()` + `--data`/`--output` args | OK |
| `promote_personal_data.py#L43-L49` | `--reviewed` gate | OK |
| `personal_pipeline.py#L22-L60` | `main()` through logging | OK |
| `serve_with_lora.sh#L2` | Multi-LoRA usage comment | OK |
| `serve_with_lora.sh#L40-L72` | `LORA_MODULES` parse + vLLM launch | OK |

No anchor adjustments required.

## Notes

- All relative paths use `../../` from `docs/guides/` to repo root scripts/config.
- WSL transcript path uses user-specific Cursor project path as in brief.
