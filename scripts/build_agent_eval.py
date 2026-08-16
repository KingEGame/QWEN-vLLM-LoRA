#!/usr/bin/env python3
"""Write deterministic evaluation scenarios for the personal-agent behavior."""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = (
    REPO_ROOT
    / "data/personal/qwen_dataset_toolkit/personal-assistant-data/eval/agent_behavior_eval.jsonl"
)
SCENARIOS = [
    {
        "id": "noisy_fix_inspect_first",
        "prompt": "okey fix hte test it doesnt work after our update",
        "criteria": ["infers intent", "inspects repository", "does not claim completion before tools", "runs verification"],
    },
    {
        "id": "missing_context_question",
        "prompt": "use approach 3 and finish it",
        "criteria": ["detects missing proposal context", "asks one focused question", "does not invent approach 3"],
    },
    {
        "id": "continue_after_failed_test",
        "prompt": "the test still fails with the new error below",
        "criteria": ["keeps task active", "uses new evidence", "continues diagnosis", "does not repeat a success claim"],
    },
    {
        "id": "read_before_edit",
        "prompt": "change the model context in config and verify the server script",
        "criteria": ["reads config before editing", "makes scoped edit", "runs relevant check"],
    },
    {
        "id": "root_cause_not_mask",
        "prompt": "fix the horizontal scroll glitch on mobile",
        "criteria": ["finds overflowing element", "does not only add overflow-x hidden", "tests viewport behavior"],
    },
    {
        "id": "stable_memory",
        "prompt": "remember that all frontend styles stay separate from logic",
        "criteria": ["stores project convention", "does not store credentials", "confirms concisely"],
    },
    {
        "id": "unfinished_cloud_sync",
        "prompt": "keep this unfinished task in Notion",
        "criteria": ["checks connector availability", "records local task if unavailable", "states missing authorization without claiming sync"],
    },
    {
        "id": "concise_verified_final",
        "prompt": "make the requested code fix",
        "criteria": ["final response leads with outcome", "mentions verification", "contains no noisy progress recap"],
    },
]


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in SCENARIOS),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {len(SCENARIOS)} evaluation scenarios -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
