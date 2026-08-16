import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "personal"
    / "qwen_dataset_toolkit"
    / "triage_review_queue.py"
)
SPEC = importlib.util.spec_from_file_location("triage_review_queue", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(user, assistant, *, record_id="x", fact_check=False):
    return {
        "id": record_id,
        "fact_check_required": fact_check,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def bucket_for(item, review=None):
    buckets = MODULE.classify_records(
        [item], {item["id"]: review} if review else {}
    )
    return next(name for name, values in buckets.items() if values)


def test_context_only_confirmation_requires_context_even_if_approved():
    item = record("looks correct", "Proceeding.")
    assert bucket_for(item, {"decision": "approve", "notes": ""}) == "needs_context"


def test_voice_to_text_option_selections_require_context():
    for user in (
        "yes approvahc 1 accept",
        "okey i think we will choose the A variant",
        "yes good with the pplan C",
        "Aproach 3",
    ):
        assert bucket_for(record(user, "Proceeding.")) == "needs_context"


def test_short_references_to_missing_plan_or_bug_require_context():
    for user in (
        "what we haven't done based on our audit?",
        "do we have anything else we haven't done for the bug i have mentioned",
        "what the next specs we need to create by the pur plan?",
    ):
        assert bucket_for(record(user, "Status.")) == "needs_context"


def test_action_request_that_only_asks_for_code_needs_edit():
    item = record(
        "Fix the mobile animation.",
        "Please share the current CSS and JavaScript.",
    )
    assert bucket_for(item) == "needs_edit"


def test_unsupported_repository_state_needs_edit():
    item = record(
        "The cursor is missing.",
        "The CustomCursor component is imported in App.tsx but not rendered.",
    )
    assert bucket_for(item, {"decision": "approve", "notes": ""}) == "needs_edit"


def test_explicit_approval_is_exported_when_no_policy_blocks_it():
    item = record("Explain this log.", "The log shows a failed request.")
    assert bucket_for(item, {"decision": "approve", "notes": ""}) == "human_approved"


def test_fact_check_stays_manual_despite_approval():
    item = record("Does this board have a GUI?", "Yes.", fact_check=True)
    assert bucket_for(item, {"decision": "approve", "notes": ""}) == "manual_review"


def test_merge_reviews_preserves_main_and_accepts_remaining_updates(tmp_path):
    main = tmp_path / "main.md"
    remaining = tmp_path / "remaining.md"
    main.write_text(
        "## 1. a\n\n- Human decision: [x] approve  [ ] edit  [ ] reject\n"
        "\n### Reviewer notes\n\nMain note.\n",
        encoding="utf-8",
    )
    remaining.write_text(
        "## 1. b\n\n- Human decision: [ ] approve  [x] edit  [ ] reject\n"
        "\n### Reviewer notes\n\nRemaining note.\n",
        encoding="utf-8",
    )
    assert MODULE.merge_reviews([main, remaining]) == {
        "a": {"decision": "approve", "notes": "Main note."},
        "b": {"decision": "edit", "notes": "Remaining note."},
    }
