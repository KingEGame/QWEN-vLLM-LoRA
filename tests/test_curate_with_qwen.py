import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "personal"
    / "qwen_dataset_toolkit"
    / "curate_with_qwen.py"
)
SPEC = importlib.util.spec_from_file_location("curate_with_qwen", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_local_endpoint_policy():
    assert MODULE.is_local_url("http://127.0.0.1:8000/v1/chat/completions")
    assert MODULE.is_local_url("http://localhost:11434/v1/chat/completions")
    assert not MODULE.is_local_url("https://example.com/v1/chat/completions")


def test_extract_exchange_supports_missing_original_answer():
    user, assistant = MODULE.extract_exchange(
        {"messages": [{"role": "user", "content": "Explain the design."}]}
    )
    assert user == "Explain the design."
    assert assistant is None


def test_teacher_cannot_approve_its_own_output():
    with pytest.raises(ValueError, match="invalid teacher_decision"):
        MODULE.validate_teacher_result(
            {
                "teacher_decision": "approved",
                "category": "technical-explanation",
                "assistant_response": "Answer",
                "fact_check_required": False,
                "notes": "Looks good",
            }
        )


def test_valid_teacher_result():
    result = MODULE.validate_teacher_result(
        {
            "teacher_decision": "rewrite",
            "category": "technical-explanation",
            "assistant_response": "  Complete answer.  ",
            "fact_check_required": False,
            "notes": "  Removed progress chatter.  ",
        }
    )
    assert result["assistant_response"] == "Complete answer."
    assert result["notes"] == "Removed progress chatter."


def test_reject_requires_no_answer_but_still_requires_notes():
    result = MODULE.validate_teacher_result(
        {
            "teacher_decision": "reject",
            "category": "failure",
            "assistant_response": "",
            "fact_check_required": False,
            "notes": "Contextless request.",
        }
    )
    assert result["teacher_decision"] == "reject"


def test_keep_may_return_empty_replacement():
    result = MODULE.validate_teacher_result(
        {
            "teacher_decision": "keep",
            "category": "technical-explanation",
            "assistant_response": "",
            "fact_check_required": False,
            "notes": "Original answer is suitable.",
        }
    )
    assert result["assistant_response"] == ""


def test_fact_check_flag_is_required():
    with pytest.raises(ValueError, match="fact_check_required"):
        MODULE.validate_teacher_result(
            {
                "teacher_decision": "rewrite",
                "category": "technical-explanation",
                "assistant_response": "Answer",
                "notes": "Rewritten.",
            }
        )


def test_prune_stale_records_keeps_only_current_candidates(tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text(
        "\n".join(
            json.dumps({"id": value}) for value in ("current", "stale")
        )
        + "\n",
        encoding="utf-8",
    )
    assert MODULE.prune_stale_records(output, {"current"}) == 1
    assert MODULE.read_jsonl(output) == [{"id": "current"}]
