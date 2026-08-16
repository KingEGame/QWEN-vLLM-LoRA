import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "personal"
    / "qwen_dataset_toolkit"
    / "preprocess_training_bank.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_dataset_preprocess", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def encode(record):
    return json.dumps(record)


def test_extracts_request_and_removes_metadata():
    content = """<USER_REQUEST>
    okey explain the model
    </USER_REQUEST>
    <ADDITIONAL_METADATA>private metadata</ADDITIONAL_METADATA>"""
    assert MODULE.extract_user_text(content) == "okey explain the model"


def test_removes_dangling_request_wrapper():
    assert MODULE.extract_user_text("<USER_REQUEST>\nerror without closing tag") == (
        "error without closing tag"
    )


def test_redacts_home_email_token_and_private_ip():
    text = (
        r"Use C:\Users\KINGWAR\project, me@example.com, "
        "sk-1234567890abcdefghijklmnop and 192.168.1.20"
    )
    cleaned, flags = MODULE.redact_sensitive(text)
    assert "KINGWAR" not in cleaned
    assert "me@example.com" not in cleaned
    assert "sk-123" not in cleaned
    assert "192.168.1.20" not in cleaned
    assert set(flags) == {"email", "private_ip", "secret_token", "user_home_path"}


def test_prefers_last_non_noise_assistant_response():
    response, reasons = MODULE.choose_assistant_response(
        [
            "Created At: 2026-08-15\nCompleted At: 2026-08-15\nTool output",
            "I'll inspect the repository now.",
            "Use a Q4 model and keep the context at 1,024 tokens.",
        ]
    )
    assert response == "Use a Q4 model and keep the context at 1,024 tokens."
    assert "platform_or_tool_noise" in reasons
    assert "progress_only_response" in reasons


def test_filters_background_task_without_completed_timestamp():
    response, reasons = MODULE.choose_assistant_response(
        ["Created At: now\nTool is running as a background task\nTask Description: scan"]
    )
    assert response is None
    assert reasons == ["platform_or_tool_noise"]


def test_deduplicates_conversations_and_retains_unanswered_prompt():
    conversation = {
        "source_file": r"C:\Users\Someone\transcript.jsonl",
        "messages": [
            {
                "role": "user",
                "content": "<USER_REQUEST>how should this work?</USER_REQUEST>",
            },
            {
                "role": "assistant",
                "content": "Tool is running as a background task with task id: 123",
            },
        ],
    }
    candidates, rejections, stats = MODULE.prepare_records(
        [encode(conversation), encode(conversation)]
    )
    assert not rejections
    assert len(candidates) == 1
    assert candidates[0]["messages"] == [
        {"role": "user", "content": "how should this work?"}
    ]
    assert "needs_teacher_generation" in candidates[0]["preprocess"]["flags"]
    assert stats.duplicate_conversations == 1


def test_rejects_contextless_choice():
    conversation = {"messages": [{"role": "user", "content": "2"}]}
    candidates, rejections, stats = MODULE.prepare_records([encode(conversation)])
    assert candidates == []
    assert rejections[0]["reason"] == "context_dependent_or_trivial_user_message"
    assert stats.rejected_turns == 1


def test_rejects_short_approval_and_platform_event():
    for text, reason in (
        ("okey go ahead", "context_dependent_or_trivial_user_message"),
        ("accepted", "context_dependent_or_trivial_user_message"),
        ("The USER performed the following action: Show a file", "platform_user_event"),
    ):
        useful, actual_reason = MODULE.meaningful_user_text(text)
        assert not useful
        assert actual_reason == reason


def test_normalize_space_repairs_common_utf8_mojibake():
    assert MODULE.normalize_space("Inspector â†’ RAW_LOGS") == "Inspector → RAW_LOGS"


def test_rejects_oversized_user_and_removes_oversized_assistant():
    useful, reason = MODULE.meaningful_user_text("u" * (MODULE.MAX_USER_CHARS + 1))
    assert not useful
    assert reason == "oversized_user_message_needs_manual_split"

    conversation = {
        "messages": [
            {"role": "user", "content": "Explain this technical problem."},
            {"role": "assistant", "content": "a" * (MODULE.MAX_ASSISTANT_CHARS + 1)},
        ]
    }
    candidates, _, stats = MODULE.prepare_records([encode(conversation)])
    assert candidates[0]["messages"] == [
        {"role": "user", "content": "Explain this technical problem."}
    ]
    assert "needs_teacher_generation" in candidates[0]["preprocess"]["flags"]
    assert "oversized_assistant_response_removed" in (
        candidates[0]["preprocess"]["filtered_assistant_reasons"]
    )
    assert stats.candidates_without_answer == 1
