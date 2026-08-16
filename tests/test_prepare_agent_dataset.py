import json

from scripts.prepare_agent_dataset import build_dataset, extract_pair


def test_extract_pair_requires_explicit_human_approval():
    pending = {
        "review_status": "pending_human_review",
        "messages": [
            {"role": "user", "content": "Fix it"},
            {"role": "assistant", "content": "Fixed"},
        ],
    }
    assert extract_pair(pending) is None
    pending["review_status"] = "human_approved"
    assert extract_pair(pending) == {"instruction": "Fix it", "response": "Fixed"}


def test_build_dataset_deduplicates_approved_pairs(tmp_path):
    row = {
        "review_status": "human_approved",
        "messages": [
            {"role": "user", "content": "Explain"},
            {"role": "assistant", "content": "Answer"},
        ],
    }
    path = tmp_path / "approved.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    assert build_dataset([path]) == [{"instruction": "Explain", "response": "Answer"}]


def test_provisional_teacher_rows_require_flag_and_skip_fact_checks(tmp_path):
    row = {
        "id": "safe",
        "review_status": "pending_human_review",
        "fact_check_required": False,
        "messages": [
            {"role": "user", "content": "Inspect this"},
            {"role": "assistant", "content": "I inspected it"},
        ],
    }
    risky = {**row, "id": "risky", "fact_check_required": True}
    path = tmp_path / "teacher.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(risky) + "\n", encoding="utf-8")
    assert build_dataset([path]) == []
    assert build_dataset([path], allow_teacher_curated=True) == [
        {"instruction": "Inspect this", "response": "I inspected it"}
    ]
