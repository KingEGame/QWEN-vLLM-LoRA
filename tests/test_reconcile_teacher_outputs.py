import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "personal"
    / "qwen_dataset_toolkit"
    / "reconcile_teacher_outputs.py"
)
SPEC = importlib.util.spec_from_file_location("reconcile_teacher_outputs", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_reconcile_deduplicates_and_quarantines_conflicts():
    candidates = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    curated = [
        {"id": "a", "teacher": {"decision": "keep"}},
        {"id": "a", "teacher": {"decision": "rewrite"}},
        {"id": "b", "teacher": {"decision": "keep"}},
        {"id": "stale", "teacher": {"decision": "keep"}},
    ]
    rejected = [{"id": "b", "teacher": {"decision": "reject"}}]
    errors = [{"id": "c", "review_status": "teacher_error"}]

    curated_out, rejected_out, errors_out, stats = MODULE.reconcile(
        candidates, curated, rejected, errors
    )

    assert [record["id"] for record in curated_out] == ["a"]
    assert rejected_out == []
    assert {record["id"] for record in errors_out} == {"b", "c"}
    assert stats["duplicate_lines_removed"] == 1
    assert stats["stale_records_removed"] == 1
    assert stats["conflicts_quarantined"] == 1
