import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "personal"
    / "qwen_dataset_toolkit"
    / "build_review_report.py"
)
SPEC = importlib.util.spec_from_file_location("build_review_report", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_render_report_escapes_private_content_and_includes_decision():
    report = MODULE.render_report(
        [
            {
                "id": "prep_1",
                "category": "technical-explanation",
                "fact_check_required": True,
                "teacher": {"decision": "rewrite", "notes": "Check <hardware>."},
                "messages": [
                    {"role": "user", "content": "Is A < B?"},
                    {"role": "assistant", "content": "Yes & verify."},
                ],
            }
        ]
    )
    assert "Total pending examples: 1" in report
    assert "[ ] approve  [ ] edit  [ ] reject" in report
    assert "Is A &lt; B?" in report
    assert "Yes &amp; verify." in report


def test_existing_review_decision_and_notes_are_preserved():
    old = """## 1. prep_1

- Human decision: [ x approve  [ ] edit  [ ] reject

### Reviewer notes

Inspect the repository first.
_Add corrections or the replacement response here._
"""
    reviews = MODULE.parse_existing_reviews(old)
    assert reviews == {
        "prep_1": {"decision": "approve", "notes": "Inspect the repository first."}
    }
    rendered = MODULE.render_report(
        [
            {
                "id": "prep_1",
                "category": "technical-explanation",
                "messages": [
                    {"role": "user", "content": "Fix it."},
                    {"role": "assistant", "content": "Working."},
                ],
            }
        ],
        reviews,
    )
    assert "[x] approve  [ ] edit  [ ] reject" in rendered
    assert "Inspect the repository first." in rendered
    assert MODULE.REVIEW_PLACEHOLDER not in rendered
