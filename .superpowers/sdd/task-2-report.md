# Task 2 Report: Personal extract library (TDD)

## Status

**DONE**

## Commits

| SHA | Subject |
|-----|---------|
| `a96be0e` | feat: extract personal LoRA candidates from transcripts and markdown |

## TDD evidence

### RED (Step 2)

```text
ModuleNotFoundError: No module named 'scripts.lib.personal_extract'
ERROR tests/test_personal_extract.py
```

Ran via WSL: `.venv/bin/python -m pytest tests/test_personal_extract.py -v`

### GREEN (Step 4)

```text
tests/test_personal_extract.py::test_extract_user_query_from_wrapper PASSED
tests/test_personal_extract.py::test_extract_user_query_plain_fallback PASSED
tests/test_personal_extract.py::test_draft_sharpen_collapses_whitespace PASSED
tests/test_personal_extract.py::test_iter_transcript_user_texts PASSED
tests/test_personal_extract.py::test_iter_transcript_qa_pairs PASSED
tests/test_personal_extract.py::test_pairs_from_markdown_heading_chunks PASSED

6 passed in 0.10s
```

## Files created

| File | Purpose |
|------|---------|
| `tests/test_personal_extract.py` | Six unit tests from brief (verbatim) |
| `scripts/lib/personal_extract.py` | Extract library per brief interfaces |

## Implementation summary

### Public API (as specified)

| Function | Behavior |
|----------|----------|
| `extract_user_query(text)` | Unwraps `<user_query>` tags; plain-text fallback; skips JSON role dumps |
| `draft_sharpen(messy)` | Collapses whitespace, capitalizes, appends `?` if no terminal punctuation, truncates at 240 chars |
| `iter_transcript_user_texts(path)` | Parses JSONL user lines, returns extracted queries |
| `iter_transcript_qa_pairs(path)` | Pairs user query with first assistant paragraph (< 4000 chars) |
| `pairs_from_markdown(text, source)` | Splits H1/H2 sections into `me_assistant` candidate dicts |
| `sharpen_candidates_from_texts(texts, source)` | Bonus helper from brief (not directly tested) |

### Candidate dict shape

```python
{"instruction": str, "response": str, "source": str, "kind": "sharpen" | "me_assistant"}
```

## Deviation from brief (bugfix)

The brief's markdown heading regex used greedy `.+` before `$` in multiline mode, which matched the entire document as one heading (body empty → zero pairs). Fixed to `[^\n]+` for single-line headings:

```python
r"(?m)^(#{1,2}\s+[^\n]+)$" r"(.*?)(?=^#{1,2}\s+|\Z)"
```

Also removed unused dead variable `sections = re.split(...)` from brief snippet.

## Self-review

| Check | Result |
|-------|--------|
| Brief test cases verbatim | Pass |
| Only specified files committed | Pass |
| All 6 tests green | Pass |
| Imports match project convention (`scripts.lib.*`) | Pass |
| `sharpen_candidates_from_texts` included per brief | Pass (untested helper) |
| No inline imports | Pass |
| Commit message matches brief | Pass |

### Concerns

1. **pytest not pre-installed in `.venv`:** Had to `pip install pytest` before RED/GREEN runs. Consider adding pytest to project dev deps.
2. **Markdown regex:** Brief regex was broken; fix is minimal and preserves intended H1/H2 section splitting behavior.
3. **`sharpen_candidates_from_texts` untested:** Present per brief but no test coverage in this task; downstream tasks may rely on it.

## Out of scope

- CLI scripts (`extract_personal_candidates.py`, etc.)
- Integration with `config/personal_sources.env`
- README updates
