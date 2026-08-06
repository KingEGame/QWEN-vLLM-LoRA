import pytest

from scripts.lib.chunking import chunk_text


def test_chunk_text_keeps_short_text_as_single_chunk():
    text = "Paragraph one.\n\nParagraph two."

    chunks = chunk_text(text, max_chars=2000)

    assert chunks == ["Paragraph one.\n\nParagraph two."]


def test_chunk_text_splits_at_paragraph_boundary_when_over_limit():
    text = "A" * 50 + "\n\n" + "B" * 50

    chunks = chunk_text(text, max_chars=60)

    assert chunks == ["A" * 50, "B" * 50]


def test_chunk_text_hard_splits_oversized_paragraph():
    text = "A" * 130

    chunks = chunk_text(text, max_chars=50)

    assert chunks == ["A" * 50, "A" * 50, "A" * 30]


def test_chunk_text_rejects_non_positive_max_chars():
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=0)


def test_chunk_text_ignores_blank_paragraphs():
    text = "First.\n\n\n\nSecond."

    chunks = chunk_text(text, max_chars=2000)

    assert chunks == ["First.\n\nSecond."]
