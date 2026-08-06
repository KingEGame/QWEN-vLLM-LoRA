"""Split raw text into chunks small enough to fit in a generation prompt."""


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Split text into paragraph-respecting chunks no longer than max_chars.

    Paragraphs (separated by blank lines) are kept whole when possible;
    a single paragraph longer than max_chars is hard-split.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start:start + max_chars])
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
