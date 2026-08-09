"""Chain question-sharper → me-assistant over an OpenAI-compatible client."""
from __future__ import annotations

from typing import Any


def run_pipeline(
    client: Any,
    raw: str,
    *,
    sharp_model: str,
    answer_model: str,
) -> dict[str, str]:
    sharp = client.chat.completions.create(
        model=sharp_model,
        messages=[{"role": "user", "content": raw}],
    )
    sharpened = (sharp.choices[0].message.content or "").strip()
    if not sharpened:
        raise ValueError("question-sharper returned an empty question")

    ans = client.chat.completions.create(
        model=answer_model,
        messages=[{"role": "user", "content": sharpened}],
    )
    answer = (ans.choices[0].message.content or "").strip()
    return {"raw": raw, "sharpened": sharpened, "answer": answer}
