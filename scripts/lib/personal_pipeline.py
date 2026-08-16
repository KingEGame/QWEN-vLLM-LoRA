"""Chain question-sharper → me-assistant over an OpenAI-compatible client."""
from __future__ import annotations

from typing import Any


def run_pipeline(
    client: Any,
    raw: str,
    *,
    sharp_model: str,
    answer_model: str,
    base_only: bool = False,
) -> dict[str, str]:
    sharp_messages = [{"role": "user", "content": raw}]
    sharp_options: dict[str, Any] = {}
    answer_options: dict[str, Any] = {}
    if base_only:
        sharp_messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "Rewrite the user's text as one clear, concise technical "
                    "question. Do not answer it. Return only the rewritten question."
                ),
            },
        )
        # Qwen3.8 reasoning can consume the response budget without producing
        # message.content. The personal adapters are left unchanged; this is
        # only for the explicit base-model fallback.
        no_thinking = {"chat_template_kwargs": {"enable_thinking": False}}
        sharp_options = {"max_tokens": 128, "extra_body": no_thinking}
        answer_options = {"max_tokens": 1024, "extra_body": no_thinking}

    sharp = client.chat.completions.create(
        model=sharp_model,
        messages=sharp_messages,
        **sharp_options,
    )
    sharpened = (sharp.choices[0].message.content or "").strip()
    if not sharpened:
        raise ValueError("question-sharper returned an empty question")

    ans = client.chat.completions.create(
        model=answer_model,
        messages=[{"role": "user", "content": sharpened}],
        **answer_options,
    )
    answer = (ans.choices[0].message.content or "").strip()
    return {"raw": raw, "sharpened": sharpened, "answer": answer}
