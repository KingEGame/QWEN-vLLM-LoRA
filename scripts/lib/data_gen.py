"""Prompt building and response parsing for synthetic Q&A generation."""
import json

GENERATION_INSTRUCTIONS = (
    "You are generating customer support training examples from the "
    "reference text below. Produce exactly {n} question-and-answer pairs "
    "a customer might ask that are answerable from this text. Respond with "
    "ONLY a JSON array of objects, each with keys \"instruction\" and "
    "\"response\". No markdown, no commentary, just the JSON array."
)


def build_generation_prompt(chunk: str, num_pairs: int = 3) -> str:
    """Build the prompt sent to the model to generate Q&A pairs from a chunk."""
    instructions = GENERATION_INSTRUCTIONS.format(n=num_pairs)
    return f"{instructions}\n\nReference text:\n{chunk}"


def parse_generated_response(response_text: str) -> list[dict[str, str]]:
    """Parse the model's JSON-array response into a list of Q&A dicts.

    Silently drops entries missing required fields rather than raising,
    since this output feeds a human review step, not training directly.
    """
    try:
        parsed = json.loads(response_text.strip())
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    pairs: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        instruction = item.get("instruction")
        response = item.get("response")
        if isinstance(instruction, str) and instruction.strip() and \
           isinstance(response, str) and response.strip():
            pairs.append({"instruction": instruction.strip(), "response": response.strip()})

    return pairs
