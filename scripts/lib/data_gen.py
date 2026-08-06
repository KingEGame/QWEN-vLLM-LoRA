"""Prompt building and response parsing for synthetic Q&A generation."""
import json
import re

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


def _extract_json_array_text(response_text: str) -> str:
    """Recover a JSON array substring from a response that may be wrapped
    in a markdown code fence or have leading/trailing commentary.

    Models frequently ignore "respond with ONLY JSON" instructions in
    exactly these two ways, so this is applied before every parse attempt.
    """
    stripped = response_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        return stripped[start:end + 1]

    return stripped


def parse_generated_response(response_text: str) -> list[dict[str, str]]:
    """Parse the model's JSON-array response into a list of Q&A dicts.

    Tolerates markdown code fences and leading/trailing commentary around
    the JSON array, since models frequently add these despite being told
    not to. Silently drops entries missing required fields rather than
    raising, since this output feeds a human review step, not training
    directly.
    """
    try:
        parsed = json.loads(_extract_json_array_text(response_text))
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
