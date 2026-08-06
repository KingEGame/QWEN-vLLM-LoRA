from scripts.lib.data_gen import build_generation_prompt, parse_generated_response


def test_build_generation_prompt_includes_chunk_and_count():
    prompt = build_generation_prompt("Refunds take 5 business days.", num_pairs=2)

    assert "Refunds take 5 business days." in prompt
    assert "exactly 2 question-and-answer pairs" in prompt


def test_parse_generated_response_extracts_valid_pairs():
    response = '[{"instruction": "How long do refunds take?", "response": "5 business days."}]'

    pairs = parse_generated_response(response)

    assert pairs == [{"instruction": "How long do refunds take?", "response": "5 business days."}]


def test_parse_generated_response_drops_incomplete_entries():
    response = '[{"instruction": "Q1"}, {"instruction": "Q2", "response": "A2"}]'

    pairs = parse_generated_response(response)

    assert pairs == [{"instruction": "Q2", "response": "A2"}]


def test_parse_generated_response_returns_empty_list_for_invalid_json():
    pairs = parse_generated_response("not json at all")

    assert pairs == []


def test_parse_generated_response_returns_empty_list_for_non_array():
    pairs = parse_generated_response('{"instruction": "Q1", "response": "A1"}')

    assert pairs == []
