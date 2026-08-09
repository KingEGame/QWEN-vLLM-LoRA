from types import SimpleNamespace

from scripts.lib.personal_pipeline import run_pipeline


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, *, model, messages):
        self.calls.append((model, messages[0]["content"]))
        if model == "question-sharper":
            text = "How do I free VRAM before QLoRA training?"
        else:
            text = "Stop the vLLM server, then run nvidia-smi."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_run_pipeline_chains_models():
    client = _FakeClient()
    result = run_pipeline(
        client,
        "okey so like train fails maybe vram?",
        sharp_model="question-sharper",
        answer_model="me-assistant",
    )
    assert result["sharpened"].startswith("How do I")
    assert "vLLM" in result["answer"]
    assert [c[0] for c in client.chat.completions.calls] == [
        "question-sharper",
        "me-assistant",
    ]


def test_run_pipeline_aborts_on_empty_sharpen():
    class EmptySharp(_FakeCompletions):
        def create(self, *, model, messages):
            self.calls.append((model, messages[0]["content"]))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  "))])

    client = _FakeClient()
    client.chat.completions = EmptySharp()
    try:
        run_pipeline(client, "x", sharp_model="question-sharper", answer_model="me-assistant")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    assert len(client.chat.completions.calls) == 1
