from types import SimpleNamespace

from scripts.lib.personal_pipeline import run_pipeline


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, *, model, messages, **kwargs):
        self.calls.append((model, messages[0]["content"], kwargs))
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
        def create(self, *, model, messages, **kwargs):
            self.calls.append((model, messages[0]["content"], kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  "))])

    client = _FakeClient()
    client.chat.completions = EmptySharp()
    try:
        run_pipeline(client, "x", sharp_model="question-sharper", answer_model="me-assistant")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    assert len(client.chat.completions.calls) == 1


def test_base_only_adds_sharpening_instruction_and_disables_thinking():
    client = _FakeClient()
    result = run_pipeline(
        client,
        "vllm oom",
        sharp_model="base",
        answer_model="base",
        base_only=True,
    )
    assert result["sharpened"]
    assert [c[0] for c in client.chat.completions.calls] == ["base", "base"]
    assert client.chat.completions.calls[0][1].startswith("Rewrite the user's text")
    assert client.chat.completions.calls[0][2]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
