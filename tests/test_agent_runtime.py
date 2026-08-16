from types import SimpleNamespace

from scripts.lib.agent_runtime import run_agent


def message(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def tool_call(call_id, name, arguments="{}"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeTools:
    def __init__(self, *, edited=False):
        self.edited = edited
        self.verified_after_edit = False
        self.events = []

    def schemas(self):
        return []

    def execute(self, name, arguments):
        self.events.append({"tool": name})
        if name == "run_command":
            self.verified_after_edit = True
        return {"ok": True}


class FakeClient:
    def __init__(self, messages):
        self.responses = list(messages)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self.responses.pop(0))]
        )


def test_agent_returns_direct_final_when_no_tools_are_needed():
    client = FakeClient([message("Need one clarification.")])
    result = run_agent(
        client,
        "approach 3",
        model="agent",
        tools=FakeTools(),
        task_context="active",
    )
    assert result["answer"] == "Need one clarification."
    assert result["steps"] == 1
    assert client.calls[0]["max_tokens"] == 512


def test_agent_requires_verification_after_an_edit():
    tools = FakeTools(edited=True)
    client = FakeClient(
        [
            message("Done."),
            message(tool_calls=[tool_call("1", "run_command", '{"argv":["pytest"]}')]),
            message("Fixed and verified."),
        ]
    )
    result = run_agent(
        client,
        "fix it",
        model="agent",
        tools=tools,
        task_context="active",
        max_steps=4,
    )
    assert result["answer"] == "Fixed and verified."
    assert result["verified_after_edit"]
    assert len(client.calls) == 3
