"""OpenAI-compatible tool loop for Ilian's persistent local agent."""
from __future__ import annotations

import json
from typing import Any

from scripts.lib.agent_tools import AgentTools


BASE_SYSTEM_PROMPT = """You are Ilian's private technical implementation agent.

Interpret noisy voice-to-text from context and silently normalize spelling.
Lead with useful work, not narration. Use tools to inspect the workspace and
current task before making project-specific claims. For an attainable,
authorized task: inspect, act, verify, and continue until the goal is achieved.

Rules:
- Never say work is done unless tool results prove it.
- Read a file before editing it. After any edit, run a relevant verification.
- Diagnose root causes before applying symptom-masking workarounds.
- Do not ask Ilian to paste files, paths, or logs that available tools can read.
- Ask one focused question only when missing context or authority blocks the
  goal or materially changes the implementation.
- Keep the final response concise: outcome, verification, and any real blocker.
- Update the task ledger as progress changes. Mark complete only after the goal
  and verification are complete; otherwise leave it active or mark it blocked.
- Store only stable preferences, conventions, and decisions in memory. Keep
  changing task state in the task ledger. Never store credentials.
- External/cloud actions require an available connector and appropriate
  authorization. If unavailable, record the local unfinished task and say what
  connection is missing.
"""


def serialize_assistant_message(message: Any) -> dict:
    result: dict[str, Any] = {
        "role": "assistant",
        "content": getattr(message, "content", None) or "",
    }
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ]
    return result


def run_agent(
    client: Any,
    prompt: str,
    *,
    model: str,
    tools: AgentTools,
    task_context: str,
    memory_context: str = "",
    max_steps: int = 12,
    max_tokens: int = 512,
) -> dict:
    system = BASE_SYSTEM_PROMPT
    if memory_context.strip():
        system += "\nStable memory and policy:\n" + memory_context.strip()
    system += "\nCurrent persistent task:\n" + task_context.strip()
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    last_text = ""
    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools.schemas(),
            tool_choice="auto",
            temperature=0.1,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        message = response.choices[0].message
        assistant_message = serialize_assistant_message(message)
        messages.append(assistant_message)
        last_text = assistant_message.get("content", "").strip() or last_text
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            if tools.edited and not tools.verified_after_edit and step < max_steps:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "A file was edited but no successful verification ran afterward. "
                            "Use run_command for a relevant test/build/check, or explain a real "
                            "verification blocker and update the task status."
                        ),
                    }
                )
                continue
            return {
                "answer": last_text,
                "steps": step,
                "tool_events": tools.events,
                "edited": tools.edited,
                "verified_after_edit": tools.verified_after_edit,
            }
        for call in calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                result = tools.execute(call.function.name, arguments)
            except (json.JSONDecodeError, ValueError) as exc:
                result = {"ok": False, "error": str(exc)}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
    return {
        "answer": last_text or "The agent reached its step limit without a final answer.",
        "steps": max_steps,
        "tool_events": tools.events,
        "edited": tools.edited,
        "verified_after_edit": tools.verified_after_edit,
        "step_limit_reached": True,
    }
