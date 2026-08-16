#!/usr/bin/env python3
"""Run or inspect Ilian's persistent local tool-using agent."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from scripts.lib.agent_runtime import run_agent
from scripts.lib.agent_tools import AgentTools
from scripts.lib.env_config import load_env_file
from scripts.lib.task_ledger import TaskLedger


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_CONFIG = REPO_ROOT / "config" / "agent.env"
PROFILE = (
    REPO_ROOT
    / "data"
    / "personal"
    / "qwen_dataset_toolkit"
    / "personal-assistant-data"
    / "memory"
    / "ilian-assistant-profile.md"
)
POLICY = PROFILE.with_name("review-derived-policy.md")


def configured_path(config: dict[str, str], key: str, default: str) -> Path:
    value = Path(config.get(key, default))
    return value if value.is_absolute() else REPO_ROOT / value


def compact_profile(text: str) -> str:
    marker = "## 12. Short system-profile version"
    if marker not in text:
        return text
    return text.split(marker, 1)[1].strip()


def load_memory_context(runtime_memory: Path, max_chars: int = 3000) -> str:
    sections: list[str] = []
    for path in (PROFILE, POLICY):
        if path.exists():
            text = path.read_text(encoding="utf-8")
            sections.append(compact_profile(text) if path == PROFILE else text)
    if runtime_memory.exists():
        lines = runtime_memory.read_text(encoding="utf-8").splitlines()[-100:]
        sections.append("Runtime memory:\n" + "\n".join(lines))
    return "\n\n".join(sections)[-max_chars:]


def status(config: dict[str, str]) -> int:
    ledger_path = configured_path(config, "AGENT_TASK_LEDGER", "output/agent/tasks.json")
    ledger = TaskLedger(ledger_path)
    tasks = ledger.load()
    counts: dict[str, int] = {}
    for task in tasks:
        value = str(task.get("status") or "unknown")
        counts[value] = counts.get(value, 0) + 1
    train_data = configured_path(config, "AGENT_TRAIN_DATA", "data/personal/agent_train.jsonl")
    examples = (
        sum(1 for line in train_data.read_text(encoding="utf-8").splitlines() if line.strip())
        if train_data.exists()
        else 0
    )
    active = ledger.active_task()
    run_log = configured_path(config, "AGENT_RUN_LOG", "output/agent/runs.jsonl")
    latest_run = None
    if run_log.exists():
        lines = [line for line in run_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            latest = json.loads(lines[-1])
            latest_run = {
                "at": latest.get("at"),
                "task_id": latest.get("task_id"),
                "steps": latest.get("steps"),
                "tool_calls": len(latest.get("tool_events") or []),
                "edited": bool(latest.get("edited")),
                "verified_after_edit": bool(latest.get("verified_after_edit")),
            }
    print(json.dumps({
        "model": config.get("AGENT_MODEL"),
        "base_model": config.get("AGENT_BASE_MODEL"),
        "train_examples": examples,
        "minimum_train_examples": int(config.get("AGENT_MIN_TRAIN_EXAMPLES", "20")),
        "task_counts": counts,
        "active_task": (
            {"id": active.get("id"), "goal": active.get("goal")}
            if active
            else None
        ),
        "latest_run": latest_run,
        "task_sync_provider": config.get("AGENT_TASK_SYNC_PROVIDER", "local"),
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show model, dataset, and task-ledger status")
    task_parser = subparsers.add_parser("task", help="Update a persistent task explicitly")
    task_parser.add_argument("task_id")
    task_parser.add_argument("--status", required=True, choices=("active", "blocked", "complete"))
    task_parser.add_argument("--note", default="")
    run_parser = subparsers.add_parser("run", help="Run one persistent agent turn")
    run_parser.add_argument("prompt")
    run_parser.add_argument("--new-task", action="store_true")
    run_parser.add_argument("--allow-write", action="store_true")
    run_parser.add_argument("--allow-command", action="store_true")
    run_parser.add_argument("--base-only", action="store_true")
    run_parser.add_argument("--model", default=None, help="Override the configured served model name")
    args = parser.parse_args()

    config = load_env_file(AGENT_CONFIG)
    if args.command == "status":
        return status(config)
    if args.command == "task":
        ledger_path = configured_path(config, "AGENT_TASK_LEDGER", "output/agent/tasks.json")
        task = TaskLedger(ledger_path).update(
            args.task_id, status=args.status, note=args.note
        )
        print(json.dumps({"id": task["id"], "status": task["status"]}, indent=2))
        return 0

    workspace = configured_path(config, "AGENT_WORKSPACE", ".").resolve()
    ledger_path = configured_path(config, "AGENT_TASK_LEDGER", "output/agent/tasks.json")
    memory_path = configured_path(config, "AGENT_RUNTIME_MEMORY", "output/agent/memory.jsonl")
    log_path = configured_path(config, "AGENT_RUN_LOG", "output/agent/runs.jsonl")
    backup_dir = REPO_ROOT / "output" / "agent" / "backups"
    ledger = TaskLedger(ledger_path)
    auto_resume = config.get("AGENT_AUTO_RESUME", "1").casefold() in {"1", "true", "yes"}
    task = ledger.start(args.prompt, resume=auto_resume and not args.new_task)
    tools = AgentTools(
        workspace,
        ledger=ledger,
        task_id=task["id"],
        memory_path=memory_path,
        backup_dir=backup_dir,
        allow_write=args.allow_write,
        allow_command=args.allow_command,
    )
    port = config.get("AGENT_PORT", "8000")
    model = args.model or (
        config.get("AGENT_BASE_MODEL") if args.base_only else config.get("AGENT_MODEL")
    )
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed")
    try:
        result = run_agent(
            client,
            args.prompt,
            model=str(model),
            tools=tools,
            task_context=TaskLedger.render_context(task),
            memory_context=load_memory_context(
                memory_path,
                max_chars=int(config.get("AGENT_MEMORY_MAX_CHARS", "3000")),
            ),
            max_steps=int(config.get("AGENT_MAX_STEPS", "12")),
            max_tokens=int(config.get("AGENT_MAX_OUTPUT_TOKENS", "512")),
        )
    except Exception as exc:
        ledger.update(task["id"], status="active", note=f"Runtime error: {exc}")
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(result["answer"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "task_id": task["id"],
            "prompt": args.prompt,
            **result,
        }, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
