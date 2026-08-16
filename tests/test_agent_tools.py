from pathlib import Path

import pytest

from scripts.lib.agent_tools import AgentTools
from scripts.lib.task_ledger import TaskLedger


def make_tools(tmp_path: Path, **kwargs) -> AgentTools:
    ledger = TaskLedger(tmp_path / "state" / "tasks.json")
    task = ledger.start("Test", resume=False)
    return AgentTools(
        tmp_path,
        ledger=ledger,
        task_id=task["id"],
        memory_path=tmp_path / "state" / "memory.jsonl",
        backup_dir=tmp_path / "state" / "backups",
        **kwargs,
    )


def test_paths_cannot_escape_workspace(tmp_path):
    tools = make_tools(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        tools.resolve("../outside.txt")


def test_replace_requires_read_and_write_authorization(tmp_path):
    target = tmp_path / "config.txt"
    target.write_text("PORT=8000\n", encoding="utf-8")
    tools = make_tools(tmp_path, allow_write=True)
    blocked = tools.execute(
        "replace_text", {"path": "config.txt", "old": "8000", "new": "9000"}
    )
    assert not blocked["ok"]
    tools.execute("read_file", {"path": "config.txt"})
    changed = tools.execute(
        "replace_text", {"path": "config.txt", "old": "8000", "new": "9000"}
    )
    assert changed["ok"]
    assert target.read_text(encoding="utf-8") == "PORT=9000\n"
    assert tools.edited and not tools.verified_after_edit


def test_command_allowlist_rejects_shells():
    with pytest.raises(PermissionError, match="allowlist"):
        AgentTools._validate_command(["bash", "-lc", "echo unsafe"])
    AgentTools._validate_command(["python", "-m", "pytest", "-q"])
    AgentTools._validate_command(["git", "diff"])


def test_create_file_never_overwrites_existing_path(tmp_path):
    tools = make_tools(tmp_path, allow_write=True)
    created = tools.execute("create_file", {"path": "new.txt", "content": "hello\n"})
    assert created["ok"]
    refused = tools.execute("create_file", {"path": "new.txt", "content": "changed\n"})
    assert not refused["ok"]
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\n"


def test_memory_rejects_credentials(tmp_path):
    tools = make_tools(tmp_path)
    result = tools.execute(
        "remember", {"kind": "preference", "text": "token sk-1234567890abcdefghijklmnop"}
    )
    assert not result["ok"]
