"""Persistent local goal ledger for the personal agent."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


VALID_STATUSES = {"active", "blocked", "complete"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskLedger:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("task ledger must contain a JSON array")
        return value

    def save(self, tasks: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(self.path)

    def active_task(self) -> dict | None:
        active = [task for task in self.load() if task.get("status") == "active"]
        return active[-1] if active else None

    def start(self, goal: str, *, resume: bool = True) -> dict:
        tasks = self.load()
        if resume:
            active = next(
                (task for task in reversed(tasks) if task.get("status") == "active"),
                None,
            )
            if active is not None:
                active.setdefault("updates", []).append(
                    {"at": utc_now(), "note": f"Follow-up request: {goal}"}
                )
                active["updated_at"] = utc_now()
                self.save(tasks)
                return active
        created = utc_now()
        digest = hashlib.sha256(f"{created}:{goal}".encode("utf-8")).hexdigest()[:10]
        task = {
            "id": f"task_{digest}",
            "goal": goal,
            "status": "active",
            "created_at": created,
            "updated_at": created,
            "updates": [],
        }
        tasks.append(task)
        self.save(tasks)
        return task

    def update(self, task_id: str, *, status: str | None = None, note: str = "") -> dict:
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"invalid task status: {status}")
        tasks = self.load()
        task = next((item for item in tasks if item.get("id") == task_id), None)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        if status is not None:
            task["status"] = status
        if note.strip():
            task.setdefault("updates", []).append({"at": utc_now(), "note": note.strip()})
        task["updated_at"] = utc_now()
        self.save(tasks)
        return task

    @staticmethod
    def render_context(task: dict) -> str:
        updates = task.get("updates") if isinstance(task.get("updates"), list) else []
        recent = "\n".join(f"- {item.get('note', '')}" for item in updates[-5:])
        return (
            f"Task ID: {task.get('id')}\n"
            f"Goal: {task.get('goal')}\n"
            f"Status: {task.get('status')}\n"
            f"Recent updates:\n{recent or '- none'}"
        )
