"""Workspace-scoped tools for the local personal agent."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.lib.task_ledger import TaskLedger


SECRET_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"-----BEGIN [^-]*PRIVATE KEY-----)",
    re.IGNORECASE,
)


class AgentTools:
    def __init__(
        self,
        workspace: Path,
        *,
        ledger: TaskLedger,
        task_id: str,
        memory_path: Path,
        backup_dir: Path,
        allow_write: bool = False,
        allow_command: bool = False,
    ):
        self.workspace = Path(workspace).resolve()
        self.ledger = ledger
        self.task_id = task_id
        self.memory_path = Path(memory_path)
        self.backup_dir = Path(backup_dir)
        self.allow_write = allow_write
        self.allow_command = allow_command
        self.read_paths: set[Path] = set()
        self.edited = False
        self.verified_after_edit = False
        self.events: list[dict] = []

    def resolve(self, value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError("tool paths must be workspace-relative")
        resolved = (self.workspace / candidate).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError("path escapes the configured workspace") from exc
        return resolved

    def schemas(self) -> list[dict]:
        return [
            self._schema("list_files", "List workspace files matching a glob.", {
                "pattern": {"type": "string"}
            }, ["pattern"]),
            self._schema("read_file", "Read a UTF-8 text file before editing it.", {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "minimum": 1, "maximum": 100000},
            }, ["path"]),
            self._schema("search_text", "Search workspace text files with a regex.", {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            }, ["pattern"]),
            self._schema("replace_text", "Replace one exact text block in a previously read file.", {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            }, ["path", "old", "new"]),
            self._schema("create_file", "Create a new UTF-8 file; refuses to overwrite an existing path.", {
                "path": {"type": "string"},
                "content": {"type": "string"},
            }, ["path", "content"]),
            self._schema("run_command", "Run a constrained argv command for inspection or verification.", {
                "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
            }, ["argv"]),
            self._schema("update_task", "Record progress or mark the persistent task active, blocked, or complete.", {
                "status": {"type": "string", "enum": ["active", "blocked", "complete"]},
                "note": {"type": "string"},
            }, ["status", "note"]),
            self._schema("remember", "Store a stable preference, convention, or decision in local memory.", {
                "kind": {"type": "string", "enum": ["preference", "project_convention", "decision"]},
                "text": {"type": "string"},
            }, ["kind", "text"]),
        ]

    @staticmethod
    def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            result = {"ok": False, "error": f"unknown tool: {name}"}
        else:
            try:
                result = handler(**arguments)
            except Exception as exc:  # Tool errors are observations for the model.
                result = {"ok": False, "error": str(exc)}
        self.events.append({"tool": name, "arguments": arguments, "result": result})
        return result

    def tool_list_files(self, pattern: str) -> dict:
        files = sorted(
            str(path.relative_to(self.workspace)).replace("\\", "/")
            for path in self.workspace.glob(pattern)
            if path.is_file()
        )[:500]
        return {"ok": True, "files": files, "truncated": len(files) == 500}

    def tool_read_file(self, path: str, max_chars: int = 30000) -> dict:
        resolved = self.resolve(path)
        if not resolved.is_file():
            raise FileNotFoundError(path)
        text = resolved.read_text(encoding="utf-8")
        self.read_paths.add(resolved)
        limit = min(max(int(max_chars), 1), 100000)
        return {"ok": True, "path": path, "content": text[:limit], "truncated": len(text) > limit}

    def tool_search_text(self, pattern: str, path: str = ".") -> dict:
        regex = re.compile(pattern)
        base = self.resolve(path)
        candidates = [base] if base.is_file() else base.rglob("*")
        matches: list[dict] = []
        for candidate in candidates:
            if not candidate.is_file() or any(part.startswith(".git") for part in candidate.parts):
                continue
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append({
                        "path": str(candidate.relative_to(self.workspace)).replace("\\", "/"),
                        "line": line_number,
                        "text": line[:500],
                    })
                    if len(matches) >= 200:
                        return {"ok": True, "matches": matches, "truncated": True}
        return {"ok": True, "matches": matches, "truncated": False}

    def tool_replace_text(self, path: str, old: str, new: str) -> dict:
        if not self.allow_write:
            raise PermissionError("write tools are disabled; start with --allow-write")
        resolved = self.resolve(path)
        if resolved not in self.read_paths:
            raise PermissionError("read_file must inspect this file before replace_text")
        original = resolved.read_text(encoding="utf-8")
        count = original.count(old)
        if count != 1:
            raise ValueError(f"old text must occur exactly once; found {count}")
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = self.backup_dir / f"{digest}-{resolved.name}"
        backup.write_text(original, encoding="utf-8", newline="\n")
        updated = original.replace(old, new, 1)
        temporary = resolved.with_suffix(resolved.suffix + ".agent-tmp")
        temporary.write_text(updated, encoding="utf-8", newline="\n")
        temporary.replace(resolved)
        self.edited = True
        self.verified_after_edit = False
        return {"ok": True, "path": path, "backup": str(backup), "changed": True}

    def tool_create_file(self, path: str, content: str) -> dict:
        if not self.allow_write:
            raise PermissionError("write tools are disabled; start with --allow-write")
        resolved = self.resolve(path)
        if resolved.exists():
            raise FileExistsError("create_file refuses to overwrite an existing path")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8", newline="\n")
        self.edited = True
        self.verified_after_edit = False
        return {"ok": True, "path": path, "created": True}

    def tool_run_command(self, argv: list[str], timeout_seconds: int = 120) -> dict:
        if not self.allow_command:
            raise PermissionError("command tools are disabled; start with --allow-command")
        self._validate_command(argv)
        completed = subprocess.run(
            argv,
            cwd=self.workspace,
            capture_output=True,
            text=True,
            timeout=min(max(int(timeout_seconds), 1), 600),
            shell=False,
        )
        if self.edited and completed.returncode == 0:
            self.verified_after_edit = True
        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
        }

    @staticmethod
    def _validate_command(argv: list[str]) -> None:
        if not argv or not all(isinstance(item, str) and item for item in argv):
            raise ValueError("argv must contain non-empty strings")
        executable = Path(argv[0]).name.casefold()
        if executable in {"rg", "pytest"}:
            return
        if executable in {"python", "python3"} and argv[1:3] == ["-m", "pytest"]:
            return
        if executable == "git" and len(argv) > 1 and argv[1] in {"status", "diff", "log", "show"}:
            return
        if executable in {"npm", "npm.cmd"} and len(argv) > 1:
            if argv[1] == "test" or (argv[1:3] in (["run", "test"], ["run", "build"], ["run", "lint"])):
                return
        raise PermissionError("command is outside the verification allowlist")

    def tool_update_task(self, status: str, note: str) -> dict:
        task = self.ledger.update(self.task_id, status=status, note=note)
        return {"ok": True, "task_id": task["id"], "status": task["status"]}

    def tool_remember(self, kind: str, text: str) -> dict:
        if kind not in {"preference", "project_convention", "decision"}:
            raise ValueError("invalid memory kind")
        if SECRET_RE.search(text):
            raise ValueError("refusing to store credential-like content in memory")
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "at": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "text": text.strip(),
            "task_id": self.task_id,
        }
        with self.memory_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"ok": True, "stored": True}
