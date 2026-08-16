import base64
import importlib.util
import json
import sys
import zipfile
from pathlib import Path


TOOLKIT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "personal"
    / "qwen_dataset_toolkit"
)
sys.path.insert(0, str(TOOLKIT))
MODULE_PATH = TOOLKIT / "import_cursor_archive.py"
SPEC = importlib.util.spec_from_file_location("import_cursor_archive", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def blob(message):
    return base64.b64encode(json.dumps(message).encode()).decode()


def test_imports_query_and_text_only_assistant_blocks(tmp_path):
    archive_path = tmp_path / "cursor.zip"
    export = {
        "version": 1,
        "blobs": {
            "system": blob({"role": "system", "content": "secret system prompt"}),
            "context": blob(
                {"role": "user", "content": "<user_info>OS and private context</user_info>"}
            ),
            "user": blob(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<system_reminder>noise</system_reminder>"},
                        {"type": "text", "text": "<user_query>help me debug this</user_query>"},
                    ],
                }
            ),
            "assistant": blob(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "hidden reasoning"},
                        {"type": "text", "text": "Here is the useful answer."},
                        {"type": "tool_call", "name": "shell"},
                    ],
                }
            ),
        },
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("cursor chats/chat.json", json.dumps(export))

    conversations, stats = MODULE.import_archive(archive_path)
    assert conversations[0]["messages"] == [
        {"role": "user", "content": "help me debug this"},
        {"role": "assistant", "content": "Here is the useful answer."},
    ]
    assert stats.ignored_system_messages == 1
    assert stats.ignored_context_only_user_messages == 1


def test_rejects_zip_traversal(tmp_path):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.json", "{}")
    with zipfile.ZipFile(archive_path) as archive:
        try:
            MODULE.safe_archive_entries(archive)
        except ValueError as exc:
            assert "unsafe archive entry" in str(exc)
        else:
            raise AssertionError("unsafe path was accepted")
