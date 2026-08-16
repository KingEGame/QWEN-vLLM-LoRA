### Task 6: Pipeline client (TDD)

**Files:**
- Create: `scripts/personal_pipeline.py`
- Create: `scripts/lib/personal_pipeline.py` (pure functions for testability)
- Test: `tests/test_personal_pipeline.py`

**Interfaces:**
- Consumes: OpenAI-compatible client; model names `question-sharper`, `me-assistant`; port from config
- Produces:
  - `run_pipeline(client, raw: str, *, sharp_model: str, answer_model: str) -> dict` with keys `raw`, `sharpened`, `answer`
  - CLI prints both steps; appends JSON line to `output/personal_runs.jsonl` unless `--no-log`
  - Empty sharpened → exit 1, no assistant call

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/test_personal_pipeline.py -v
```

- [ ] **Step 3: Implement library + CLI**

`scripts/lib/personal_pipeline.py`:

```python
"""Chain question-sharper → me-assistant over an OpenAI-compatible client."""
from __future__ import annotations

from typing import Any


def run_pipeline(
    client: Any,
    raw: str,
    *,
    sharp_model: str,
    answer_model: str,
) -> dict[str, str]:
    sharp = client.chat.completions.create(
        model=sharp_model,
        messages=[{"role": "user", "content": raw}],
    )
    sharpened = (sharp.choices[0].message.content or "").strip()
    if not sharpened:
        raise ValueError("question-sharper returned an empty question")

    ans = client.chat.completions.create(
        model=answer_model,
        messages=[{"role": "user", "content": sharpened}],
    )
    answer = (ans.choices[0].message.content or "").strip()
    return {"raw": raw, "sharpened": sharpened, "answer": answer}
```

`scripts/personal_pipeline.py`:

```python
#!/usr/bin/env python3
"""Run personal tech pipeline: sharpen question, then answer as me-assistant."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from scripts.lib.env_config import load_env_file
from scripts.lib.personal_pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "output" / "personal_runs.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Messy tech thought / question")
    parser.add_argument("--sharp-model", default="question-sharper")
    parser.add_argument("--answer-model", default="me-assistant")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    config = load_env_file(REPO_ROOT / "config" / "model.env")
    port = config.get("PORT", "8000")
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

    try:
        result = run_pipeline(
            client,
            args.prompt,
            sharp_model=args.sharp_model,
            answer_model=args.answer_model,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Raw: {result['raw']}")
    print(f"Sharpened: {result['sharpened']}")
    print(f"Answer: {result['answer']}")

    if not args.no_log:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Logged → {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit tests — expect PASS**

```bash
python -m pytest tests/test_personal_pipeline.py tests/test_personal_extract.py -v
```

- [ ] **Step 5: Commit** (if requested)

```bash
git add scripts/lib/personal_pipeline.py scripts/personal_pipeline.py tests/test_personal_pipeline.py
git commit -m "feat: personal pipeline client with run logging"
```

---

