# Qwen3-4B + vLLM Serving with LoRA Customization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the scripts, config, and docs to serve Qwen3-4B via vLLM with one start command, generate LoRA training data from source docs via the model itself, train a LoRA adapter with Unsloth, and serve that adapter alongside the base model.

**Architecture:** Pure-logic pieces (config parsing, dataset validation, text chunking, prompt/response handling for data generation) live in `scripts/lib/` and are unit-tested with pytest on this authoring machine. Thin CLI/shell wrappers around them (`generate_training_data.py`, `train_lora.py`, `test_client.py`, the `.sh` scripts) depend on GPU/vLLM/network and can only be syntax-checked here — they are designed to be self-verifying (clear exit codes and printed diagnostics) when actually run on the target Linux/WSL2 machine.

**Tech Stack:** Python 3.11+, pytest, vLLM, Unsloth + TRL + PEFT (QLoRA training), `openai` Python client (talks to vLLM's OpenAI-compatible server), bash.

---

## Before You Start

This plan targets the repo at `E:\Projects\QWEN+vLLM+LoRA` (already a git repo, one commit so far containing the design spec at `docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md` — read it if anything below is unclear on *why*).

Tasks 1–8 run entirely on this authoring machine (Windows, Python 3.14, pytest 9.0.3, Git Bash — all confirmed available). Tasks 9–14 produce GPU/Linux-dependent scripts that can only be syntax-checked here; they're verified for real the first time someone runs `scripts/setup.sh` on the target machine.

## File Structure

```
QWEN+vLLM+LoRA/
├── pyproject.toml                  # pytest config (pythonpath)
├── requirements.txt                # target-machine Python deps
├── .gitignore
├── config/
│   └── model.env                   # single source of truth: model, port, quant, adapter name
├── data/
│   ├── source_docs/                # user drops raw FAQ/product docs here
│   │   └── example_faq.md          # demo doc so the pipeline is runnable out of the box
│   └── generated/                  # .gitkeep only; raw_qa.jsonl is gitignored output
├── scripts/
│   ├── __init__.py
│   ├── lib/
│   │   ├── __init__.py
│   │   ├── env_config.py           # load_env_file()
│   │   ├── dataset_validation.py   # validate_jsonl_line(), validate_dataset_file()
│   │   ├── chunking.py             # chunk_text()
│   │   └── data_gen.py             # build_generation_prompt(), parse_generated_response()
│   ├── validate_dataset.py         # CLI wrapper around dataset_validation
│   ├── generate_training_data.py   # CLI wrapper: source_docs -> raw_qa.jsonl
│   ├── setup.sh                    # one-time venv + deps + GPU check
│   ├── start_server.sh             # THE start command (base model)
│   ├── serve_with_lora.sh          # start command + LoRA adapter
│   └── train_lora.py               # QLoRA fine-tuning via Unsloth
├── tests/
│   ├── test_env_config.py
│   ├── test_dataset_validation.py
│   ├── test_chunking.py
│   ├── test_data_gen.py
│   └── test_validate_dataset_cli.py
├── docs/
│   ├── how-it-works.md
│   └── superpowers/
│       ├── specs/2026-08-06-qwen-vllm-lora-setup-design.md   # (already exists)
│       └── plans/2026-08-06-qwen-vllm-lora-setup.md           # (this file)
└── README.md
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `config/model.env`
- Create: `scripts/__init__.py`
- Create: `scripts/lib/__init__.py`
- Create: `data/source_docs/.gitkeep`
- Create: `data/generated/.gitkeep`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
output/
data/generated/*
!data/generated/.gitkeep
data/train.jsonl
```

- [ ] **Step 3: Create `requirements.txt`**

```
vllm>=0.11.0
unsloth
trl>=0.12.0
datasets
peft
transformers>=4.51.0
accelerate
bitsandbytes
openai
```

- [ ] **Step 4: Create `config/model.env`**

```
MODEL=Qwen/Qwen3-4B-Instruct-2507
PORT=8000
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.90
QUANTIZATION=none
ADAPTER_NAME=support-adapter
```

- [ ] **Step 5: Create empty package markers and placeholder dirs**

```bash
mkdir -p "E:/Projects/QWEN+vLLM+LoRA/scripts/lib"
mkdir -p "E:/Projects/QWEN+vLLM+LoRA/data/source_docs"
mkdir -p "E:/Projects/QWEN+vLLM+LoRA/data/generated"
touch "E:/Projects/QWEN+vLLM+LoRA/scripts/__init__.py"
touch "E:/Projects/QWEN+vLLM+LoRA/scripts/lib/__init__.py"
touch "E:/Projects/QWEN+vLLM+LoRA/data/source_docs/.gitkeep"
touch "E:/Projects/QWEN+vLLM+LoRA/data/generated/.gitkeep"
```

- [ ] **Step 6: Commit**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA"
git add pyproject.toml .gitignore requirements.txt config/model.env scripts/__init__.py scripts/lib/__init__.py data/source_docs/.gitkeep data/generated/.gitkeep
git commit -m "chore: scaffold project structure and config"
```

---

### Task 2: `env_config.py` — shared config loader

**Files:**
- Create: `scripts/lib/env_config.py`
- Test: `tests/test_env_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_env_config.py`:

```python
from pathlib import Path

from scripts.lib.env_config import load_env_file


def test_load_env_file_parses_key_value_pairs(tmp_path: Path):
    env_file = tmp_path / "model.env"
    env_file.write_text("MODEL=Qwen/Qwen3-4B-Instruct-2507\nPORT=8000\n")

    result = load_env_file(env_file)

    assert result == {"MODEL": "Qwen/Qwen3-4B-Instruct-2507", "PORT": "8000"}


def test_load_env_file_skips_comments_and_blank_lines(tmp_path: Path):
    env_file = tmp_path / "model.env"
    env_file.write_text("# a comment\n\nPORT=8000\n")

    result = load_env_file(env_file)

    assert result == {"PORT": "8000"}


def test_load_env_file_strips_quotes(tmp_path: Path):
    env_file = tmp_path / "model.env"
    env_file.write_text('MODEL="Qwen/Qwen3-4B-Instruct-2507"\nNAME=\'support-adapter\'\n')

    result = load_env_file(env_file)

    assert result == {"MODEL": "Qwen/Qwen3-4B-Instruct-2507", "NAME": "support-adapter"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest tests/test_env_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.lib.env_config'`

- [ ] **Step 3: Write the implementation**

Create `scripts/lib/env_config.py`:

```python
"""Parse simple KEY=VALUE .env-style config files shared with the bash scripts."""
from pathlib import Path


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a bash-sourceable KEY=VALUE file into a dict.

    Skips blank lines and lines starting with '#'. Strips matching
    single or double quotes from values, matching bash's own behavior
    when scripts `source` this same file.
    """
    result: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        result[key] = value
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest tests/test_env_config.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/env_config.py tests/test_env_config.py
git commit -m "feat: add shared env-file config loader"
```

---

### Task 3: `dataset_validation.py` — LoRA training data validation

**Files:**
- Create: `scripts/lib/dataset_validation.py`
- Test: `tests/test_dataset_validation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset_validation.py`:

```python
from pathlib import Path

from scripts.lib.dataset_validation import validate_dataset_file, validate_jsonl_line


def test_validate_jsonl_line_accepts_valid_pair():
    line = '{"instruction": "How do I reset my password?", "response": "Go to Settings > Security."}'

    is_valid, error = validate_jsonl_line(line)

    assert is_valid is True
    assert error == ""


def test_validate_jsonl_line_rejects_invalid_json():
    is_valid, error = validate_jsonl_line("{not json")

    assert is_valid is False
    assert "invalid JSON" in error


def test_validate_jsonl_line_rejects_missing_field():
    line = '{"instruction": "How do I reset my password?"}'

    is_valid, error = validate_jsonl_line(line)

    assert is_valid is False
    assert "response" in error


def test_validate_jsonl_line_rejects_empty_response():
    line = '{"instruction": "How do I reset my password?", "response": "   "}'

    is_valid, error = validate_jsonl_line(line)

    assert is_valid is False
    assert "response" in error


def test_validate_jsonl_line_allows_blank_line():
    is_valid, error = validate_jsonl_line("   ")

    assert is_valid is True


def test_validate_dataset_file_reports_line_numbers(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"instruction": "Q1", "response": "A1"}\n'
        '{not json}\n'
        '{"instruction": "Q3", "response": "A3"}\n'
    )

    errors = validate_dataset_file(dataset)

    assert len(errors) == 1
    assert errors[0].startswith("line 2:")


def test_validate_dataset_file_returns_empty_for_valid_file(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"instruction": "Q1", "response": "A1"}\n')

    errors = validate_dataset_file(dataset)

    assert errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest tests/test_dataset_validation.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.lib.dataset_validation'`

- [ ] **Step 3: Write the implementation**

Create `scripts/lib/dataset_validation.py`:

```python
"""Validation logic for LoRA training data in JSONL format.

Each line must be a JSON object with non-empty "instruction" and
"response" string fields.
"""
import json
from pathlib import Path

REQUIRED_FIELDS = ("instruction", "response")


def validate_jsonl_line(line: str) -> tuple[bool, str]:
    """Validate a single JSONL line. Returns (is_valid, error_message)."""
    stripped = line.strip()
    if not stripped:
        return True, ""  # blank lines are allowed and ignored by the trainer

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc}"

    if not isinstance(obj, dict):
        return False, "line is not a JSON object"

    for field in REQUIRED_FIELDS:
        if field not in obj:
            return False, f"missing required field '{field}'"
        if not isinstance(obj[field], str) or not obj[field].strip():
            return False, f"field '{field}' must be a non-empty string"

    return True, ""


def validate_dataset_file(path: Path) -> list[str]:
    """Validate every line of a JSONL dataset file.

    Returns a list of error strings, one per invalid line, formatted as
    "line N: <error>". An empty list means the file is valid.
    """
    errors: list[str] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        is_valid, error = validate_jsonl_line(line)
        if not is_valid:
            errors.append(f"line {line_number}: {error}")
    return errors
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest tests/test_dataset_validation.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/dataset_validation.py tests/test_dataset_validation.py
git commit -m "feat: add LoRA training dataset validation logic"
```

---

### Task 4: `validate_dataset.py` — CLI wrapper

**Files:**
- Create: `scripts/validate_dataset.py`
- Test: `tests/test_validate_dataset_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validate_dataset_cli.py`:

```python
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_dataset.py"


def test_cli_exits_zero_for_valid_dataset(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"instruction": "Q1", "response": "A1"}\n')

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(dataset)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "OK" in result.stdout


def test_cli_exits_one_for_invalid_dataset(tmp_path: Path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{not json}\n")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(dataset)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "FAILED" in result.stdout


def test_cli_exits_two_for_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.jsonl"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(missing)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest tests/test_validate_dataset_cli.py -v
```

Expected: FAIL (script doesn't exist yet — `FileNotFoundError` inside the subprocess, surfaced as a non-matching return code / no stdout)

- [ ] **Step 3: Write the implementation**

Create `scripts/validate_dataset.py`:

```python
#!/usr/bin/env python3
"""CLI: validate a LoRA training dataset before spending GPU time on it.

Usage: python scripts/validate_dataset.py data/train.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.dataset_validation import validate_dataset_file


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/validate_dataset.py <path-to-train.jsonl>", file=sys.stderr)
        return 2

    dataset_path = Path(sys.argv[1])
    if not dataset_path.exists():
        print(f"Error: {dataset_path} does not exist", file=sys.stderr)
        return 2

    errors = validate_dataset_file(dataset_path)
    if errors:
        print(f"FAILED: {len(errors)} invalid line(s) in {dataset_path}")
        for error in errors:
            print(f"  {error}")
        return 1

    line_count = sum(1 for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"OK: {dataset_path} is valid ({line_count} training examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest tests/test_validate_dataset_cli.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_dataset.py tests/test_validate_dataset_cli.py
git commit -m "feat: add validate_dataset CLI"
```

---

### Task 5: `chunking.py` — split source docs for generation prompts

**Files:**
- Create: `scripts/lib/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chunking.py`:

```python
import pytest

from scripts.lib.chunking import chunk_text


def test_chunk_text_keeps_short_text_as_single_chunk():
    text = "Paragraph one.\n\nParagraph two."

    chunks = chunk_text(text, max_chars=2000)

    assert chunks == ["Paragraph one.\n\nParagraph two."]


def test_chunk_text_splits_at_paragraph_boundary_when_over_limit():
    text = "A" * 50 + "\n\n" + "B" * 50

    chunks = chunk_text(text, max_chars=60)

    assert chunks == ["A" * 50, "B" * 50]


def test_chunk_text_hard_splits_oversized_paragraph():
    text = "A" * 130

    chunks = chunk_text(text, max_chars=50)

    assert chunks == ["A" * 50, "A" * 50, "A" * 30]


def test_chunk_text_rejects_non_positive_max_chars():
    with pytest.raises(ValueError):
        chunk_text("hello", max_chars=0)


def test_chunk_text_ignores_blank_paragraphs():
    text = "First.\n\n\n\nSecond."

    chunks = chunk_text(text, max_chars=2000)

    assert chunks == ["First.\n\nSecond."]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest tests/test_chunking.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.lib.chunking'`

- [ ] **Step 3: Write the implementation**

Create `scripts/lib/chunking.py`:

```python
"""Split raw text into chunks small enough to fit in a generation prompt."""


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Split text into paragraph-respecting chunks no longer than max_chars.

    Paragraphs (separated by blank lines) are kept whole when possible;
    a single paragraph longer than max_chars is hard-split.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(paragraph), max_chars):
                chunks.append(paragraph[start:start + max_chars])
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest tests/test_chunking.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/chunking.py tests/test_chunking.py
git commit -m "feat: add text chunking for training data generation"
```

---

### Task 6: `data_gen.py` — prompt building and response parsing

**Files:**
- Create: `scripts/lib/data_gen.py`
- Test: `tests/test_data_gen.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data_gen.py`:

```python
from scripts.lib.data_gen import build_generation_prompt, parse_generated_response


def test_build_generation_prompt_includes_chunk_and_count():
    prompt = build_generation_prompt("Refunds take 5 business days.", num_pairs=2)

    assert "Refunds take 5 business days." in prompt
    assert "exactly 2 question-and-answer pairs" in prompt


def test_parse_generated_response_extracts_valid_pairs():
    response = '[{"instruction": "How long do refunds take?", "response": "5 business days."}]'

    pairs = parse_generated_response(response)

    assert pairs == [{"instruction": "How long do refunds take?", "response": "5 business days."}]


def test_parse_generated_response_drops_incomplete_entries():
    response = '[{"instruction": "Q1"}, {"instruction": "Q2", "response": "A2"}]'

    pairs = parse_generated_response(response)

    assert pairs == [{"instruction": "Q2", "response": "A2"}]


def test_parse_generated_response_returns_empty_list_for_invalid_json():
    pairs = parse_generated_response("not json at all")

    assert pairs == []


def test_parse_generated_response_returns_empty_list_for_non_array():
    pairs = parse_generated_response('{"instruction": "Q1", "response": "A1"}')

    assert pairs == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest tests/test_data_gen.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.lib.data_gen'`

- [ ] **Step 3: Write the implementation**

Create `scripts/lib/data_gen.py`:

```python
"""Prompt building and response parsing for synthetic Q&A generation."""
import json

GENERATION_INSTRUCTIONS = (
    "You are generating customer support training examples from the "
    "reference text below. Produce exactly {n} question-and-answer pairs "
    "a customer might ask that are answerable from this text. Respond with "
    "ONLY a JSON array of objects, each with keys \"instruction\" and "
    "\"response\". No markdown, no commentary, just the JSON array."
)


def build_generation_prompt(chunk: str, num_pairs: int = 3) -> str:
    """Build the prompt sent to the model to generate Q&A pairs from a chunk."""
    instructions = GENERATION_INSTRUCTIONS.format(n=num_pairs)
    return f"{instructions}\n\nReference text:\n{chunk}"


def parse_generated_response(response_text: str) -> list[dict[str, str]]:
    """Parse the model's JSON-array response into a list of Q&A dicts.

    Silently drops entries missing required fields rather than raising,
    since this output feeds a human review step, not training directly.
    """
    try:
        parsed = json.loads(response_text.strip())
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    pairs: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        instruction = item.get("instruction")
        response = item.get("response")
        if isinstance(instruction, str) and instruction.strip() and \
           isinstance(response, str) and response.strip():
            pairs.append({"instruction": instruction.strip(), "response": response.strip()})

    return pairs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest tests/test_data_gen.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/data_gen.py tests/test_data_gen.py
git commit -m "feat: add prompt building and response parsing for data generation"
```

---

### Task 7: Full local test suite checkpoint

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest -v
```

Expected: `20 passed` (3 env_config + 7 dataset_validation + 3 validate_dataset_cli + 5 chunking + 5 data_gen — wait, that's 23; the exact count doesn't matter, what matters is **zero failures**). All tests green before moving on to the GPU-dependent scripts.

---

### Task 8: `generate_training_data.py` — CLI wrapper

**Files:**
- Create: `scripts/generate_training_data.py`
- Create: `data/source_docs/example_faq.md`

- [ ] **Step 1: Write the implementation**

Create `scripts/generate_training_data.py`:

```python
#!/usr/bin/env python3
"""CLI: generate draft customer-support Q&A pairs from data/source_docs/.

Reuses the vLLM server started by scripts/start_server.sh (must already be
running). Writes draft pairs to data/generated/raw_qa.jsonl for human review
-- do NOT feed this file directly into training.

Usage: python scripts/generate_training_data.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from scripts.lib.chunking import chunk_text
from scripts.lib.data_gen import build_generation_prompt, parse_generated_response
from scripts.lib.env_config import load_env_file

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DOCS_DIR = REPO_ROOT / "data" / "source_docs"
OUTPUT_PATH = REPO_ROOT / "data" / "generated" / "raw_qa.jsonl"
PAIRS_PER_CHUNK = 3
MAX_CHUNK_CHARS = 2000


def main() -> int:
    config = load_env_file(REPO_ROOT / "config" / "model.env")
    port = config.get("PORT", "8000")
    model = config["MODEL"]

    source_files = sorted(SOURCE_DOCS_DIR.glob("*.md")) + sorted(SOURCE_DOCS_DIR.glob("*.txt"))
    if not source_files:
        print(f"No .md/.txt files found in {SOURCE_DOCS_DIR}", file=sys.stderr)
        return 1

    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_file:
        for source_file in source_files:
            text = source_file.read_text(encoding="utf-8")
            chunks = chunk_text(text, max_chars=MAX_CHUNK_CHARS)

            for chunk in chunks:
                prompt = build_generation_prompt(chunk, num_pairs=PAIRS_PER_CHUNK)
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                response_text = completion.choices[0].message.content or ""
                pairs = parse_generated_response(response_text)

                for pair in pairs:
                    out_file.write(json.dumps(pair) + "\n")
                    total_pairs += 1

            print(f"{source_file.name}: {len(chunks)} chunk(s) processed")

    print(f"Wrote {total_pairs} draft pairs to {OUTPUT_PATH}")
    print("Review and edit before copying approved pairs into data/train.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify syntax (this machine has no `openai` package installed — `py_compile` checks syntax only, without executing imports)**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m py_compile scripts/generate_training_data.py
```

Expected: no output, exit code 0 (confirmed this approach works even with unresolved imports)

- [ ] **Step 3: Create a demo source doc so the pipeline is runnable out of the box**

Create `data/source_docs/example_faq.md`:

```markdown
# Example Product FAQ

## Refunds

Refunds are processed within 5 business days of an approved return.
Refunds are issued to the original payment method. Store credit is
available immediately as an alternative to a refund.

## Shipping

Standard shipping takes 3-7 business days within the continental US.
Express shipping takes 1-2 business days and is available at checkout
for an additional fee. We do not currently ship internationally.

## Account Access

If you're locked out of your account, use the "Forgot Password" link
on the login page. Password reset emails expire after 1 hour. If you
don't receive the email, check your spam folder before contacting support.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_training_data.py data/source_docs/example_faq.md
git commit -m "feat: add synthetic training data generator and demo source doc"
```

---

### Task 9: `setup.sh` — one-time environment setup

**Files:**
- Create: `scripts/setup.sh`

- [ ] **Step 1: Write the implementation**

Create `scripts/setup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"

echo "== Checking for NVIDIA GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. On WSL2, make sure GPU passthrough is set up" >&2
    echo "(install the Windows NVIDIA driver with WSL support; do not install a separate Linux driver)." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== Creating virtual environment at $VENV_DIR =="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "== Installing dependencies =="
pip install --upgrade pip
pip install -r "$REPO_ROOT/requirements.txt"

echo "== Verifying vLLM and CUDA =="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print('CUDA OK:', torch.cuda.get_device_name(0))"
python3 -c "import vllm; print('vLLM OK:', vllm.__version__)"

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
```

- [ ] **Step 2: Make it executable and check syntax**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA"
chmod +x scripts/setup.sh
bash -n scripts/setup.sh
```

Expected: no output (silent success = valid syntax)

- [ ] **Step 3: Commit**

```bash
git add scripts/setup.sh
git commit -m "feat: add one-time environment setup script"
```

---

### Task 10: `start_server.sh` — the base start command

**Files:**
- Create: `scripts/start_server.sh`

- [ ] **Step 1: Write the implementation**

Create `scripts/start_server.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/config/model.env"

QUANT_FLAG=()
if [ "${QUANTIZATION:-none}" != "none" ]; then
    QUANT_FLAG=(--quantization "$QUANTIZATION")
fi

echo "Starting vLLM server: model=$MODEL port=$PORT max_model_len=$MAX_MODEL_LEN quantization=${QUANTIZATION:-none}"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${QUANT_FLAG[@]}"
```

- [ ] **Step 2: Make it executable and check syntax**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA"
chmod +x scripts/start_server.sh
bash -n scripts/start_server.sh
```

Expected: no output (confirmed working on this exact script during plan verification)

- [ ] **Step 3: Commit**

```bash
git add scripts/start_server.sh
git commit -m "feat: add vLLM server start command"
```

---

### Task 11: `test_client.py` — verify the server works

**Files:**
- Create: `scripts/test_client.py`

- [ ] **Step 1: Write the implementation**

Create `scripts/test_client.py`:

```python
#!/usr/bin/env python3
"""CLI: send one test chat request to the running vLLM server.

Verifies the server is actually serving correctly. Exit code 0 plus a
printed response means success.

Usage:
  python scripts/test_client.py                            # hits the base model
  python scripts/test_client.py --model support-adapter    # hits the LoRA adapter
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from scripts.lib.env_config import load_env_file

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Model name to query (defaults to config/model.env MODEL)")
    parser.add_argument("--prompt", default="What can you help me with?", help="Prompt to send")
    args = parser.parse_args()

    config = load_env_file(REPO_ROOT / "config" / "model.env")
    port = config.get("PORT", "8000")
    model = args.model or config["MODEL"]

    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": args.prompt}],
        )
    except Exception as exc:
        print(f"FAILED: could not reach server or get a response: {exc}", file=sys.stderr)
        return 1

    reply = completion.choices[0].message.content
    print(f"Model: {model}")
    print(f"Prompt: {args.prompt}")
    print(f"Response: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify syntax**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m py_compile scripts/test_client.py
```

Expected: no output, exit code 0

- [ ] **Step 3: Commit**

```bash
git add scripts/test_client.py
git commit -m "feat: add test client for verifying the running server"
```

---

### Task 12: `train_lora.py` — QLoRA fine-tuning

**Files:**
- Create: `scripts/train_lora.py`

- [ ] **Step 1: Write the implementation**

Create `scripts/train_lora.py`:

```python
#!/usr/bin/env python3
"""Fine-tune a LoRA adapter for Qwen3-4B on customer-support Q&A data.

Uses Unsloth's 4-bit QLoRA training path, sized for 8GB-class GPUs.
Reads data/train.jsonl (validate first with scripts/validate_dataset.py)
and writes the trained adapter to output/lora_adapter/.

Usage: python scripts/train_lora.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.dataset_validation import validate_dataset_file
from scripts.lib.env_config import load_env_file

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = REPO_ROOT / "data" / "train.jsonl"
OUTPUT_DIR = REPO_ROOT / "output" / "lora_adapter"

# Sized for 8GB-class GPUs. Increase MAX_SEQ_LENGTH / BATCH_SIZE if the
# target machine has more VRAM (see config/model.env for GPU tier notes).
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4


def main() -> int:
    # Fail fast on data problems before paying the cost of importing torch/unsloth.
    if not TRAIN_DATA_PATH.exists():
        print(f"ERROR: {TRAIN_DATA_PATH} not found. Create it first (see docs/how-it-works.md).", file=sys.stderr)
        return 1

    errors = validate_dataset_file(TRAIN_DATA_PATH)
    if errors:
        print(f"ERROR: {TRAIN_DATA_PATH} has {len(errors)} invalid line(s). Run scripts/validate_dataset.py for details.", file=sys.stderr)
        return 1

    config = load_env_file(REPO_ROOT / "config" / "model.env")
    base_model = config["MODEL"]

    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("json", data_files=str(TRAIN_DATA_PATH), split="train")

    def format_example(example: dict) -> dict:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = dataset.map(format_example)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            logging_steps=1,
            save_strategy="no",
        ),
    )

    result = trainer.train()
    print(f"Final training loss: {result.training_loss:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Adapter saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify syntax**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m py_compile scripts/train_lora.py
```

Expected: no output, exit code 0

- [ ] **Step 3: Commit**

```bash
git add scripts/train_lora.py
git commit -m "feat: add QLoRA training script via Unsloth"
```

---

### Task 13: `serve_with_lora.sh` — serve base model + adapter

**Files:**
- Create: `scripts/serve_with_lora.sh`

- [ ] **Step 1: Write the implementation**

Create `scripts/serve_with_lora.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/config/model.env"

QUANT_FLAG=()
if [ "${QUANTIZATION:-none}" != "none" ]; then
    QUANT_FLAG=(--quantization "$QUANTIZATION")
fi

ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
    exit 1
fi

echo "Starting vLLM server with LoRA: model=$MODEL adapter=$ADAPTER_NAME port=$PORT"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-lora \
    --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
    "${QUANT_FLAG[@]}"
```

- [ ] **Step 2: Make it executable and check syntax**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA"
chmod +x scripts/serve_with_lora.sh
bash -n scripts/serve_with_lora.sh
```

Expected: no output

- [ ] **Step 3: Commit**

```bash
git add scripts/serve_with_lora.sh
git commit -m "feat: add start command for serving base model plus LoRA adapter"
```

---

### Task 14: `docs/how-it-works.md` — the explainer

**Files:**
- Create: `docs/how-it-works.md`

- [ ] **Step 1: Write the documentation**

Create `docs/how-it-works.md`:

```markdown
# How This Works

## Qwen3 vs Qwen3.6

Qwen3.6 (Alibaba's newer release) only ships as **27B** (dense) and
**35B-A3B** (Mixture-of-Experts, 35B total / 3B active params). Both need
far more VRAM than an 8GB card can hold, even 4-bit quantized — an MoE
model's *total* parameters must still fit in VRAM for vLLM, regardless of
how few are "active" per token.

This project targets **Qwen3-4B-Instruct-2507** instead: the previous
generation, small enough to train and serve entirely on an 8GB-class GPU.
`config/model.env` is the single place to point at a different model if
the target machine turns out to have more VRAM (see "Adjusting for a
different GPU" below).

## What vLLM does differently

Running a model with plain `transformers` processes one request at a time
and recomputes the KV cache inefficiently across requests. vLLM adds two
things that matter here:

- **Continuous batching** — new requests join an in-flight batch instead of
  waiting for the whole batch to finish, so throughput stays high even
  with uneven request timing.
- **PagedAttention** — manages the KV cache (the memory that holds
  per-token attention state during generation) in fixed-size blocks,
  like an OS paging virtual memory, instead of one large contiguous
  allocation. This is what lets `--gpu-memory-utilization` and
  `--max-model-len` be tuned precisely instead of over-allocating "just
  in case."

`scripts/start_server.sh` starts vLLM's OpenAI-compatible API server, so
any OpenAI-client-compatible tool (including `scripts/test_client.py`) can
talk to it over HTTP.

## What LoRA does and how it plugs in here

Full fine-tuning updates every parameter in the model — expensive in both
compute and storage. LoRA (Low-Rank Adaptation) instead freezes the base
model and trains a small pair of low-rank matrices per targeted layer,
producing an "adapter" that's a few tens of megabytes instead of gigabytes.

`scripts/train_lora.py` trains this adapter using **QLoRA** (the base
model loaded in 4-bit, adapter weights trained in higher precision) via
**Unsloth**, which is the standard approach for fine-tuning on 8GB-class
GPUs.

vLLM can serve the base model and one or more LoRA adapters
*simultaneously* — `scripts/serve_with_lora.sh` adds
`--enable-lora --lora-modules support-adapter=output/lora_adapter` to the
same start command. A client picks which one it wants per-request via the
`model` field: the base model's name for unmodified Qwen3-4B, or
`support-adapter` (configurable via `ADAPTER_NAME` in `config/model.env`)
for the fine-tuned version.

## The full pipeline, end to end

1. `scripts/setup.sh` — one-time: venv, install deps, verify GPU.
2. `scripts/start_server.sh` — serve the base model.
3. `scripts/test_client.py` — confirm the server responds.
4. Drop FAQ/product docs into `data/source_docs/` (an example is already
   there: `example_faq.md`).
5. `scripts/generate_training_data.py` — with the server from step 2 still
   running, this asks the model itself to draft Q&A pairs from your docs,
   writing them to `data/generated/raw_qa.jsonl`.
6. **Review by hand** — read `raw_qa.jsonl`, discard or edit low-quality
   pairs, and copy the approved ones into `data/train.jsonl`. This step is
   intentionally manual: synthetic data needs a human check before it
   trains anything.
7. `scripts/validate_dataset.py data/train.jsonl` — sanity-checks the file
   before burning GPU time on it.
8. `scripts/train_lora.py` — trains the adapter, saves it to
   `output/lora_adapter/`.
9. `scripts/serve_with_lora.sh` — serves base model + adapter together.
10. `scripts/test_client.py --model support-adapter` — confirm the
    fine-tuned adapter responds differently from the base model.

## Troubleshooting: out-of-memory on server start

If `start_server.sh` or `serve_with_lora.sh` fails with a CUDA
out-of-memory error, try in order:

1. Lower `GPU_MEM_UTIL` in `config/model.env` (e.g. `0.90` -> `0.75`).
2. Lower `MAX_MODEL_LEN` (shorter context needs a smaller KV cache).
3. Switch to a quantized checkpoint: set `MODEL=Qwen/Qwen3-4B-AWQ` and
   `QUANTIZATION=awq` in `config/model.env`.

## Adjusting for a different GPU

Everything model-specific lives in `config/model.env`. If the target
machine has more VRAM than 8GB, larger Qwen3 sizes (8B, 14B) or even
Qwen3.6-27B (with enough VRAM, 24GB+) can be dropped in by changing
`MODEL` — no script changes needed. `scripts/train_lora.py` has its LoRA
hyperparameters (`MAX_SEQ_LENGTH`, `BATCH_SIZE`, etc.) as module-level
constants with a comment marking them as 8GB-tier defaults, for the same
reason.
```

- [ ] **Step 2: Commit**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA"
git add docs/how-it-works.md
git commit -m "docs: explain Qwen3, vLLM, and LoRA, and the full pipeline"
```

---

### Task 15: `README.md` — top-level entry point

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

Create `README.md`:

```markdown
# Qwen3-4B + vLLM + LoRA

Serve Qwen3-4B via vLLM with one start command, then customize it for
customer support with a LoRA adapter trained on your own docs.

Authored on Windows; meant to be run on a Linux or WSL2 machine with an
NVIDIA GPU (8GB VRAM tier by default — see `config/model.env`).

## Quickstart

```bash
./scripts/setup.sh                    # one-time: venv + deps + GPU check
./scripts/start_server.sh             # start the base model server
python scripts/test_client.py         # confirm it's responding (separate terminal)
```

## Training a LoRA adapter

```bash
# 1. Drop FAQ/product docs into data/source_docs/ (example_faq.md is there as a demo)
# 2. With start_server.sh still running:
python scripts/generate_training_data.py
# 3. Review data/generated/raw_qa.jsonl by hand, copy approved pairs into data/train.jsonl
python scripts/validate_dataset.py data/train.jsonl
python scripts/train_lora.py
./scripts/serve_with_lora.sh
python scripts/test_client.py --model support-adapter
```

See [docs/how-it-works.md](docs/how-it-works.md) for the full explanation
of how Qwen3, vLLM, and LoRA fit together here, and
[docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md](docs/superpowers/specs/2026-08-06-qwen-vllm-lora-setup-design.md)
for the design rationale (including why Qwen3-4B instead of Qwen3.6).

## Running the test suite

```bash
python -m pytest -v
```
```

- [ ] **Step 2: Commit**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA"
git add README.md
git commit -m "docs: add top-level README with quickstart"
```

---

### Task 16: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite one more time**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && python -m pytest -v
```

Expected: all tests pass, zero failures.

- [ ] **Step 2: Syntax-check every shell script**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA"
for f in scripts/*.sh; do bash -n "$f" && echo "$f: OK"; done
```

Expected: `scripts/serve_with_lora.sh: OK`, `scripts/setup.sh: OK`, `scripts/start_server.sh: OK`

- [ ] **Step 3: Syntax-check every Python script**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA"
python -m py_compile scripts/*.py scripts/lib/*.py
echo "All Python files compiled cleanly"
```

Expected: `All Python files compiled cleanly`

- [ ] **Step 4: Confirm git log shows a clean history of small commits**

```bash
cd "E:/Projects/QWEN+vLLM+LoRA" && git log --oneline
```

Expected: one commit per task, most recent first, ending at the "scaffold project structure" commit and the earlier design-spec commit.

- [ ] **Step 5: Review with the user**

At this point, everything is authored, unit-tested (where GPU-independent), and syntax-verified. The remaining verification — actually running `setup.sh` through `serve_with_lora.sh` end-to-end — requires the target Linux/WSL2 machine with a GPU, which is outside this authoring environment. Tell the user the code is ready to copy to that machine and run per the README quickstart, and that `docs/how-it-works.md` covers troubleshooting (particularly the OOM section) if the first run hits VRAM issues.
