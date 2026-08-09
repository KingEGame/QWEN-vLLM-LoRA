### Task 4: Train script `--data` / `--output`

**Files:**
- Modify: `scripts/train_lora.py`

**Interfaces:**
- Consumes: CLI `--data` (Path), `--output` (Path); env `TRAIN_DATA`, `TRAIN_OUTPUT` as overrides
- Produces: adapter written to `--output` (default remains `output/lora_adapter`)

- [ ] **Step 1: Add argparse at top of `main()`**

Replace fixed `TRAIN_DATA_PATH` / `OUTPUT_DIR` usage with:

```python
import argparse

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Train JSONL path (default data/train.jsonl)")
    parser.add_argument("--output", default=None, help="Adapter output dir (default output/lora_adapter)")
    args = parser.parse_args()

    train_data = Path(
        args.data
        or os.environ.get("TRAIN_DATA")
        or (REPO_ROOT / "data" / "train.jsonl")
    )
    output_dir = Path(
        args.output
        or os.environ.get("TRAIN_OUTPUT")
        or (REPO_ROOT / "output" / "lora_adapter")
    )
```

Then replace every `TRAIN_DATA_PATH` → `train_data` and `OUTPUT_DIR` → `output_dir` in `main()`.

Keep module-level constants for backward-compatible imports if tests reference them, or update tests if any break.

- [ ] **Step 2: Verify help + dry path check**

```bash
python scripts/train_lora.py --help
```

Expected: shows `--data` and `--output`.

```bash
python scripts/train_lora.py --data /no/such.jsonl --output /tmp/x; echo EXIT:$?
```

Expected: ERROR about missing file, non-zero exit (no GPU load).

- [ ] **Step 3: Commit** (if requested)

```bash
git add scripts/train_lora.py
git commit -m "feat: train_lora accepts --data and --output paths"
```

---

