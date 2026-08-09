### Task 1: Privacy paths + personal data scaffold

**Files:**
- Modify: `.gitignore`
- Create: `data/personal/candidates/.gitkeep`
- Create: `data/personal/README.md`
- Create: `config/personal_sources.env`

**Interfaces:**
- Consumes: design privacy rule (local-only personal JSONL)
- Produces: ignored `data/personal/**/*.jsonl`; documented source config keys `AGENT_TRANSCRIPTS_DIR`, `MARKDOWN_GLOBS`

- [ ] **Step 1: Update `.gitignore`**

Append:

```gitignore
# Personal pipeline datasets (may contain private chat text)
data/personal/**/*.jsonl
!data/personal/candidates/.gitkeep
```

Keep existing `data/train.jsonl` / `data/generated/*` rules.

- [ ] **Step 2: Create scaffold files**

`data/personal/candidates/.gitkeep` — empty file.

`data/personal/README.md`:

```markdown
# Personal tech datasets

1. Run `python scripts/extract_personal_candidates.py`
2. Review/edit files under `candidates/`
3. Promote: `python scripts/promote_personal_data.py --reviewed`
4. Train each adapter with `scripts/train_lora.py --data ... --output ...`
```

`config/personal_sources.env`:

```env
# Absolute or ~ paths OK. Override with env vars of the same name.
AGENT_TRANSCRIPTS_DIR=~/.cursor/projects/c-Users-supre-Documents-QWEN-vLLM-LoRA/agent-transcripts
MARKDOWN_GLOBS=docs/superpowers/specs/*.md,docs/superpowers/plans/*.md,README.md
```

- [ ] **Step 3: Verify ignore**

```bash
git check-ignore -v data/personal/candidates/foo.jsonl
```

Expected: a matching `.gitignore` rule line.

- [ ] **Step 4: Commit** (only if user/execution skill requests)

```bash
git add .gitignore data/personal/candidates/.gitkeep data/personal/README.md config/personal_sources.env
git commit -m "chore: scaffold personal pipeline data paths"
```

---

