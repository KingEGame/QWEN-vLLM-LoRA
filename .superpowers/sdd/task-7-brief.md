### Task 7: README + end-to-end operator path

**Files:**
- Modify: `README.md`
- Optionally link the new design spec under Troubleshooting / docs list

**Interfaces:**
- Consumes: all prior tasks’ CLIs
- Produces: documented personal loop operators can follow

- [ ] **Step 1: Add README section** after the FAQ LoRA section

Insert a section titled `## Personal tech pipeline (question-sharper → me-assistant)` that documents:

1. `python scripts/extract_personal_candidates.py`
2. Review under `data/personal/candidates/`, then `python scripts/promote_personal_data.py --reviewed`
3. Validate both `data/personal/*.jsonl` files
4. Stop vLLM; train with `--data` / `--output` for each adapter path
5. Serve with `LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant"`
6. `python scripts/personal_pipeline.py "..."`

Link `config/personal_sources.env` and
`docs/superpowers/specs/2026-08-08-personal-tech-pipeline-design.md`.

Also add the design link to the docs bullet list.

- [ ] **Step 2: Run full unit suite**

```bash
python -m pytest -v
```

Expected: all existing + new tests PASS.

- [ ] **Step 3: Commit** (if requested)

```bash
git add README.md
git commit -m "docs: document personal tech LoRA pipeline"
```

---

