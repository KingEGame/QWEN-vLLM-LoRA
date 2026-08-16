### Task 3: Link guides from README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: both guide paths
- Produces: discoverable entry from repo root README

- [ ] **Step 1: Insert a Guides section** near the top (after the opening paragraphs, before Onboarding) or in the docs list at the bottom — prefer **after the first intro block**:

```markdown
## Guides

- [Architecture learning](docs/guides/architecture-learning.md) — what we built, vLLM vs LoRA vs Qwen, limits
- [Operator cheatsheet](docs/guides/operator-cheatsheet.md) — commands by scenario
```

Also add the same two links to the Troubleshooting / design docs bullet list at the bottom of README if one exists.

- [ ] **Step 2: Verify links resolve**

```bash
test -f docs/guides/architecture-learning.md && test -f docs/guides/operator-cheatsheet.md && echo OK
```

- [ ] **Step 3: Commit** (if required)

```bash
git add README.md
git commit -m "docs: link learning guides from README"
```

---

## Spec coverage

| Spec item | Task |
|---|---|
| `docs/guides/architecture-learning.md` | 1 |
| `docs/guides/operator-cheatsheet.md` | 2 |
| README Guides links | 3 |
| Honest quality / % roles | 1 |
| Commands by scenario | 2 |
| Code line citations | 1–2 |
| No push / retrain in this work | constraints |

## Self-review notes

- No TBD placeholders
- Cheatsheet train knobs match `train_lora.py` env reads (not dead `TRAIN_GRAD_ACCUM` keys)
- Paths use `docs/guides/` as locked in the design
