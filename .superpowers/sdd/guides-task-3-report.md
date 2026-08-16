# Task 3 Report: Link guides from README

## Status

**Complete.**

## Changes

- Added `## Guides` section after intro block (before Onboarding) with links to both guide files and short descriptions.
- Added the same two links to the Troubleshooting design-docs bullet list.

## Commit

- **Hash:** `ed39be0`
- **Message:** `docs: link learning guides from README`
- **Files committed:** `README.md` only

## Verification

```bash
test -f docs/guides/architecture-learning.md && test -f docs/guides/operator-cheatsheet.md && echo OK
# OK
```

## Spec coverage

| Spec item | Done |
|---|---|
| README Guides section near top | yes |
| Troubleshooting docs list links | yes |
| Paths `docs/guides/architecture-learning.md` | yes |
| Paths `docs/guides/operator-cheatsheet.md` | yes |
