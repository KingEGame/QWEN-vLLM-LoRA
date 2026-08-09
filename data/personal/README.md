# Personal tech datasets

1. Run `python scripts/extract_personal_candidates.py`
2. Review/edit files under `candidates/`
3. Promote: `python scripts/promote_personal_data.py --reviewed`
4. Train each adapter with `scripts/train_lora.py --data ... --output ...`
