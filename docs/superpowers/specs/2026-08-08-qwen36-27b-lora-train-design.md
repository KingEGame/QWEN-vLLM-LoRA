# Design: Qwen3.6-27B LoRA train (from example FAQ)

**Date:** 2026-08-08  
**Status:** approved  
**Depends on:** [2026-08-08-qwen36-27b-awq-serve-design.md](2026-08-08-qwen36-27b-awq-serve-design.md)

## Goal

Run the existing customer-support LoRA loop against **Qwen3.6-27B** on this
machine (24GB RTX 5090 Laptop / WSL2):

1. Generate draft Q&A from `data/source_docs/example_faq.md` using the running
   AWQ vLLM server.
2. Promote data into `data/train.jsonl` (light review; first run may auto-copy).
3. Stop the server, QLoRA-train a LoRA adapter on the **dense** base
   `Qwen/Qwen3.6-27B`.
4. Serve AWQ base + adapter via `serve_with_lora.sh`.

## Context

- Serving uses `shawnw3i/Qwen3.6-27B-AWQ-MTP` (compressed). QLoRA cannot train
  on that checkpoint; training must load dense `Qwen/Qwen3.6-27B` in 4-bit.
- `scripts/train_lora.py` already supports `TRAIN_MODEL` override and QLoRA via
  transformers + PEFT + bitsandbytes + TRL.
- There is no `data/train.jsonl` yet; only `example_faq.md`.
- User chose: generate from the example FAQ (option A), Approach 1 (full 27B
  QLoRA), design accepted 2026-08-08.

## Decision

| Item | Choice |
|---|---|
| Generation model | Running AWQ serve model (`MODEL` in `config/model.env`) |
| Train base | `TRAIN_MODEL=Qwen/Qwen3.6-27B` (dense) |
| Method | QLoRA 4-bit NF4, existing `train_lora.py` stack (not Unsloth-first) |
| First-run data | Auto-copy `data/generated/raw_qa.jsonl` → `data/train.jsonl` after validate; keep manual review as documented optional step |
| Train memory knobs | `MAX_SEQ_LENGTH=1024`, `BATCH_SIZE=1`, `GRADIENT_ACCUMULATION_STEPS` keep/raise as needed, `NUM_EPOCHS=1` for first smoke then 3 for real |
| Adapter out | `output/lora_adapter/` (unchanged) |
| Serve after train | AWQ `MODEL` + `--enable-lora` via `serve_with_lora.sh` |

## Pipeline

```
start_server.sh (AWQ 27B)
        │
        ▼
generate_training_data.py  →  data/generated/raw_qa.jsonl
        │
        ▼
promote (auto-copy first run)  →  data/train.jsonl
        │
        ▼
validate_dataset.py
        │
        ▼
stop server (free ~20GB VRAM)
        │
        ▼
TRAIN_MODEL=Qwen/Qwen3.6-27B train_lora.py  →  output/lora_adapter/
        │
        ▼
serve_with_lora.sh  (AWQ base + adapter)
        │
        ▼
test_client.py --model support-adapter
```

## File / config changes (expected)

```
config/model.env              # add TRAIN_MODEL=Qwen/Qwen3.6-27B
scripts/train_lora.py         # 27B-safe defaults / target modules if needed
scripts/serve_with_lora.sh    # reuse start_server flag pattern (reasoning, etc.) if missing
README.md                     # document generate → train → serve-with-lora for 27B
docs/... (this design)
```

Optional small helper (only if needed): `scripts/promote_generated_data.py` or a
documented `cp` step — prefer documented copy over new code unless validation
requires it.

### `config/model.env` additions

```env
TRAIN_MODEL=Qwen/Qwen3.6-27B
```

Serving keys stay on the AWQ checkpoint. Training always prefers `TRAIN_MODEL`.

### Training defaults for 24GB + 27B

| Knob | First smoke | Follow-up |
|---|---|---|
| `MAX_SEQ_LENGTH` | 1024 | 2048 if VRAM allows |
| `BATCH_SIZE` | 1 | 1–2 |
| `NUM_EPOCHS` | 1 | 3 |
| LoRA rank/alpha | 16/16 (existing) | unchanged unless OOM |

If Qwen3.6 module names differ from the current
`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` list, discover
trainable Linear names once and update `LORA_TARGET_MODULES` (or use a PEFT
regex / `all-linear` policy) so GDN-only layers are not required targets.

## Verification

1. Server up → `python scripts/generate_training_data.py` writes non-empty
   `raw_qa.jsonl`.
2. Promote → `python scripts/validate_dataset.py` passes on `train.jsonl`.
3. Server stopped; `nvidia-smi` shows free VRAM.
4. `TRAIN_MODEL=Qwen/Qwen3.6-27B MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 python scripts/train_lora.py`
   completes and writes `output/lora_adapter/`.
5. `./scripts/serve_with_lora.sh` starts; `python scripts/test_client.py --model support-adapter`
   returns a non-empty reply.

## Risks

- **VRAM:** 27B QLoRA on 24GB is tight; OOM → shorten seq length, ensure server
  is fully dead, enable more aggressive checkpointing if needed.
- **Architecture:** Qwen3.6 hybrid (GDN + attention) may not match Unsloth; stay
  on transformers/PEFT. Some modules may not accept LoRA — target attention/MLP
  projections that exist.
- **Download:** Dense `Qwen/Qwen3.6-27B` is a large first-time download (~50GB+
  before 4-bit runtime load).
- **Data quality:** Auto-copy is fine for a first end-to-end proof; real support
  use still wants human review of `raw_qa.jsonl`.
- **Serve+LoRA:** Adapter trained on dense base must load against AWQ serve
  weights in vLLM; if incompatible, document fallback (serve dense quantized or
  merge — out of scope for v1 unless smoke fails).

## Out of scope

- Unsloth-primary rewrite
- Multi-GPU / DeepSpeed
- Training directly on the AWQ weights
- Replacing the example FAQ with real product corpora (later)
- MTP speculative decoding during serve-with-lora
