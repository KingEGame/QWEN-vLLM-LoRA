# LoRA, QLoRA, and AWQ learning

Companions: [Architecture learning](architecture-learning.md) and
[Operator cheatsheet](operator-cheatsheet.md).

This guide explains what is trained in this repository, why the dense and AWQ
checkpoints are both present, and how an adapter trained with QLoRA can be used
with the AWQ serving model.

## The short version

This repository uses three different techniques for three different jobs:

| Technique | Job in this repo | Changes the base weights? |
|---|---|---|
| **LoRA** | Defines the small trainable adapter matrices | No |
| **QLoRA** | Makes LoRA training fit by loading the frozen dense base in 4-bit NF4 | No |
| **AWQ** | Makes inference fit in 24GB VRAM | No training occurs |

The resulting path is:

```text
Dense Qwen checkpoint (BF16 files)
        |
        | load frozen base as 4-bit NF4 for training
        v
QLoRA training -> LoRA adapter weights (~159 MB each)
        |
        | attach at inference; do not merge into the AWQ files
        v
AWQ Qwen checkpoint + selected LoRA -> vLLM OpenAI-compatible API
```

QLoRA is the training procedure. Its output is still an ordinary PEFT LoRA
adapter, not a special "QLoRA adapter" format.

## What is actually used here

The training model is configured as `Qwen/Qwen3.6-27B`. Its local model
metadata declares BF16 (`bfloat16`) text weights. It is loaded by
[`train_lora.py`](../../scripts/train_lora.py) through bitsandbytes with:

- 4-bit NF4 base-weight loading;
- double quantization;
- BF16 computation;
- rank 16 LoRA, alpha 16;
- adapters on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  and `down_proj`;
- only the LoRA parameters trainable; the base parameters remain frozen.

The serving model is `shawnw3i/Qwen3.6-27B-AWQ-MTP`, configured in
[`config/model.env`](../../config/model.env). Its metadata declares 4-bit AWQ
with group size 128. vLLM loads that checkpoint and applies one or more LoRA
adapters through [`serve_with_lora.sh`](../../scripts/serve_with_lora.sh).

The personal adapter roles are:

| Adapter | Training data | Intended behavior |
|---|---|---|
| `question-sharper` | `data/personal/question_sharp.jsonl` | Turn a messy request into one concise, precise question |
| `me-assistant` | `data/personal/me_assistant.jsonl` | Answer using the reviewed preferences and examples in the personal data |
| `support-adapter` | `data/train.jsonl` | Tiny FAQ end-to-end smoke test |

LoRA does not create a classifier in the strict sense unless it is trained
with labels and a classification head. The `question-sharper` here is a
generative transformation adapter: text in, rewritten question out.

## Memory: disk, RAM, and VRAM are different

The dense cache occupying about 65GB does not mean that 65GB is placed in VRAM.
On the audited machine:

- GPU: RTX 5090 Laptop GPU with 24,463 MiB VRAM;
- physical system RAM: about 63.5 GiB;
- WSL allocation without a custom `.wslconfig`: 31 GiB RAM plus 8 GiB swap;
- dense Hugging Face cache: about 65GB including 12.54 GiB of stale partial
  download files;
- AWQ Hugging Face cache: about 19GB.

A full BF16 load of a 27B model needs roughly 54GB just for parameter values,
before activations and training overhead, so it cannot fit directly in 24GB
VRAM. QLoRA reduces the frozen base to approximately 4 bits per parameter,
then adds quantization metadata, temporary activations, LoRA weights, and
optimizer state. That is why batch size 1, sequence length 1024, gradient
accumulation, and gradient checkpointing make this training feasible on the
24GB GPU.

The current machine has already completed these smoke training runs, which is
the strongest practical evidence that the configured path fits. Longer
sequences, larger batches, higher LoRA rank, or other running GPU processes can
still cause an out-of-memory error.

## Why train against dense BF16 but serve against AWQ?

The dense checkpoint is the clean training source. QLoRA temporarily quantizes
its frozen weights to NF4, a training-oriented 4-bit representation supported
by bitsandbytes. The gradients update only the LoRA matrices.

AWQ is an inference-oriented 4-bit checkpoint. It is efficient for vLLM but is
not the checkpoint used by this training script. At serving time, vLLM keeps
the AWQ base quantized and applies the floating-point LoRA delta during each
forward pass. It does not merge the adapter permanently into the AWQ files.

This is compatible when the dense source and AWQ checkpoint have the same base
architecture, layer names, tensor shapes, tokenizer behavior, and underlying
pretrained weights. vLLM currently accepts and serves both personal adapters,
which confirms structural compatibility. Quantization can still introduce a
small quality difference, so behavioral evaluation remains necessary.

For maximum reproducibility, pin both model revisions. If a permanently merged
AWQ model is required, the safer sequence is:

1. merge the LoRA into a dense copy of the matching base;
2. test the merged dense model;
3. quantize that merged model to AWQ;
4. test the resulting AWQ checkpoint again.

Do not treat merging directly into already-quantized AWQ tensors as the normal
path.

## Why this design was chosen

| Choice | Reason |
|---|---|
| LoRA instead of full fine-tuning | Each task stores about 159MB rather than another complete 27B model, and optimizer memory stays manageable |
| QLoRA instead of ordinary BF16 LoRA training | A full BF16 27B base cannot fit in 24GB VRAM |
| Dense source instead of training the AWQ checkpoint | NF4/bitsandbytes is designed for adapter training; AWQ is designed primarily for inference |
| Separate adapters instead of one combined adapter | The sharper and assistant have different jobs and can be selected, replaced, and evaluated independently |
| AWQ serving instead of dense serving | The approximately 19GB checkpoint fits the 24GB GPU and leaves limited space for KV cache and LoRA execution |
| vLLM for serving | It exposes the OpenAI-compatible API, manages KV cache, and supports multiple named LoRA modules |

The biggest current limitation is not the quantization method. It is the
personal dataset: many question-sharper rows copy the input rather than produce
a meaningfully sharper question.

## Alternatives and their tradeoffs

| Alternative | When it is useful | Why it is not the current default |
|---|---|---|
| Full BF16 fine-tuning | Maximum control with multi-GPU/server hardware | Far beyond 24GB VRAM and creates very large optimizer/checkpoint state |
| Ordinary BF16 LoRA | More direct training without a quantized base | The frozen 27B BF16 base alone is too large for this GPU |
| 8-bit LoRA | Possible quality/memory middle ground | Likely too tight after activations and training overhead on 24GB |
| Train a 4B-8B model | Fast iteration and much lower memory use | Lower base capability, although it may be the best data-development model |
| Merge LoRA and re-quantize | One standalone AWQ checkpoint for deployment | Must repeat quantization for every adapter/version and loses dynamic switching |
| Cloud or multi-GPU training | Larger batches, context, or full fine-tuning | Additional cost and data/privacy considerations |
| One combined personal adapter | Simpler serving | Task interference makes sharper and answer behavior harder to diagnose |
| Prompting or retrieval without training | Rapid updates and factual grounding | Does not learn the same persistent response transformation or personal style |

For this laptop, the present QLoRA-training plus AWQ-serving split is a sound
engineering choice. A smaller model is the strongest alternative while the
dataset is being cleaned because it shortens the experiment cycle.

## Local weight and adapter locations

The base checkpoints live in the WSL Hugging Face cache:

```text
# Dense training source
/home/supre/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/

# AWQ serving source
/home/supre/.cache/huggingface/hub/models--shawnw3i--Qwen3.6-27B-AWQ-MTP/
```

The learned adapter weights live in the repository:

```text
output/lora_adapter/adapter_model.safetensors
output/lora_question_sharper/adapter_model.safetensors
output/lora_me_assistant/adapter_model.safetensors
```

Keep each `adapter_config.json` with its corresponding `.safetensors` file.

## Incomplete Hugging Face downloads

The audited dense-model cache contains 17 `*.incomplete` files totaling 12.54
GiB. The completed snapshot has all 15 weight shards and no dangling links, and
training has already succeeded. Therefore these fragments are not required by
the current snapshot and no resume is needed.

Preview only:

```bash
find ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/blobs \
  -type f -name '*.incomplete' -print
```

Delete only those partial files after stopping download processes:

```bash
find ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/blobs \
  -type f -name '*.incomplete' -delete
```

This narrowly scoped command preserves the completed blobs and snapshot. Do
not delete the whole `models--Qwen--Qwen3.6-27B` directory if more training is
planned. If a future snapshot is genuinely incomplete, rerun `hf download
Qwen/Qwen3.6-27B`; Hugging Face will reuse completed cache content and download
what is missing.

## Training configuration warning

[`train_lora.py`](../../scripts/train_lora.py) reads `MAX_SEQ_LENGTH`,
`BATCH_SIZE`, `NUM_EPOCHS`, `GRADIENT_ACCUMULATION_STEPS`, and
`GRADIENT_CHECKPOINTING`. Similarly named `TRAIN_*` resource keys in
[`config/model.env`](../../config/model.env) are not currently wired, except
for `TRAIN_MODEL`. Use the exact environment names from the operator
cheatsheet until the configuration is consolidated.
