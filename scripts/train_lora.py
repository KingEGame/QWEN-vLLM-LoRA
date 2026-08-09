#!/usr/bin/env python3
"""Fine-tune a LoRA adapter for Qwen3 / Qwen3.6-27B on customer-support Q&A data.

Uses 4-bit QLoRA via transformers + PEFT + bitsandbytes + TRL, sized for
consumer GPUs. Reads data/train.jsonl (validate first with
scripts/validate_dataset.py) and writes the trained adapter to
output/lora_adapter/.

Usage:
  python scripts/train_lora.py

Serving may use an AWQ checkpoint while training needs the dense base. Set
TRAIN_MODEL in config/model.env or the environment (e.g.
TRAIN_MODEL=Qwen/Qwen3.6-27B for 27B QLoRA).

Resource / quality knobs (env overrides config/model.env TRAIN_* keys):
  MAX_SEQ_LENGTH, BATCH_SIZE, NUM_EPOCHS, GRADIENT_ACCUMULATION_STEPS,
  DATALOADER_NUM_WORKERS, TRAIN_RESOURCE_FRACTION (default 0.70 of free GPU),
  GRADIENT_CHECKPOINTING (1/0).

Example (quality-oriented on a free 24GB card):
  TRAIN_MODEL=Qwen/Qwen3.6-27B MAX_SEQ_LENGTH=1024 BATCH_SIZE=2 NUM_EPOCHS=3 \\
    GRADIENT_ACCUMULATION_STEPS=8 python scripts/train_lora.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.dataset_validation import validate_dataset_file
from scripts.lib.env_config import load_env_file

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = REPO_ROOT / "data" / "train.jsonl"
OUTPUT_DIR = REPO_ROOT / "output" / "lora_adapter"

LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4


def _cfg_get(config: dict[str, str], env_key: str, config_key: str | None = None, default: str = "") -> str:
    config_key = config_key or env_key
    return os.environ.get(env_key) or config.get(config_key) or default


def _resolve_int(config: dict[str, str], env_key: str, config_key: str, default: int) -> int:
    raw = _cfg_get(config, env_key, config_key, str(default))
    return int(raw)


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

    if not train_data.exists():
        print(f"ERROR: {train_data} not found. Create it first (see README).", file=sys.stderr)
        return 1

    errors = validate_dataset_file(train_data)
    if errors:
        print(
            f"ERROR: {train_data} has {len(errors)} invalid line(s). "
            "Run scripts/validate_dataset.py for details.",
            file=sys.stderr,
        )
        return 1

    config = load_env_file(REPO_ROOT / "config" / "model.env")
    # Serving may use an AWQ/compressed checkpoint; QLoRA training needs the
    # dense base instruct model. Override with TRAIN_MODEL when they differ.
    base_model = os.environ.get("TRAIN_MODEL") or config.get("TRAIN_MODEL") or config["MODEL"]

    is_27b = "27B" in base_model.upper()
    # Quality-oriented defaults for 27B on ~24GB; smaller models keep older defaults.
    default_seq = 1024 if is_27b else 2048
    default_batch = 2 if is_27b else 2
    default_epochs = 3 if is_27b else 3
    default_accum = 8 if is_27b else 4

    max_seq_length = _resolve_int(config, "MAX_SEQ_LENGTH", "TRAIN_MAX_SEQ_LENGTH", default_seq)
    batch_size = _resolve_int(config, "BATCH_SIZE", "TRAIN_BATCH_SIZE", default_batch)
    num_epochs = _resolve_int(config, "NUM_EPOCHS", "TRAIN_NUM_EPOCHS", default_epochs)
    grad_accum = _resolve_int(
        config, "GRADIENT_ACCUMULATION_STEPS", "TRAIN_GRAD_ACCUM", default_accum
    )
    resource_fraction = float(
        _cfg_get(config, "TRAIN_RESOURCE_FRACTION", "TRAIN_RESOURCE_FRACTION", "0.70")
    )
    grad_ckpt_raw = _cfg_get(config, "GRADIENT_CHECKPOINTING", "TRAIN_GRADIENT_CHECKPOINTING", "0" if is_27b else "1")
    gradient_checkpointing = grad_ckpt_raw.strip().lower() in {"1", "true", "yes"}

    print(f"Training LoRA on base model: {base_model}")
    print(
        f"Examples: {train_data} | seq={max_seq_length} batch={batch_size} "
        f"accum={grad_accum} (effective≈{batch_size * grad_accum}) epochs={num_epochs} "
        f"grad_ckpt={gradient_checkpointing} resource_fraction={resource_fraction:.2f}"
    )
    if is_27b:
        print("NOTE: Stop the vLLM server before training to free VRAM.")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available to torch.", file=sys.stderr)
        return 1

    cpu_count = os.cpu_count() or 4
    default_workers = max(1, int(cpu_count * resource_fraction))
    dataloader_workers = _resolve_int(
        config, "DATALOADER_NUM_WORKERS", "TRAIN_DATALOADER_WORKERS", default_workers
    )

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    # resource_fraction sizes CPU workers; do NOT pass a tight max_memory into
    # 4-bit from_pretrained — accelerate will offload layers to CPU and
    # bitsandbytes rejects that without llm_int8_enable_fp32_cpu_offload.
    print(
        f"GPU free={free_bytes // (1024**2)}MiB / total={total_bytes // (1024**2)}MiB; "
        f"dataloader_workers={dataloader_workers}/{cpu_count} "
        f"(resource_fraction={resource_fraction:.2f})"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Single-GPU QLoRA: keep the whole 4-bit model on cuda:0.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    target_modules = list(LORA_TARGET_MODULES)
    named = {n.split(".")[-1] for n, _ in model.named_modules()}
    if not any(t in named for t in target_modules):
        target_modules = sorted(
            {
                name.split(".")[-1]
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear)
                and name.split(".")[-1] not in {"lm_head"}
            }
        )
        print(f"WARNING: default LoRA targets missing; using Linear modules: {target_modules}")
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(train_data), split="train")

    def format_example(example: dict) -> dict:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = dataset.map(format_example)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=max_seq_length,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_epochs,
            learning_rate=LEARNING_RATE,
            output_dir=str(output_dir / "checkpoints"),
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            bf16=True,
            gradient_checkpointing=gradient_checkpointing,
            dataloader_num_workers=dataloader_workers,
            dataloader_pin_memory=True,
        ),
    )

    result = trainer.train()
    print(f"Final training loss: {result.training_loss:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
