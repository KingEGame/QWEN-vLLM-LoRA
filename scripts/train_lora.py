#!/usr/bin/env python3
"""Fine-tune a LoRA adapter for Qwen3-4B on customer-support Q&A data.

Uses 4-bit QLoRA via transformers + PEFT + bitsandbytes + TRL, sized for
consumer GPUs. Reads data/train.jsonl (validate first with
scripts/validate_dataset.py) and writes the trained adapter to
output/lora_adapter/.

Usage:
  python scripts/train_lora.py

On 6GB cards / when serving uses an AWQ checkpoint:
  TRAIN_MODEL=Qwen/Qwen3-4B-Instruct-2507 MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 \\
    python scripts/train_lora.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.dataset_validation import validate_dataset_file
from scripts.lib.env_config import load_env_file

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = REPO_ROOT / "data" / "train.jsonl"
OUTPUT_DIR = REPO_ROOT / "output" / "lora_adapter"

# Sized for 8GB-class GPUs. Override via env on tighter cards.
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
LEARNING_RATE = 2e-4


def main() -> int:
    if not TRAIN_DATA_PATH.exists():
        print(f"ERROR: {TRAIN_DATA_PATH} not found. Create it first (see README).", file=sys.stderr)
        return 1

    errors = validate_dataset_file(TRAIN_DATA_PATH)
    if errors:
        print(
            f"ERROR: {TRAIN_DATA_PATH} has {len(errors)} invalid line(s). "
            "Run scripts/validate_dataset.py for details.",
            file=sys.stderr,
        )
        return 1

    config = load_env_file(REPO_ROOT / "config" / "model.env")
    # Serving may use an AWQ/compressed checkpoint; QLoRA training needs the
    # dense base instruct model. Override with TRAIN_MODEL when they differ.
    base_model = os.environ.get("TRAIN_MODEL") or config.get("TRAIN_MODEL") or config["MODEL"]
    print(f"Training LoRA on base model: {base_model}")
    print(f"Examples: {TRAIN_DATA_PATH} | seq={MAX_SEQ_LENGTH} batch={BATCH_SIZE} epochs={NUM_EPOCHS}")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available to torch.", file=sys.stderr)
        return 1

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

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
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            bf16=True,
            gradient_checkpointing=True,
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
