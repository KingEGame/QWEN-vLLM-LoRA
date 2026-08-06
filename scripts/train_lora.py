#!/usr/bin/env python3
"""Fine-tune a LoRA adapter for Qwen3-4B on customer-support Q&A data.

Uses Unsloth's 4-bit QLoRA training path, sized for 8GB-class GPUs.
Reads data/train.jsonl (validate first with scripts/validate_dataset.py)
and writes the trained adapter to output/lora_adapter/.

Usage: python scripts/train_lora.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.lib.dataset_validation import validate_dataset_file
from scripts.lib.env_config import load_env_file

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = REPO_ROOT / "data" / "train.jsonl"
OUTPUT_DIR = REPO_ROOT / "output" / "lora_adapter"

# Sized for 8GB-class GPUs. Increase MAX_SEQ_LENGTH / BATCH_SIZE if the
# target machine has more VRAM (see config/model.env for GPU tier notes).
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4


def main() -> int:
    if not TRAIN_DATA_PATH.exists():
        print(f"ERROR: {TRAIN_DATA_PATH} not found. Create it first (see docs/how-it-works.md).", file=sys.stderr)
        return 1

    errors = validate_dataset_file(TRAIN_DATA_PATH)
    if errors:
        print(f"ERROR: {TRAIN_DATA_PATH} has {len(errors)} invalid line(s). Run scripts/validate_dataset.py for details.", file=sys.stderr)
        return 1

    config = load_env_file(REPO_ROOT / "config" / "model.env")
    base_model = config["MODEL"]

    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

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
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            logging_steps=1,
            save_strategy="no",
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
