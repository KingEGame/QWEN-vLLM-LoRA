# Review package
BASE: e5160988f4fe2b92713d566972f76aa852716fb5
HEAD: e0ac82fea081c0c200b22ea591414077003abe4b

## Commits
e0ac82f fix: scope Task 4 to CLI --data/--output only
1874a6d feat: train_lora accepts --data and --output paths

## Stat
 scripts/train_lora.py | 39 ++++++++++++++++++++++++++++-----------
 1 file changed, 28 insertions(+), 11 deletions(-)

## Diff
diff --git a/scripts/train_lora.py b/scripts/train_lora.py
index aa5aecc..8a896c4 100644
--- a/scripts/train_lora.py
+++ b/scripts/train_lora.py
@@ -10,20 +10,21 @@ Usage:
   python scripts/train_lora.py
 
 Serving may use an AWQ checkpoint while training needs the dense base. Set
 TRAIN_MODEL in config/model.env or the environment (e.g.
 TRAIN_MODEL=Qwen/Qwen3.6-27B for 27B QLoRA).
 
 On 6GB cards / tight VRAM (especially 27B):
   TRAIN_MODEL=Qwen/Qwen3.6-27B MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1 \\
     python scripts/train_lora.py
 """
+import argparse
 import os
 import sys
 from pathlib import Path
 
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
 from scripts.lib.dataset_validation import validate_dataset_file
 from scripts.lib.env_config import load_env_file
 
 REPO_ROOT = Path(__file__).resolve().parent.parent
@@ -35,39 +36,55 @@ MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
 LORA_RANK = 16
 LORA_ALPHA = 16
 LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
 BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
 GRADIENT_ACCUMULATION_STEPS = 4
 NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
 LEARNING_RATE = 2e-4
 
 
 def main() -> int:
-    if not TRAIN_DATA_PATH.exists():
-        print(f"ERROR: {TRAIN_DATA_PATH} not found. Create it first (see README).", file=sys.stderr)
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--data", default=None, help="Train JSONL path (default data/train.jsonl)")
+    parser.add_argument("--output", default=None, help="Adapter output dir (default output/lora_adapter)")
+    args = parser.parse_args()
+
+    train_data = Path(
+        args.data
+        or os.environ.get("TRAIN_DATA")
+        or (REPO_ROOT / "data" / "train.jsonl")
+    )
+    output_dir = Path(
+        args.output
+        or os.environ.get("TRAIN_OUTPUT")
+        or (REPO_ROOT / "output" / "lora_adapter")
+    )
+
+    if not train_data.exists():
+        print(f"ERROR: {train_data} not found. Create it first (see README).", file=sys.stderr)
         return 1
 
-    errors = validate_dataset_file(TRAIN_DATA_PATH)
+    errors = validate_dataset_file(train_data)
     if errors:
         print(
-            f"ERROR: {TRAIN_DATA_PATH} has {len(errors)} invalid line(s). "
+            f"ERROR: {train_data} has {len(errors)} invalid line(s). "
             "Run scripts/validate_dataset.py for details.",
             file=sys.stderr,
         )
         return 1
 
     config = load_env_file(REPO_ROOT / "config" / "model.env")
     # Serving may use an AWQ/compressed checkpoint; QLoRA training needs the
     # dense base instruct model. Override with TRAIN_MODEL when they differ.
     base_model = os.environ.get("TRAIN_MODEL") or config.get("TRAIN_MODEL") or config["MODEL"]
     print(f"Training LoRA on base model: {base_model}")
-    print(f"Examples: {TRAIN_DATA_PATH} | seq={MAX_SEQ_LENGTH} batch={BATCH_SIZE} epochs={NUM_EPOCHS}")
+    print(f"Examples: {train_data} | seq={MAX_SEQ_LENGTH} batch={BATCH_SIZE} epochs={NUM_EPOCHS}")
     if "27B" in base_model.upper():
         print(
             "NOTE: Stop the vLLM server before training to free VRAM. "
             "Recommended: MAX_SEQ_LENGTH=1024 BATCH_SIZE=1 NUM_EPOCHS=1"
         )
 
     import torch
     from datasets import load_dataset
     from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
     from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
@@ -113,21 +130,21 @@ def main() -> int:
             r=LORA_RANK,
             lora_alpha=LORA_ALPHA,
             target_modules=target_modules,
             lora_dropout=0.0,
             bias="none",
             task_type="CAUSAL_LM",
         ),
     )
     model.print_trainable_parameters()
 
-    dataset = load_dataset("json", data_files=str(TRAIN_DATA_PATH), split="train")
+    dataset = load_dataset("json", data_files=str(train_data), split="train")
 
     def format_example(example: dict) -> dict:
         messages = [
             {"role": "user", "content": example["instruction"]},
             {"role": "assistant", "content": example["response"]},
         ]
         return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
 
     dataset = dataset.map(format_example)
 
@@ -135,31 +152,31 @@ def main() -> int:
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
-            output_dir=str(OUTPUT_DIR / "checkpoints"),
+            output_dir=str(output_dir / "checkpoints"),
             logging_steps=1,
             save_strategy="no",
             report_to="none",
             bf16=True,
             gradient_checkpointing=True,
         ),
     )
 
     result = trainer.train()
     print(f"Final training loss: {result.training_loss:.4f}")
 
-    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
-    model.save_pretrained(str(OUTPUT_DIR))
-    tokenizer.save_pretrained(str(OUTPUT_DIR))
-    print(f"Adapter saved to {OUTPUT_DIR}")
+    output_dir.mkdir(parents=True, exist_ok=True)
+    model.save_pretrained(str(output_dir))
+    tokenizer.save_pretrained(str(output_dir))
+    print(f"Adapter saved to {output_dir}")
     return 0
 
 
 if __name__ == "__main__":
     raise SystemExit(main())
