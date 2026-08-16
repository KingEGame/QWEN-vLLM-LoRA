# Review package
BASE: e0ac82fea081c0c200b22ea591414077003abe4b
HEAD: 0d44c08f3858104252b38f742917c7c296e16285

## Commits
0d44c08 feat: serve_with_lora supports multiple LORA_MODULES

## Stat
 scripts/serve_with_lora.sh | 32 ++++++++++++++++++++++++++------
 1 file changed, 26 insertions(+), 6 deletions(-)

## Diff
diff --git a/scripts/serve_with_lora.sh b/scripts/serve_with_lora.sh
index 3372888..90c48e4 100644
--- a/scripts/serve_with_lora.sh
+++ b/scripts/serve_with_lora.sh
@@ -1,11 +1,12 @@
 #!/usr/bin/env bash
+# Multi-LoRA: LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" ./scripts/serve_with_lora.sh
 set -euo pipefail
 
 REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
 # shellcheck disable=SC1091
 if [ -f "$REPO_ROOT/scripts/wsl_runtime_env.sh" ]; then
     source "$REPO_ROOT/scripts/wsl_runtime_env.sh"
 fi
 # shellcheck disable=SC1091
 source "$REPO_ROOT/config/model.env"
 
@@ -28,30 +29,49 @@ MAX_SEQS_FLAG=()
 if [ -n "${MAX_NUM_SEQS:-}" ]; then
     MAX_SEQS_FLAG=(--max-num-seqs "$MAX_NUM_SEQS")
 fi
 
 EXTRA_FLAGS=()
 if [ -n "${EXTRA_ARGS:-}" ]; then
     # shellcheck disable=SC2206
     EXTRA_FLAGS=($EXTRA_ARGS)
 fi
 
-ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
-if [ ! -d "$ADAPTER_PATH" ]; then
-    echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
-    exit 1
+LORA_MODULE_ARGS=()
+if [ -n "${LORA_MODULES:-}" ]; then
+    IFS=',' read -r -a _mods <<< "$LORA_MODULES"
+    for spec in "${_mods[@]}"; do
+        spec_trimmed="$(echo "$spec" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
+        name="${spec_trimmed%%=*}"
+        path="${spec_trimmed#*=}"
+        if [[ "$path" != /* ]]; then
+            path="$REPO_ROOT/$path"
+        fi
+        if [ ! -d "$path" ]; then
+            echo "ERROR: LoRA path missing for $name: $path" >&2
+            exit 1
+        fi
+        LORA_MODULE_ARGS+=("${name}=${path}")
+    done
+else
+    ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
+    if [ ! -d "$ADAPTER_PATH" ]; then
+        echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
+        exit 1
+    fi
+    LORA_MODULE_ARGS+=("${ADAPTER_NAME}=${ADAPTER_PATH}")
 fi
 
-echo "Starting vLLM server with LoRA: model=$MODEL adapter=$ADAPTER_NAME port=$PORT"
+echo "Starting vLLM server with LoRA modules: ${LORA_MODULE_ARGS[*]} port=$PORT"
 
 vllm serve "$MODEL" \
     --port "$PORT" \
     --max-model-len "$MAX_MODEL_LEN" \
     --gpu-memory-utilization "$GPU_MEM_UTIL" \
     --enable-lora \
     --max-lora-rank 16 \
-    --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}" \
+    --lora-modules "${LORA_MODULE_ARGS[@]}" \
     "${QUANT_FLAG[@]}" \
     "${REASONING_FLAG[@]}" \
     "${LM_ONLY_FLAG[@]}" \
     "${MAX_SEQS_FLAG[@]}" \
     "${EXTRA_FLAGS[@]}"
