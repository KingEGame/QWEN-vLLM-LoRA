### Task 5: Multi-LoRA serve

**Files:**
- Modify: `scripts/serve_with_lora.sh`

**Interfaces:**
- Consumes: optional `LORA_MODULES` env (comma-separated `name=rel/or/abs/path`)
- Produces: `vllm serve ... --lora-modules name=path [name2=path2 ...]`
- Default when unset: keep today’s single `ADAPTER_NAME=output/lora_adapter` behavior

- [ ] **Step 1: Replace adapter resolution block** in `scripts/serve_with_lora.sh`

Replace the single `ADAPTER_PATH` / `--lora-modules` section with:

```bash
LORA_MODULE_ARGS=()
if [ -n "${LORA_MODULES:-}" ]; then
    IFS=',' read -r -a _mods <<< "$LORA_MODULES"
    for spec in "${_mods[@]}"; do
        spec_trimmed="$(echo "$spec" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        name="${spec_trimmed%%=*}"
        path="${spec_trimmed#*=}"
        if [[ "$path" != /* ]]; then
            path="$REPO_ROOT/$path"
        fi
        if [ ! -d "$path" ]; then
            echo "ERROR: LoRA path missing for $name: $path" >&2
            exit 1
        fi
        LORA_MODULE_ARGS+=("${name}=${path}")
    done
else
    ADAPTER_PATH="$REPO_ROOT/output/lora_adapter"
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "ERROR: no adapter found at $ADAPTER_PATH. Run scripts/train_lora.py first." >&2
        exit 1
    fi
    LORA_MODULE_ARGS+=("${ADAPTER_NAME}=${ADAPTER_PATH}")
fi

echo "Starting vLLM server with LoRA modules: ${LORA_MODULE_ARGS[*]} port=$PORT"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "${LORA_MODULE_ARGS[@]}" \
    "${QUANT_FLAG[@]}" \
    "${REASONING_FLAG[@]}" \
    "${LM_ONLY_FLAG[@]}" \
    "${MAX_SEQS_FLAG[@]}" \
    "${EXTRA_FLAGS[@]}"
```

- [ ] **Step 2: Syntax check**

```bash
bash -n scripts/serve_with_lora.sh
```

Expected: no output, exit 0.

- [ ] **Step 3: Document personal invoke** (comment at top of script or README in Task 7)

Personal serve example:

```bash
LORA_MODULES="question-sharper=output/lora_question_sharper,me-assistant=output/lora_me_assistant" \
  ./scripts/serve_with_lora.sh
```

- [ ] **Step 4: Commit** (if requested)

```bash
git add scripts/serve_with_lora.sh
git commit -m "feat: serve_with_lora supports multiple LORA_MODULES"
```

---

