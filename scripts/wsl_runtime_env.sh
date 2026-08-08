#!/usr/bin/env bash
# Runtime env for WSL2 + Blackwell (RTX 50-series) + Qwen3.6 Triton/FlashInfer.
# Sourced by start_server.sh when present. Safe no-op if toolchain dirs are missing.

CC_ENV="${CC_ENV:-$HOME/micromamba/envs/cc}"
if [ -d "$CC_ENV/bin" ]; then
    export PATH="$CC_ENV/bin:${PATH:-}"
    export CC="${CC:-$CC_ENV/bin/gcc}"
    export CXX="${CXX:-$CC_ENV/bin/g++}"
    export CUDA_HOME="${CUDA_HOME:-$CC_ENV}"
    export CPATH="${CC_ENV}/include:${CC_ENV}/targets/x86_64-linux/include:${CPATH:-}"
    export LIBRARY_PATH="${CC_ENV}/lib:${CC_ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${CC_ENV}/lib:${CC_ENV}/targets/x86_64-linux/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

# sm_120 (Blackwell) needs CUDA >= 12.9 for FlashInfer arch normalization.
# Only set arch lists when the GPU reports compute capability major >= 12.
_compute_cap_major=
if command -v nvidia-smi >/dev/null 2>&1; then
    _compute_cap_major="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 | tr -d ' ')"
fi
if [ -n "$_compute_cap_major" ] && [ "$_compute_cap_major" -ge 12 ] 2>/dev/null; then
    export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0f}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
fi
unset _compute_cap_major

# Skip FlashInfer sampling JIT by default (needs full CUDA math headers e.g. curand.h).
# Operators with complete CUDA dev headers can override: VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
