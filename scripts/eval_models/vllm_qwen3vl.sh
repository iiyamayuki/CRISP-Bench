#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-$REPO_ROOT/lmms-eval}"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi

DEFAULT_UV_CACHE_DIR="$WORKSPACE_ROOT/uv_cache"
if [ ! -d "$DEFAULT_UV_CACHE_DIR" ]; then
  DEFAULT_UV_CACHE_DIR="$REPO_ROOT/.uv_cache"
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-$DEFAULT_UV_CACHE_DIR}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/HF_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export LMMS_EVAL_HOME="${LMMS_EVAL_HOME:-$REPO_ROOT/LMMS_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$REPO_ROOT/vllm_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-18000000}"

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
NUM_GPUS="${NUM_GPUS:-2}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$NUM_GPUS}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TASKS="${TASKS:-crisp_qa,crisp_sgc}"
TASKS_DIR="${TASKS_DIR:-$REPO_ROOT/tasks}"
OUTPUT_PATH="${OUTPUT_PATH:-./lmms-eval/logs/qwen3vl_vllm}"
LOG_SUFFIX="${LOG_SUFFIX:-qwen3vl_vllm}"

cd "$REPO_ROOT"

uv run --project "$LMMS_EVAL_ROOT" \
  python -m lmms_eval \
  --model vllm \
  --model_args "model=${MODEL},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},mm_processor_cache_gb=${MM_PROCESSOR_CACHE_GB},max_model_len=${MAX_MODEL_LEN}" \
  --tasks "$TASKS" \
  --include_path "$TASKS_DIR" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --log_samples_suffix "$LOG_SUFFIX"