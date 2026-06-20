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

TASKS_DIR="${TASKS_DIR:-$REPO_ROOT/tasks}"
OUTPUT_PATH="${OUTPUT_PATH:-./lmms-eval/logs/GPT5}"
LOG_SUFFIX="${LOG_SUFFIX:-gpt5}"
TASKS="${TASKS:-crisp_qa,crisp_sgc}"
MODEL_VERSION="${MODEL_VERSION:-gpt-5-mini}"
MODEL_EFFORT="${MODEL_EFFORT:-high}"
RESPONSE_PERSISTENT_FOLDER="${RESPONSE_PERSISTENT_FOLDER:-./lmms-eval/logs/gpt_persistent_folder}"

cd "$REPO_ROOT"

uv run --project "$LMMS_EVAL_ROOT" \
  python -m lmms_eval \
  --model gpt4v \
  --model_args "model_version=${MODEL_VERSION},effort=${MODEL_EFFORT},modality=image,concurrency_limit=20,response_persistent_folder=${RESPONSE_PERSISTENT_FOLDER}" \
  --tasks "$TASKS" \
  --include_path "$TASKS_DIR" \
  --batch_size 1 \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --log_samples_suffix "$LOG_SUFFIX"