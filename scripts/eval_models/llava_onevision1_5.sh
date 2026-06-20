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
OUTPUT_PATH="${OUTPUT_PATH:-./lmms-eval/logs/llava_onevision1_5}"
LOG_SUFFIX="${LOG_SUFFIX:-llava_onevision1_5}"
TASKS="${TASKS:-crisp_qa,crisp_sgc}"
MODEL_NAME="${MODEL_NAME:-lmms-lab/LLaVA-OneVision-1.5-8B-Instruct}"
NUM_GPUS="${NUM_GPUS:-2}"
ACCELERATE_NUM_PROCESSES="${ACCELERATE_NUM_PROCESSES:-$NUM_GPUS}"
ACCELERATE_MAIN_PROCESS_PORT="${ACCELERATE_MAIN_PROCESS_PORT:-12399}"

cd "$REPO_ROOT"

uv run --project "$LMMS_EVAL_ROOT" \
  accelerate launch --num_processes="$ACCELERATE_NUM_PROCESSES" --main_process_port "$ACCELERATE_MAIN_PROCESS_PORT" -m lmms_eval \
  --model=llava_onevision1_5 \
  --model_args="pretrained=${MODEL_NAME},attn_implementation=sdpa,max_pixels=3240000" \
  --tasks="$TASKS" \
  --include_path "$TASKS_DIR" \
  --batch_size=1 \
  --output_path="$OUTPUT_PATH" \
  --log_samples \
  --log_samples_suffix "$LOG_SUFFIX"