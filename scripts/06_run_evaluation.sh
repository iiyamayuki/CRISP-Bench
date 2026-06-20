#!/usr/bin/env bash
set -euo pipefail

# Stage 06
# Environment: local lmms-eval checkout for model runs; main repository environment for result collection
#
# Usage:
#   bash scripts/06_run_evaluation.sh [model-script]
#   bash scripts/06_run_evaluation.sh --config configs/eval.yaml
#   bash scripts/06_run_evaluation.sh --config configs/eval.yaml --model-script gpt5.sh
#   bash scripts/06_run_evaluation.sh --config configs/eval.yaml --dry-run
#
# When --config is provided and configs/eval.yaml contains a model_matrix section,
# the script iterates over all entries, injecting each entry's env overrides before
# calling the corresponding wrapper script. After all models finish, result collection
# runs once.
#
# Priority: config values > environment variables > script defaults.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi

# Parse options.
CONFIG_FILE=""
MODEL_SCRIPT=""
DRY_RUN=0
POSITIONAL=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --model-script) MODEL_SCRIPT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [ -n "$CONFIG_FILE" ]; then
  eval "$(python "$SCRIPT_DIR/load_config.py" "$CONFIG_FILE" --export)"
fi

# Legacy positional: first positional arg treated as model script name.
if [ -z "$MODEL_SCRIPT" ] && [ "$#" -gt 0 ]; then
  MODEL_SCRIPT="$1"
  shift || true
fi

LMMS_LOGS_DIR="${LMMS_LOGS_DIR:-${LOGS_DIR:-$REPO_ROOT/lmms-eval/logs}}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
CONSISTENCY_RESULTS_DIR="${CONSISTENCY_RESULTS_DIR:-$RESULTS_DIR/consistency_score}"
GLOBAL_SGC_RESULTS_DIR="${GLOBAL_SGC_RESULTS_DIR:-}"
QA_SG_RESULTS_DIR="${QA_SG_RESULTS_DIR:-$RESULTS_DIR/qa_sg}"

run_single_model() {
  local script="$1"
  if [ -f "$script" ]; then
    bash "$script"
  else
    bash "$REPO_ROOT/scripts/eval_models/$script"
  fi
}

run_model_matrix() {
  # Read model_matrix from the config and iterate.
  local count
  count=$(python "$SCRIPT_DIR/read_model_matrix.py" "$CONFIG_FILE" --count)

  if [ "$count" -eq 0 ]; then
    echo "No model_matrix entries found in $CONFIG_FILE" >&2
    return 0
  fi

  echo "=== Model matrix: $count entries ==="

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] Would execute the following model entries:"
    python "$SCRIPT_DIR/read_model_matrix.py" "$CONFIG_FILE" --dry-run
    return 0
  fi

  local idx=0
  python "$SCRIPT_DIR/read_model_matrix.py" "$CONFIG_FILE" | while IFS= read -r entry; do
    idx=$((idx + 1))
    local script
    script=$(echo "$entry" | python3 -c "import sys,json; print(json.load(sys.stdin).get('script',''))")

    if [ -z "$script" ]; then
      echo "Warning: model_matrix entry $idx has no 'script' field, skipping" >&2
      continue
    fi

    echo "=== [$idx/$count] Running: $script ==="

    # Extract env overrides and export them for this run only.
    local env_json
    env_json=$(echo "$entry" | python3 -c "
import sys, json
env = json.load(sys.stdin).get('env', {})
for k, v in env.items():
    print(f'{k}={v}')
")

    (
      # Subshell: env overrides are scoped to this model run.
      if [ -n "$env_json" ]; then
        while IFS= read -r line; do
          export "${line?}"
        done <<< "$env_json"
      fi
      run_single_model "$script"
    )
  done
}

# --- Model execution ---

if [ -n "$CONFIG_FILE" ] && [ -z "$MODEL_SCRIPT" ]; then
  # Config-driven multi-model mode: iterate over model_matrix.
  run_model_matrix
elif [ -n "$MODEL_SCRIPT" ]; then
  # Single model script mode.
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] Would run: $MODEL_SCRIPT"
  else
    run_single_model "$MODEL_SCRIPT"
  fi
fi

# --- Result collection (runs after all models, unless dry-run) ---

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] Would collect results from $LMMS_LOGS_DIR"
  exit 0
fi

mkdir -p "$RESULTS_DIR" "$QA_SG_RESULTS_DIR"

cd "$REPO_ROOT"

collect_results_args=(
  --input_dir "$LMMS_LOGS_DIR"
  --output_dir "$RESULTS_DIR"
  --consistency_dir "$CONSISTENCY_RESULTS_DIR"
)

if [ -n "$GLOBAL_SGC_RESULTS_DIR" ]; then
  collect_results_args+=(--global_sgc_dir "$GLOBAL_SGC_RESULTS_DIR")
fi

python collect_results/collect_results.py "${collect_results_args[@]}"

python collect_results/collect_qa_sg.py \
  --input_dir "$LMMS_LOGS_DIR" \
  --output_dir "$QA_SG_RESULTS_DIR"
