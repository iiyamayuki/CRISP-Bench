#!/usr/bin/env bash
set -euo pipefail

# Stage 07
# Consistency evaluation pipeline.
#
# Usage:
#   bash scripts/07_consistency_eval.sh [--config <yaml>] batch [--dry_run]
#       Auto-discover all models in lmms-eval/logs and evaluate.
#
#   bash scripts/07_consistency_eval.sh [--config <yaml>] single
#       Run for a single model; requires SGC_LOG_JSONL and
#       DIRECT_QA_LOG_JSONL env vars.
#
# Priority: config values > environment variables > script defaults.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi

# Parse --config option.
CONFIG_FILE=""
POSITIONAL=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [ -n "$CONFIG_FILE" ]; then
  eval "$(python "$SCRIPT_DIR/load_config.py" "$CONFIG_FILE")"
fi

DATASETS_ROOT="${DATASETS_ROOT:-$REPO_ROOT/data/processed}"
COMBINED_ROOT="${COMBINED_ROOT:-$DATASETS_ROOT/combined}"
GENERATED_SG_DIR="${GENERATED_SG_DIR:-$REPO_ROOT/generated_sg}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
CONSISTENCY_RESULTS_DIR="${CONSISTENCY_RESULTS_DIR:-$RESULTS_DIR/consistency_score}"

QA_TEMPLATES="${QA_TEMPLATES:-$REPO_ROOT/qa/question_template.json}"
CONSISTENCY_QA_LIST="${CONSISTENCY_QA_LIST:-$COMBINED_ROOT/QA_pairs/qa_data.jsonl}"

LOGS_DIR="${LOGS_DIR:-$REPO_ROOT/lmms-eval/logs}"

cd "$REPO_ROOT"

MODE="${1:-}"
shift || true

case "$MODE" in
  batch)
    echo "=== Batch consistency evaluation ==="
    python evaluation/run_consistency_eval.py \
      --logs_dir "$LOGS_DIR" \
      --output_dir "$CONSISTENCY_RESULTS_DIR" \
      --qa_list "$CONSISTENCY_QA_LIST" \
      --templates "$QA_TEMPLATES" \
      "$@"
    ;;

  single)
    : "${SGC_LOG_JSONL:?Set SGC_LOG_JSONL to the lmms-eval SGC samples JSONL path.}"
    : "${DIRECT_QA_LOG_JSONL:?Set DIRECT_QA_LOG_JSONL to the lmms-eval direct-QA samples JSONL path.}"

    DERIVED_SG_JSON="${DERIVED_SG_JSON:-$GENERATED_SG_DIR/derived_scene_graph.json}"
    DERIVED_QA_JSONL="${DERIVED_QA_JSONL:-$GENERATED_SG_DIR/derived_qa.jsonl}"
    CONSISTENCY_OUTPUT_JSON="${CONSISTENCY_OUTPUT_JSON:-$CONSISTENCY_RESULTS_DIR/consistency_results.json}"

    mkdir -p "$GENERATED_SG_DIR" "$CONSISTENCY_RESULTS_DIR"

    python evaluation/extract_sg.py \
      --input "$SGC_LOG_JSONL" \
      --output "$DERIVED_SG_JSON"

    python evaluation/scene_graph_solver.py \
      --scene-graphs "$DERIVED_SG_JSON" \
      --qa-list "$CONSISTENCY_QA_LIST" \
      --templates "$QA_TEMPLATES" \
      --output "$DERIVED_QA_JSONL"

    python evaluation/calculate_consistency.py \
      --derived-file "$DERIVED_QA_JSONL" \
      --direct-file "$DIRECT_QA_LOG_JSONL" \
      --output "$CONSISTENCY_OUTPUT_JSON"
    ;;

  *)
    echo "Usage: $0 {batch [--dry_run] | single}"
    echo ""
    echo "  batch   Auto-discover models in \$LOGS_DIR and evaluate all pairs."
    echo "  single  Evaluate one pair (requires SGC_LOG_JSONL, DIRECT_QA_LOG_JSONL)."
    exit 1
    ;;
esac
