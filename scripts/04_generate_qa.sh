#!/usr/bin/env bash
set -euo pipefail

# Stage 04
# Environment: main repository environment
# Shared QA generation for NuScenes / ScanNet++ plus optional cross-dataset aggregation.

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

TARGET_DATASET="${1:-all}"

DATASETS_ROOT="${DATASETS_ROOT:-$REPO_ROOT/data/processed}"
NUSCENES_ROOT="${NUSCENES_ROOT:-$DATASETS_ROOT/nuscenes}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-$DATASETS_ROOT/scannetpp}"
COMBINED_ROOT="${COMBINED_ROOT:-$DATASETS_ROOT/combined}"
QA_TEMPLATES="${QA_TEMPLATES:-$REPO_ROOT/qa/question_template.json}"

NUSCENES_FILTERED_SG="${NUSCENES_FILTERED_SG:-$NUSCENES_ROOT/scene_graph/filtered_scene_graph.json}"
NUSCENES_QA_DIR="${NUSCENES_QA_DIR:-$NUSCENES_ROOT/QA_pairs}"
NUSCENES_QA_JSONL="${NUSCENES_QA_JSONL:-$NUSCENES_QA_DIR/qa_data.jsonl}"
NUSCENES_QA_BALANCED_JSONL="${NUSCENES_QA_BALANCED_JSONL:-$NUSCENES_QA_DIR/qa_data_balanced.jsonl}"

SCANNETPP_FILTERED_SG="${SCANNETPP_FILTERED_SG:-$SCANNETPP_ROOT/scene_graph/filtered_scene_graph.json}"
SCANNETPP_QA_DIR="${SCANNETPP_QA_DIR:-$SCANNETPP_ROOT/QA_pairs}"
SCANNETPP_QA_JSONL="${SCANNETPP_QA_JSONL:-$SCANNETPP_QA_DIR/qa_data.jsonl}"
SCANNETPP_QA_BALANCED_JSONL="${SCANNETPP_QA_BALANCED_JSONL:-$SCANNETPP_QA_DIR/qa_data_balanced.jsonl}"

COMBINED_QA_DIR="${COMBINED_QA_DIR:-$COMBINED_ROOT/QA_pairs}"
COMBINED_QA_JSONL="${COMBINED_QA_JSONL:-$COMBINED_QA_DIR/qa_data.jsonl}"

mkdir -p "$NUSCENES_QA_DIR" "$SCANNETPP_QA_DIR" "$COMBINED_QA_DIR"

cd "$REPO_ROOT"

run_dataset() {
  local dataset="$1"
  local filtered_sg=""
  local qa_jsonl=""
  local qa_balanced_jsonl=""
  local use_vertical=""

  case "$dataset" in
    nuscenes)
      filtered_sg="$NUSCENES_FILTERED_SG"
      qa_jsonl="$NUSCENES_QA_JSONL"
      qa_balanced_jsonl="$NUSCENES_QA_BALANCED_JSONL"
      use_vertical="0"
      ;;
    scannetpp)
      filtered_sg="$SCANNETPP_FILTERED_SG"
      qa_jsonl="$SCANNETPP_QA_JSONL"
      qa_balanced_jsonl="$SCANNETPP_QA_BALANCED_JSONL"
      use_vertical="1"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac

  python qa/generate_qa_pairs.py \
    --scene-graphs "$filtered_sg" \
    --templates "$QA_TEMPLATES" \
    --output "$qa_jsonl" \
    --seed 42 \
    --use-vertical "$use_vertical" \
    --dataset "$dataset"

  python qa/qa_balance_mcq.py \
    --input "$qa_jsonl" \
    --output "$qa_balanced_jsonl"
}

aggregate_selected() {
  local inputs=()
  local dataset_names=()

  if [ "$1" = "nuscenes" ] || [ "$1" = "all" ]; then
    inputs+=("$NUSCENES_QA_BALANCED_JSONL")
    dataset_names+=("nuscenes")
  fi

  if [ "$1" = "scannetpp" ] || [ "$1" = "all" ]; then
    inputs+=("$SCANNETPP_QA_BALANCED_JSONL")
    dataset_names+=("scannetpp")
  fi

  python data_preprocessing/aggregate_dataset.py \
    --inputs "${inputs[@]}" \
    --output "$COMBINED_QA_JSONL" \
    --dataset-names "${dataset_names[@]}"
}

case "$TARGET_DATASET" in
  nuscenes)
    run_dataset nuscenes
    ;;
  scannetpp)
    run_dataset scannetpp
    ;;
  all)
    run_dataset nuscenes
    run_dataset scannetpp
    aggregate_selected all
    ;;
  *)
    echo "Usage: bash scripts/04_generate_qa.sh [nuscenes|scannetpp|all]" >&2
    exit 1
    ;;
esac
