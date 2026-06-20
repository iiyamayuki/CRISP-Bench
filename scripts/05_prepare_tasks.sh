#!/usr/bin/env bash
set -euo pipefail

# Stage 05
# Environment: main repository environment
# Shared task preparation for NuScenes / ScanNet++ plus cross-dataset aggregation.

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

NUSCENES_FILTERED_SG="${NUSCENES_FILTERED_SG:-$NUSCENES_ROOT/scene_graph/filtered_scene_graph.json}"
NUSCENES_QA_DIR="${NUSCENES_QA_DIR:-$NUSCENES_ROOT/QA_pairs}"
NUSCENES_QA_BALANCED_JSONL="${NUSCENES_QA_BALANCED_JSONL:-$NUSCENES_QA_DIR/qa_data_balanced.jsonl}"
NUSCENES_SGC_DIR="${NUSCENES_SGC_DIR:-$NUSCENES_ROOT/SGC_task}"
NUSCENES_SGC_JSON="${NUSCENES_SGC_JSON:-$NUSCENES_SGC_DIR/sgc_task.json}"

SCANNETPP_FILTERED_SG="${SCANNETPP_FILTERED_SG:-$SCANNETPP_ROOT/scene_graph/filtered_scene_graph.json}"
SCANNETPP_QA_DIR="${SCANNETPP_QA_DIR:-$SCANNETPP_ROOT/QA_pairs}"
SCANNETPP_QA_BALANCED_JSONL="${SCANNETPP_QA_BALANCED_JSONL:-$SCANNETPP_QA_DIR/qa_data_balanced.jsonl}"
SCANNETPP_SGC_DIR="${SCANNETPP_SGC_DIR:-$SCANNETPP_ROOT/SGC_task}"
SCANNETPP_SGC_JSON="${SCANNETPP_SGC_JSON:-$SCANNETPP_SGC_DIR/sgc_task.json}"

COMBINED_QA_DIR="${COMBINED_QA_DIR:-$COMBINED_ROOT/QA_pairs}"
COMBINED_QA_JSONL="${COMBINED_QA_JSONL:-$COMBINED_QA_DIR/qa_data.jsonl}"
COMBINED_SGC_DIR="${COMBINED_SGC_DIR:-$COMBINED_ROOT/SGC_task}"
COMBINED_SGC_JSONL="${COMBINED_SGC_JSONL:-$COMBINED_SGC_DIR/sgc_task.jsonl}"

mkdir -p \
  "$NUSCENES_SGC_DIR" "$SCANNETPP_SGC_DIR" \
  "$COMBINED_SGC_DIR" "$COMBINED_QA_DIR"

cd "$REPO_ROOT"

run_dataset() {
  local dataset="$1"
  local filtered_sg=""
  local sgc_json=""

  case "$dataset" in
    nuscenes)
      filtered_sg="$NUSCENES_FILTERED_SG"
      sgc_json="$NUSCENES_SGC_JSON"
      ;;
    scannetpp)
      filtered_sg="$SCANNETPP_FILTERED_SG"
      sgc_json="$SCANNETPP_SGC_JSON"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac

  python scene_graph/convert_sg_to_sharegpt.py \
    --input "$filtered_sg" \
    --output "$sgc_json"
}

aggregate_combined_tasks() {
  python data_preprocessing/aggregate_dataset.py \
    --inputs "$NUSCENES_SGC_JSON" "$SCANNETPP_SGC_JSON" \
    --output "$COMBINED_SGC_JSONL" \
    --dataset-names nuscenes scannetpp
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
    aggregate_combined_tasks
    ;;
  *)
    echo "Usage: bash scripts/05_prepare_tasks.sh [nuscenes|scannetpp|all]" >&2
    exit 1
    ;;
esac
