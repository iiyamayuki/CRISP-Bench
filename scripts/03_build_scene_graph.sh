#!/usr/bin/env bash
set -euo pipefail

# Stage 03
# Environment: main repository environment
# Shared preprocessing and scene graph construction for NuScenes / ScanNet++.
# The dataset-specific preprocessing that feeds this step lives in steps 01 and 02.

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

EDGE_EPS="${EDGE_EPS:-0.2}"
EDGE_CENTROID_TOL="${EDGE_CENTROID_TOL:-0.5}"
EDGE_MAX_DISTANCE="${EDGE_MAX_DISTANCE:-2.5}"
EDGE_FILTER_THRESHOLD="${EDGE_FILTER_THRESHOLD:-0.9}"
EDGE_FILTER_USE_CLIP="${EDGE_FILTER_USE_CLIP:-1}"

NUSCENES_DATAROOT="${NUSCENES_DATAROOT:-$REPO_ROOT/data/raw/nuscenes}"
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-trainval}"
NUSCENES_SCENE_GRAPH_DIR="${NUSCENES_SCENE_GRAPH_DIR:-$REPO_ROOT/data/processed/nuscenes/scene_graph}"
NUSCENES_SHARED_INPUT_JSON="${NUSCENES_SHARED_INPUT_JSON:-$NUSCENES_SCENE_GRAPH_DIR/filtered_nodes_cam.json}"
NUSCENES_NODES_FILTERED_JSON="${NUSCENES_NODES_FILTERED_JSON:-$NUSCENES_SCENE_GRAPH_DIR/nodes_filtered.json}"
NUSCENES_ANNOTATED_DIR="${NUSCENES_ANNOTATED_DIR:-$REPO_ROOT/data/processed/nuscenes/annotated_image}"
NUSCENES_ANNOTATED_JSON="${NUSCENES_ANNOTATED_JSON:-$NUSCENES_ANNOTATED_DIR/nodes_with_2dbox.json}"
NUSCENES_SCENE_JSON="${NUSCENES_SCENE_JSON:-$NUSCENES_SCENE_GRAPH_DIR/scene_graph.json}"
NUSCENES_FILTERED_SCENE_GRAPH_JSON="${NUSCENES_FILTERED_SCENE_GRAPH_JSON:-$NUSCENES_SCENE_GRAPH_DIR/filtered_scene_graph.json}"
NUSCENES_DIVERSITY_THRESHOLD="${NUSCENES_DIVERSITY_THRESHOLD:-0.8}"
NUSCENES_IMAGE_CAP="${NUSCENES_IMAGE_CAP:-600}"

SCANNETPP_DATAROOT="${SCANNETPP_DATAROOT:-$REPO_ROOT/data/raw/scannetpp}"
SCANNETPP_SCENE_GRAPH_DIR="${SCANNETPP_SCENE_GRAPH_DIR:-$REPO_ROOT/data/processed/scannetpp/scene_graph}"
SCANNETPP_SG_JSONL_DIR="${SCANNETPP_SG_JSONL_DIR:-$SCANNETPP_SCENE_GRAPH_DIR/sg_jsonl}"
SCANNETPP_SHARED_INPUT_JSON="${SCANNETPP_SHARED_INPUT_JSON:-$SCANNETPP_SCENE_GRAPH_DIR/scene_graph.json}"
SCANNETPP_NODES_FILTERED_JSON="${SCANNETPP_NODES_FILTERED_JSON:-$SCANNETPP_SCENE_GRAPH_DIR/nodes_filtered.json}"
SCANNETPP_ANNOTATED_DIR="${SCANNETPP_ANNOTATED_DIR:-$REPO_ROOT/data/processed/scannetpp/annotated_image}"
SCANNETPP_ANNOTATED_JSON="${SCANNETPP_ANNOTATED_JSON:-$SCANNETPP_ANNOTATED_DIR/nodes_with_2dbox.json}"
SCANNETPP_SCENE_JSON="${SCANNETPP_SCENE_JSON:-$SCANNETPP_SCENE_GRAPH_DIR/scene_graph.json}"
SCANNETPP_FILTERED_SCENE_GRAPH_JSON="${SCANNETPP_FILTERED_SCENE_GRAPH_JSON:-$SCANNETPP_SCENE_GRAPH_DIR/filtered_scene_graph.json}"
SCANNETPP_DIVERSITY_THRESHOLD="${SCANNETPP_DIVERSITY_THRESHOLD:-0.8}"
SCANNETPP_IMAGE_CAP="${SCANNETPP_IMAGE_CAP:-600}"

mkdir -p \
  "$NUSCENES_SCENE_GRAPH_DIR" "$NUSCENES_ANNOTATED_DIR" \
  "$SCANNETPP_SCENE_GRAPH_DIR" "$SCANNETPP_ANNOTATED_DIR"

cd "$REPO_ROOT"

run_dataset() {
  local dataset="$1"
  local input_json=""
  local filtered_nodes_json=""
  local annotated_dir=""
  local annotated_json=""
  local scene_json=""
  local filtered_scene_json=""
  local dataroot=""
  local diversity_threshold=""
  local image_cap=""
  local mark_args=()

  case "$dataset" in
    nuscenes)
      input_json="$NUSCENES_SHARED_INPUT_JSON"
      filtered_nodes_json="$NUSCENES_NODES_FILTERED_JSON"
      annotated_dir="$NUSCENES_ANNOTATED_DIR"
      annotated_json="$NUSCENES_ANNOTATED_JSON"
      scene_json="$NUSCENES_SCENE_JSON"
      filtered_scene_json="$NUSCENES_FILTERED_SCENE_GRAPH_JSON"
      dataroot="$NUSCENES_DATAROOT"
      diversity_threshold="$NUSCENES_DIVERSITY_THRESHOLD"
      image_cap="$NUSCENES_IMAGE_CAP"
      mark_args=(
        --dataset nuscenes
        --nusc_root "$NUSCENES_DATAROOT"
        --nusc_version "$NUSCENES_VERSION"
      )
      ;;
    scannetpp)
      input_json="$SCANNETPP_SHARED_INPUT_JSON"
      filtered_nodes_json="$SCANNETPP_NODES_FILTERED_JSON"
      annotated_dir="$SCANNETPP_ANNOTATED_DIR"
      annotated_json="$SCANNETPP_ANNOTATED_JSON"
      scene_json="$SCANNETPP_SCENE_JSON"
      filtered_scene_json="$SCANNETPP_FILTERED_SCENE_GRAPH_JSON"
      dataroot="$SCANNETPP_DATAROOT"
      diversity_threshold="$SCANNETPP_DIVERSITY_THRESHOLD"
      image_cap="$SCANNETPP_IMAGE_CAP"
      mark_args=(
        --dataset scannetpp
        --original_jsonl_dir "$SCANNETPP_SG_JSONL_DIR"
      )
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac

  python data_preprocessing/object_filter.py \
    --input "$input_json" \
    --output "$filtered_nodes_json" \
    --seed 42 \
    --diversity-threshold "$diversity_threshold" \
    --image_cap "$image_cap"

  python data_preprocessing/mark_objects.py \
    "${mark_args[@]}" \
    --input "$filtered_nodes_json" \
    --output "$annotated_dir" \
    --dataroot "$dataroot"

  python scene_graph/edge_construction.py \
    --input "$annotated_json" \
    --output "$scene_json" \
    --eps "$EDGE_EPS" \
    --centroid-tol "$EDGE_CENTROID_TOL" \
    --max-distance "$EDGE_MAX_DISTANCE"

  edge_filter_args=(
    --input "$scene_json"
    --output "$filtered_scene_json"
    --threshold "$EDGE_FILTER_THRESHOLD"
  )

  if [ "$EDGE_FILTER_USE_CLIP" = "1" ]; then
    edge_filter_args+=(--clip --dataroot "$dataroot")
  fi

  python scene_graph/edge_based_filter.py "${edge_filter_args[@]}"
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
    ;;
  *)
    echo "Usage: bash scripts/03_build_scene_graph.sh [nuscenes|scannetpp|all]" >&2
    exit 1
    ;;
esac
