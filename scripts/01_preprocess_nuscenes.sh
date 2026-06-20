#!/usr/bin/env bash
set -euo pipefail

# Stage 01
# Environment: main repository environment
# Purpose: NuScenes-specific preprocessing only.
#   nodes  -> node_construction.py  (build raw node annotations)
#   filter -> filter_nodes.py       (visibility + bbox size filtering)
#   camera -> world_to_camera_translation.py (coordinate transform)
# Shared preprocessing now starts in scripts/03_build_scene_graph.sh.

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

# Load config defaults (only sets variables not already in the environment).
if [ -n "$CONFIG_FILE" ]; then
  eval "$(python "$SCRIPT_DIR/load_config.py" "$CONFIG_FILE")"
fi

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODE="${1:-nodes}"

NUSCENES_DATAROOT="${NUSCENES_DATAROOT:-$REPO_ROOT/data/raw/nuscenes}"
NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-trainval}"
NUSCENES_SCENE_GRAPH_DIR="${NUSCENES_SCENE_GRAPH_DIR:-$REPO_ROOT/data/processed/nuscenes/scene_graph}"
NUSCENES_NODE_JSON="${NUSCENES_NODE_JSON:-$NUSCENES_SCENE_GRAPH_DIR/node_annotations.json}"
NUSCENES_FILTERED_NODES_JSON="${NUSCENES_FILTERED_NODES_JSON:-$NUSCENES_SCENE_GRAPH_DIR/filtered_nodes.json}"
NUSCENES_WORLD_INPUT_JSON="${NUSCENES_WORLD_INPUT_JSON:-$NUSCENES_SCENE_GRAPH_DIR/filtered_nodes.json}"
NUSCENES_WORLD_OUTPUT_JSON="${NUSCENES_WORLD_OUTPUT_JSON:-$NUSCENES_SCENE_GRAPH_DIR/filtered_nodes_cam.json}"
MIN_BOX_H_PX="${MIN_BOX_H_PX:-40}"
MIN_INSIDE_RATIO="${MIN_INSIDE_RATIO:-0.2}"
MIN_BOX_PIXELS="${MIN_BOX_PIXELS:-40}"
MIN_VISIBILITY="${MIN_VISIBILITY:-0}"
NUSCENES_VERIFY_INSTANCE="${NUSCENES_VERIFY_INSTANCE:-1}"
NUSCENES_SYNC_ATTRS="${NUSCENES_SYNC_ATTRS:-0}"
NUSCENES_TOLERANCE="${NUSCENES_TOLERANCE:-0.5}"

mkdir -p "$NUSCENES_SCENE_GRAPH_DIR"

cd "$REPO_ROOT"

run_nodes() {
  python scene_graph/node_construction.py \
    --dataroot "$NUSCENES_DATAROOT" \
    --version "$NUSCENES_VERSION" \
    --out-json "$NUSCENES_NODE_JSON" \
    --min-box-h-px "$MIN_BOX_H_PX" \
    --min-inside-ratio "$MIN_INSIDE_RATIO"
}

run_filter_nodes() {
  python data_preprocessing/filter_nodes.py \
    --nodes "$NUSCENES_NODE_JSON" \
    --out "$NUSCENES_FILTERED_NODES_JSON" \
    --dataroot "$NUSCENES_DATAROOT" \
    --version "$NUSCENES_VERSION" \
    --min_box_pixels "$MIN_BOX_PIXELS" \
    --min_visibility "$MIN_VISIBILITY" \
    --drop_empty_images
}

run_world_to_camera() {
  world_args=(
    --input "$NUSCENES_WORLD_INPUT_JSON"
    --output "$NUSCENES_WORLD_OUTPUT_JSON"
    --dataroot "$NUSCENES_DATAROOT"
    --version "$NUSCENES_VERSION"
    --tolerance "$NUSCENES_TOLERANCE"
  )

  if [ "$NUSCENES_VERIFY_INSTANCE" = "1" ]; then
    world_args+=(--verify_instance)
  fi

  if [ "$NUSCENES_SYNC_ATTRS" = "1" ]; then
    world_args+=(--sync-attrs)
  fi

  python data_preprocessing/world_to_camera_translation.py "${world_args[@]}"
}

case "$MODE" in
  nodes)
    run_nodes
    ;;
  filter)
    run_filter_nodes
    ;;
  camera)
    run_world_to_camera
    ;;
  all)
    run_nodes
    run_filter_nodes
    run_world_to_camera
    ;;
  *)
    echo "Usage: bash scripts/01_preprocess_nuscenes.sh [nodes|filter|camera|all]" >&2
    echo "  nodes  -> scene_graph/node_construction.py" >&2
    echo "  filter -> data_preprocessing/filter_nodes.py" >&2
    echo "  camera -> data_preprocessing/world_to_camera_translation.py" >&2
    echo "  all    -> nodes -> filter -> camera (full pipeline)" >&2
    exit 1
    ;;
esac
