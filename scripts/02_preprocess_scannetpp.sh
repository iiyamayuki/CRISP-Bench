#!/usr/bin/env bash
set -euo pipefail

# Stage 02
# ScanNet++-specific preprocessing only.
# Shared preprocessing now starts in scripts/03_build_scene_graph.sh.
# The official conversion still requires the separate ScanNet++ environment.

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

PART="${1:-official}"

SCANNETPP_DATAROOT="${SCANNETPP_DATAROOT:-$REPO_ROOT/data/raw/scannetpp}"
SCANNETPP_SCENE_DIR="${SCANNETPP_SCENE_DIR:-$SCANNETPP_DATAROOT/scene}"
SCANNETPP_SCENE_GRAPH_DIR="${SCANNETPP_SCENE_GRAPH_DIR:-$REPO_ROOT/data/processed/scannetpp/scene_graph}"
SCANNETPP_SG_JSONL_DIR="${SCANNETPP_SG_JSONL_DIR:-$SCANNETPP_SCENE_GRAPH_DIR/sg_jsonl}"
SCANNETPP_SCENE_JSON="${SCANNETPP_SCENE_JSON:-$SCANNETPP_SCENE_GRAPH_DIR/scene_graph.json}"
SCANNETPP_BATCH_SIZE="${SCANNETPP_BATCH_SIZE:-4}"

mkdir -p "$SCANNETPP_SCENE_GRAPH_DIR"

cd "$REPO_ROOT"

run_official() {
  python data_preprocessing/scannetpp_official/scannetpp_convert_sg.py \
    --input "$SCANNETPP_SCENE_DIR" \
    --output "$SCANNETPP_SG_JSONL_DIR" \
    --batch_size "$SCANNETPP_BATCH_SIZE" \
    --dataroot "$SCANNETPP_DATAROOT"

  python data_preprocessing/scannetpp_official/merge_sg.py \
    --input_dir "$SCANNETPP_SG_JSONL_DIR" \
    --output_file "$SCANNETPP_SCENE_JSON"
}

case "$PART" in
  official)
    run_official
    ;;
  all)
    run_official
    ;;
  main)
    echo "scripts/02_preprocess_scannetpp.sh main has moved to scripts/03_build_scene_graph.sh scannetpp" >&2
    if [ -n "$CONFIG_FILE" ]; then
      bash "$SCRIPT_DIR/03_build_scene_graph.sh" --config "$CONFIG_FILE" scannetpp
    else
      bash "$SCRIPT_DIR/03_build_scene_graph.sh" scannetpp
    fi
    ;;
  *)
    echo "Usage: bash scripts/02_preprocess_scannetpp.sh [official|main|all]" >&2
    echo "  official -> ScanNet++ official conversion only" >&2
    echo "  main     -> compatibility alias for scripts/03_build_scene_graph.sh scannetpp" >&2
    echo "  all      -> run official conversion only; shared steps now live in step 03" >&2
    exit 1
    ;;
esac
