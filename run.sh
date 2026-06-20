#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

usage() {
    cat <<'EOF'
CRISP Bench staged workflow entrypoint.

Usage:
    bash run.sh <stage> [options] [args]
    bash run.sh eval-model <model-script>
    bash run.sh all-pre-eval [--config <yaml>]

Stages:
    01  scripts/01_preprocess_nuscenes.sh   [nodes|filter|camera|all]
    02  scripts/02_preprocess_scannetpp.sh  [official|main|all]
    03  scripts/03_build_scene_graph.sh     [nuscenes|scannetpp|all]
    04  scripts/04_generate_qa.sh           [nuscenes|scannetpp|all]
    05  scripts/05_prepare_tasks.sh         [nuscenes|scannetpp|all]
    06  scripts/06_run_evaluation.sh        [--config <yaml>] [--model-script <name>] [--dry-run]
    07  scripts/07_consistency_eval.sh      [--config <yaml>] {batch [--dry_run] | single}

Config-driven usage (recommended for public workflows):
    bash run.sh 01 --config configs/nuscenes.yaml [nodes|filter|camera|all]
    bash run.sh 03 --config configs/nuscenes.yaml [nuscenes|scannetpp|all]
    bash run.sh 06 --config configs/eval.yaml
    bash run.sh 06 --config configs/eval.yaml --model-script gpt5.sh

Priority: config values > environment variables > script defaults.

Examples:
    bash run.sh 01 camera
    bash run.sh 03 scannetpp
    bash run.sh 06 --config configs/eval.yaml
    bash run.sh 06 --config configs/eval.yaml --model-script gpt5.sh
    bash run.sh 06 --config configs/eval.yaml --dry-run
    bash run.sh 07 batch
    bash run.sh 07 --config configs/eval.yaml batch --dry_run
    bash run.sh eval-model vllm_qwen3vl.sh

Notes:
    - run.sh is a pure dispatcher that forwards to scripts/0*.sh.
    - Steps 01-02 contain dataset-specific preprocessing only.
    - Shared preprocessing starts in step 03, parameterized by dataset.
    - Model-specific lmms-eval wrappers live in scripts/eval_models/.
    - Config files under configs/ are the recommended user configuration surface.
    - Bottom-level Python CLI arguments are preserved for development/debugging.
EOF
}

run_stage() {
    local stage_script="$1"
    shift || true
    bash "$REPO_ROOT/scripts/$stage_script" "$@"
}

# Extract --config from the remaining arguments and forward to stage scripts.
CONFIG_FILE=""
REMAINING_ARGS=()
STAGE="${1:-help}"
shift || true

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG_FILE="${2:-}"
            if [ -z "$CONFIG_FILE" ]; then
                echo "Error: --config requires a file argument" >&2
                exit 1
            fi
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build forwarded args: if --config was given, pass it through to stage scripts.
FORWARD_ARGS=()
if [ -n "$CONFIG_FILE" ]; then
    FORWARD_ARGS+=(--config "$CONFIG_FILE")
fi
FORWARD_ARGS+=("${REMAINING_ARGS[@]}")

case "$STAGE" in
    01|preprocess_nuscenes)
        run_stage 01_preprocess_nuscenes.sh "${FORWARD_ARGS[@]}"
        ;;
    02|preprocess_scannetpp)
        run_stage 02_preprocess_scannetpp.sh "${FORWARD_ARGS[@]}"
        ;;
    03|build_scene_graph)
        run_stage 03_build_scene_graph.sh "${FORWARD_ARGS[@]}"
        ;;
    04|generate_qa)
        run_stage 04_generate_qa.sh "${FORWARD_ARGS[@]}"
        ;;
    05|prepare_tasks)
        run_stage 05_prepare_tasks.sh "${FORWARD_ARGS[@]}"
        ;;
    06|run_evaluation)
        run_stage 06_run_evaluation.sh "${FORWARD_ARGS[@]}"
        ;;
    07|consistency_eval)
        run_stage 07_consistency_eval.sh "${FORWARD_ARGS[@]}"
        ;;
    eval-model)
        if [ "${#REMAINING_ARGS[@]}" -lt 1 ]; then
            echo "Usage: bash run.sh eval-model <model-script>" >&2
            exit 1
        fi
        bash "$REPO_ROOT/scripts/eval_models/${REMAINING_ARGS[0]}"
        ;;
    all-pre-eval)
        run_stage 01_preprocess_nuscenes.sh "${FORWARD_ARGS[@]}" nodes
        run_stage 02_preprocess_scannetpp.sh "${FORWARD_ARGS[@]}" official
        if [ -f "${NUSCENES_SHARED_INPUT_JSON:-$REPO_ROOT/../Datasets/NuScenes/scene_graph/merged_nodes_with_captions_cam.json}" ]; then
            run_stage 03_build_scene_graph.sh "${FORWARD_ARGS[@]}" all
            run_stage 04_generate_qa.sh "${FORWARD_ARGS[@]}" all
            run_stage 05_prepare_tasks.sh "${FORWARD_ARGS[@]}" all
        else
            echo "NuScenes shared-input JSON not found; running shared stages for ScanNet++ only." >&2
            run_stage 03_build_scene_graph.sh "${FORWARD_ARGS[@]}" scannetpp
            run_stage 04_generate_qa.sh "${FORWARD_ARGS[@]}" scannetpp
            run_stage 05_prepare_tasks.sh "${FORWARD_ARGS[@]}" scannetpp
        fi
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $STAGE" >&2
        usage
        exit 1
        ;;
esac