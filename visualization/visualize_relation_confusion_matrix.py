#!/usr/bin/env python3

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Relation pair definitions
RELATION_PAIRS = {
    "front_back": ("in front of", "behind"),
    "left_right": ("left", "right"),
    "up_down": ("up", "down")
}

LABEL_ORDERS = {
    "front_back": ["in front of", "behind", "none"],
    "left_right": ["left", "right", "none"],
    "up_down": ["up", "down", "none"]
}

RELATION_DISPLAY_NAMES = {
    "front_back": "Front-Back Relation",
    "left_right": "Left-Right Relation",
    "up_down": "Up-Down Relation"
}


def load_aggregated_results(aggregated_file: str) -> dict:
    print(f"Loading aggregated results: {aggregated_file}")

    if not os.path.exists(aggregated_file):
        raise FileNotFoundError(f"File not found: {aggregated_file}")

    with open(aggregated_file, encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded results for {len(data)} models")
    return data


def extract_source_files(aggregated_data: dict) -> dict[str, str]:
    source_files = {}

    for model_key, model_data in aggregated_data.items():
        tasks = model_data.get("tasks", {})
        sgc_task = tasks.get("crisp_sgc", {})
        multimodal = sgc_task.get("multimodal", {})

        if "source_file" in multimodal:
            source_files[model_key] = multimodal["source_file"]

    print(f"Found source files for {len(source_files)} models\n")
    return source_files


def load_sample_data(source_file: str) -> list[dict]:
    source_path = Path(source_file)
    timestamp = source_path.stem.replace("_results", "")
    samples_file = source_path.parent / f"{timestamp}_samples_crisp_sgc.jsonl"

    if not samples_file.exists():
        print(f"Warning: Sample file not found: {samples_file}")
        return []

    samples = []
    with open(samples_file, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error at line {line_num}: {e}")

    return samples


def extract_scene_graphs(samples: list[dict]) -> list[tuple[dict, dict]]:
    scene_graphs = []

    for idx, sample in enumerate(samples):
        try:
            # Extract GT scene graph
            if "target" in sample:
                gt_sg = json.loads(sample["target"])
            elif "sgc_score" in sample:
                conversations = sample["sgc_score"].get("conversations", [])
                if len(conversations) > 1:
                    gt_sg = json.loads(conversations[1]["value"])
                else:
                    continue
            else:
                continue

            # Extract predicted scene graph
            if "sgc_score" in sample and "prediction_json" in sample["sgc_score"]:
                pred_sg = sample["sgc_score"]["prediction_json"]
            else:
                continue

            scene_graphs.append((gt_sg, pred_sg))

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to extract scene graph for sample {idx}: {e}")

    return scene_graphs


def find_edge_in_pred(pred_edges: list[dict], from_id: int, to_id: int) -> dict | None:
    for edge in pred_edges:
        if edge.get("from") == from_id and edge.get("to") == to_id:
            return edge
    return None


def extract_relation_value(relation: dict, relation_pair: tuple[str, str]) -> str:
    key1, key2 = relation_pair
    val1 = relation.get(key1, 0)
    val2 = relation.get(key2, 0)

    if val1 == 0 and val2 == 0:
        return "none"
    return key1 if val1 == 1 else key2


def extract_relation_labels(
    gt_sg: dict,
    pred_sg: dict,
    relation_pair: tuple[str, str]
) -> tuple[list[str], list[str]]:
    gt_labels = []
    pred_labels = []

    gt_edges = gt_sg.get("edges", [])
    pred_edges = pred_sg.get("edges", [])

    for gt_edge in gt_edges:
        from_id = gt_edge.get("from")
        to_id = gt_edge.get("to")
        gt_relation = gt_edge.get("relation", {})

        gt_label = extract_relation_value(gt_relation, relation_pair)

        pred_edge = find_edge_in_pred(pred_edges, from_id, to_id)

        if pred_edge is None:
            pred_label = "none"
        else:
            pred_relation = pred_edge.get("relation", {})

            # Handle relation as list format
            if isinstance(pred_relation, list):
                pred_relation_dict = {rel: 0 for rel in ["left", "right", "in front of", "behind", "up", "down"]}
                for rel in pred_relation:
                    if rel in pred_relation_dict:
                        pred_relation_dict[rel] = 1
                pred_relation = pred_relation_dict

            pred_label = extract_relation_value(pred_relation, relation_pair)

        gt_labels.append(gt_label)
        pred_labels.append(pred_label)

    return gt_labels, pred_labels


def count_relations(
    scene_graphs: list[tuple[dict, dict]],
    use_gt: bool = True,
    match_gt_only: bool = False
) -> dict[str, int]:
    """Count relation occurrences in scene graphs."""
    relation_counts = defaultdict(int)
    total_edges = 0
    missing_edges = 0

    for gt_sg, pred_sg in scene_graphs:
        if use_gt:
            edges = gt_sg.get("edges", [])
            total_edges += len(edges)

            for edge in edges:
                relation = edge.get("relation", {})
                if isinstance(relation, dict):
                    for rel_name, rel_value in relation.items():
                        if rel_value == 1:
                            relation_counts[rel_name] += 1
        else:
            if match_gt_only:
                gt_edges = gt_sg.get("edges", [])
                pred_edges = pred_sg.get("edges", [])

                for gt_edge in gt_edges:
                    pred_edge = find_edge_in_pred(pred_edges, gt_edge.get("from"), gt_edge.get("to"))

                    if pred_edge is None:
                        missing_edges += 1
                    else:
                        total_edges += 1
                        relation = pred_edge.get("relation", {})

                        if isinstance(relation, list):
                            for rel in relation:
                                relation_counts[rel] += 1
                        elif isinstance(relation, dict):
                            for rel_name, rel_value in relation.items():
                                if rel_value == 1:
                                    relation_counts[rel_name] += 1
            else:
                edges = pred_sg.get("edges", [])
                total_edges += len(edges)

                for edge in edges:
                    relation = edge.get("relation", {})

                    if isinstance(relation, list):
                        for rel in relation:
                            relation_counts[rel] += 1
                    elif isinstance(relation, dict):
                        for rel_name, rel_value in relation.items():
                            if rel_value == 1:
                                relation_counts[rel_name] += 1

    relation_counts["__total_edges__"] = total_edges
    if match_gt_only and not use_gt:
        relation_counts["__missing_edges__"] = missing_edges

    return dict(relation_counts)


def build_confusion_matrix_data(
    all_model_data: dict[str, list[tuple[dict, dict]]]
) -> dict[str, dict[str, np.ndarray]]:
    all_confusion_matrices = {}

    for model_name, scene_graphs in all_model_data.items():
        confusion_data = {
            relation_type: defaultdict(lambda: defaultdict(int))
            for relation_type in RELATION_PAIRS.keys()
        }

        for gt_sg, pred_sg in scene_graphs:
            for relation_type, relation_pair in RELATION_PAIRS.items():
                gt_labels, pred_labels = extract_relation_labels(gt_sg, pred_sg, relation_pair)

                for gt_label, pred_label in zip(gt_labels, pred_labels):
                    confusion_data[relation_type][gt_label][pred_label] += 1

        model_confusion_matrices = {}
        for relation_type in RELATION_PAIRS.keys():
            labels = LABEL_ORDERS[relation_type]
            n = len(labels)
            cm = np.zeros((n, n), dtype=int)

            for i, gt_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    cm[i, j] = confusion_data[relation_type][gt_label][pred_label]

            model_confusion_matrices[relation_type] = cm

        all_confusion_matrices[model_name] = model_confusion_matrices

    return all_confusion_matrices


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    output_path: str,
    cmap: str = "Blues"
):
    plt.figure(figsize=(10, 8))

    # Normalize by row
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype(float) / row_sums

    ax = sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Proportion'}
    )

    # Add actual counts
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j + 0.5, i + 0.7,
                f'({cm[i, j]})',
                ha="center", va="center",
                color="gray", fontsize=8
            )

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)

    # Move x-axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_display_model_name(model_name: str) -> str:
    """Extract display name after double underscore if present."""
    return model_name.split("__")[-1] if "__" in model_name else model_name


def visualize_all_confusion_matrices(
    all_confusion_matrices: dict[str, dict[str, np.ndarray]],
    aggregated_data: dict,
    output_dir: str
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Generating confusion matrix plots...")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}\n")

    total_plots = 0

    for model_key, matrices in all_confusion_matrices.items():
        model_info = aggregated_data.get(model_key, {})
        model_name = model_info.get("model", model_key)
        display_name = get_display_model_name(model_name)

        model_dir = output_path / model_key.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        for relation_type, cm in matrices.items():
            labels = LABEL_ORDERS[relation_type]
            relation_display = RELATION_DISPLAY_NAMES[relation_type]

            title = f"{display_name}\n{relation_display}"
            filename = f"{relation_type}_confusion_matrix.png"
            output_file = model_dir / filename

            plot_confusion_matrix(cm, labels, title, str(output_file))
            total_plots += 1

    print(f"\n{'='*60}")
    print(f"Completed! Generated {total_plots} confusion matrix plots")
    print(f"{'='*60}")


def print_debug_statistics(
    model_name: str,
    scene_graphs: list[tuple[dict, dict]],
    matrices: dict[str, np.ndarray]
):
    """Print detailed statistics for debugging."""
    print(f"\nModel: {model_name}")
    print(f"\n{'='*60}")
    print("Relation Distribution Statistics:")
    print(f"{'='*60}")

    gt_counts = count_relations(scene_graphs, use_gt=True)
    pred_counts_all = count_relations(scene_graphs, use_gt=False, match_gt_only=False)
    pred_counts_matched = count_relations(scene_graphs, use_gt=False, match_gt_only=True)

    total_edges = gt_counts.pop("__total_edges__", 0)
    print("\nGT Scene Graph Statistics:")
    print(f"  Total edges: {total_edges}")
    for rel_name in sorted(gt_counts.keys()):
        count = gt_counts[rel_name]
        pct = (count / total_edges * 100) if total_edges > 0 else 0
        print(f"  {rel_name:20s}: {count:5d} ({pct:5.2f}%)")

    total_pred = pred_counts_all.pop("__total_edges__", 0)
    print("\nPredicted Scene Graph Statistics (all edges):")
    print(f"  Total edges: {total_pred}")
    for rel_name in sorted(pred_counts_all.keys()):
        count = pred_counts_all[rel_name]
        pct = (count / total_pred * 100) if total_pred > 0 else 0
        print(f"  {rel_name:20s}: {count:5d} ({pct:5.2f}%)")

    total_matched = pred_counts_matched.pop("__total_edges__", 0)
    missing = pred_counts_matched.pop("__missing_edges__", 0)
    print("\nPredicted Scene Graph Statistics (matched edges only):")
    print(f"  Matched edges: {total_matched}")
    print(f"  Missing edges: {missing} ({missing/total_edges*100:.2f}%)" if total_edges > 0 else f"  Missing edges: {missing}")
    for rel_name in sorted(pred_counts_matched.keys()):
        count = pred_counts_matched[rel_name]
        pct = (count / total_matched * 100) if total_matched > 0 else 0
        print(f"  {rel_name:20s}: {count:5d} ({pct:5.2f}%)")

    print(f"\n{'='*60}")
    print("Confusion Matrix Details:")
    print(f"{'='*60}")

    for relation_type, cm in matrices.items():
        labels = LABEL_ORDERS[relation_type]
        print(f"\n{relation_type} confusion matrix:")
        print(f"Labels: {labels}")
        print(cm)

        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm.astype(float) / row_sums
        print("\nNormalized (by row):")
        np.set_printoptions(precision=3, suppress=True)
        print(cm_normalized)


def load_all_model_data(aggregated_file: str) -> tuple[dict[str, list[tuple[dict, dict]]], dict]:
    aggregated_data = load_aggregated_results(aggregated_file)
    source_files = extract_source_files(aggregated_data)

    all_model_data = {}

    for model_name, source_file in source_files.items():
        print(f"Processing model: {model_name}")

        samples = load_sample_data(source_file)
        if not samples:
            print("  Skipped: No sample data\n")
            continue

        scene_graphs = extract_scene_graphs(samples)
        if scene_graphs:
            all_model_data[model_name] = scene_graphs
            print(f"  Loaded {len(scene_graphs)} scene graph pairs\n")

    return all_model_data, aggregated_data


def main():
    parser = argparse.ArgumentParser(
        description="Visualize confusion matrices for scene graph spatial relations"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to aggregated results JSON file (aggregated.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./confusion_matrices",
        help="Output directory for confusion matrix images"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed statistics and validation information"
    )
    args = parser.parse_args()

    print("="*60)
    print("Scene Graph Confusion Matrix Visualization Tool")
    print("="*60)

    try:
        all_model_data, aggregated_data = load_all_model_data(args.input)

        print(f"\n{'='*60}")
        print("Building confusion matrices...")
        print(f"{'='*60}")

        all_confusion_matrices = build_confusion_matrix_data(all_model_data)

        if args.debug and all_confusion_matrices:
            print(f"\n{'='*60}")
            print("Debug Information (first model only):")
            print(f"{'='*60}")

            first_model = list(all_confusion_matrices.keys())[0]
            print_debug_statistics(
                first_model,
                all_model_data[first_model],
                all_confusion_matrices[first_model]
            )

        visualize_all_confusion_matrices(all_confusion_matrices, aggregated_data, args.output_dir)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
