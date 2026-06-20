#!/usr/bin/env python3
"""
Batch consistency evaluation for CRISP-Bench.

Automates the three-step pipeline (extract_sg → scene_graph_solver →
calculate_consistency) across all models discovered in a log directory.

File pairing strategy
---------------------
1. Scan ``--logs_dir`` for SGC sample files (``*_samples_crisp_sgc*.jsonl``).
2. Determine task family from the SGC file name:
   * ``crisp_sgc``     → **base** family — paired with ``crisp_qa``
   * ``crisp_sgc_cot`` → **cot**  family — paired with ``crisp_qa_cot``
3. For each model directory × family, independently select the **latest**
   SGC file and the **latest** QA file (by timestamp prefix) and pair them.
   This is the default because users may not evaluate both tasks at the
   same time.
4. Each valid (SGC, QA) pair forms one evaluation job.

Usage::

    # Dry-run: show discovered pairs without executing
    python evaluation/run_consistency_eval.py \\
        --logs_dir lmms-eval/logs \\
        --output_dir results/consistency_score \\
        --dry_run

    # Full run
    python evaluation/run_consistency_eval.py \\
        --logs_dir lmms-eval/logs \\
        --output_dir results/consistency_score
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import the three pipeline stages
# ---------------------------------------------------------------------------

# Ensure the repo root is on sys.path so that absolute imports work
# even when invoked from an arbitrary cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.calculate_consistency import (
    ConsistencyEvaluator,
    load_direct_qa_map,
)
from evaluation.extract_sg import process_vlm_output
from evaluation.scene_graph_solver import SceneGraphSolver, TemplateMatcher

# ---------------------------------------------------------------------------
# Task-family definitions
# ---------------------------------------------------------------------------

# Maps SGC task name → corresponding direct-QA task name
_FAMILY_MAP: dict[str, str] = {
    "crisp_sgc": "crisp_qa",
    "crisp_sgc_cot": "crisp_qa_cot",
}

_SAMPLES_RE = re.compile(
    r"^(\d{8}_\d{6})_samples_(crisp_.+)\.jsonl$"
)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _discover_log_files(
    logs_dir: str,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Walk *logs_dir* and return a nested mapping:

        model_dir → timestamp → task_name → [file_path, …]

    Only files matching ``*_samples_crisp_*.jsonl`` are collected.
    """
    tree: dict[str, dict[str, dict[str, list[str]]]] = {}
    for root, _dirs, files in os.walk(logs_dir):
        for fname in files:
            m = _SAMPLES_RE.match(fname)
            if not m:
                continue
            ts, task = m.groups()
            fpath = os.path.join(root, fname)
            tree.setdefault(root, {}).setdefault(ts, {}).setdefault(task, []).append(fpath)
    return tree


def _build_pairs(
    tree: dict[str, dict[str, dict[str, list[str]]]],
) -> list[dict[str, Any]]:
    """Given the discovery tree, produce valid (SGC, QA) pair descriptors.

    For each model directory × family, independently select the **latest**
    SGC file and the **latest** QA file (by timestamp prefix) and pair them.
    """
    pairs: list[dict[str, Any]] = []

    for model_dir, ts_map in sorted(tree.items()):
        # Determine model label from directory path
        # e.g., …/logs/GPT5/gpt-5-mini → "GPT5/gpt-5-mini"
        parts = Path(model_dir).parts
        try:
            logs_idx = parts.index("logs")
            model_label = "/".join(parts[logs_idx + 1 :])
        except ValueError:
            model_label = Path(model_dir).name

        for sgc_task, qa_task in _FAMILY_MAP.items():
            # Collect all SGC files for this family, pick the latest
            sgc_candidates: list[tuple[str, str]] = []
            for ts, task_map in ts_map.items():
                if sgc_task in task_map:
                    for f in task_map[sgc_task]:
                        sgc_candidates.append((ts, f))

            if not sgc_candidates:
                continue

            sgc_candidates.sort(key=lambda x: x[0], reverse=True)
            sgc_ts, sgc_file = sgc_candidates[0]

            # Collect all QA files for the corresponding family, pick the latest
            qa_candidates: list[tuple[str, str]] = []
            for ts, task_map in ts_map.items():
                if qa_task in task_map:
                    for f in task_map[qa_task]:
                        qa_candidates.append((ts, f))

            if not qa_candidates:
                continue

            qa_candidates.sort(key=lambda x: x[0], reverse=True)
            qa_ts, qa_file = qa_candidates[0]

            family = "base" if sgc_task == "crisp_sgc" else "cot"
            pairs.append(
                {
                    "model_label": model_label,
                    "model_dir": model_dir,
                    "family": family,
                    "sgc_task": sgc_task,
                    "qa_task": qa_task,
                    "sgc_file": sgc_file,
                    "qa_file": qa_file,
                    "sgc_ts": sgc_ts,
                    "qa_ts": qa_ts,
                }
            )

    return pairs


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _run_pipeline(
    sgc_file: str,
    qa_file: str,
    qa_list_path: str,
    templates_path: str,
    output_dir: str,
    model_label: str,
    family: str,
) -> dict[str, Any]:
    """Execute the three-step consistency pipeline for one (SGC, QA) pair.

    Returns a summary dict with accuracy/consistency scores.
    """
    safe_label = model_label.replace("/", "_").replace(" ", "_")
    prefix = f"{safe_label}_{family}"

    sg_json_path = os.path.join(output_dir, "generated_sg", f"{prefix}_sg.json")
    derived_qa_path = os.path.join(output_dir, "generated_sg", f"{prefix}_qa.jsonl")
    results_path = os.path.join(output_dir, f"{prefix}_consistency_results.json")

    os.makedirs(os.path.dirname(sg_json_path), exist_ok=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # ---- Step 1: extract scene graphs ----
    print(f"\n{'='*60}")
    print(f"[Step 1/3] Extracting scene graphs: {os.path.basename(sgc_file)}")
    process_vlm_output(sgc_file, sg_json_path)

    # ---- Step 2: solve QA with extracted SGs ----
    print("[Step 2/3] Solving QA with scene graphs ...")
    with open(sg_json_path) as f:
        pred_sgs = json.load(f)
    with open(templates_path) as f:
        templates = json.load(f)

    # Build lookup by basename for robust matching.
    # extract_sg now writes relative paths, but historical SG files may still
    # contain absolute paths.  Image filenames are unique hashes, so basename
    # matching is safe and handles both cases.
    pred_lookup: dict[str, dict] = {}
    for sg in pred_sgs:
        if "image" in sg:
            pred_lookup[os.path.basename(sg["image"])] = sg

    matcher = TemplateMatcher(templates)
    results_buffer: list[dict] = []
    missing_count = 0
    total_count = 0

    with open(qa_list_path) as f:
        for line in f:
            if not line.strip():
                continue
            qa_item = json.loads(line)
            total_count += 1

            image_key = os.path.basename(qa_item.get("image", ""))
            pred_sg = pred_lookup.get(image_key)

            if not pred_sg:
                prediction = "FAILED: Scene Graph Missing"
                missing_count += 1
            else:
                solver = SceneGraphSolver(pred_sg, matcher)
                prediction = solver.solve(qa_item)
                if prediction == SceneGraphSolver.MISSING_LABEL:
                    missing_count += 1

            qa_item["predict"] = prediction
            results_buffer.append(qa_item)

    with open(derived_qa_path, "w") as f:
        for item in results_buffer:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Solved {total_count} items, {missing_count} failed.")

    # ---- Step 3: calculate consistency ----
    print("[Step 3/3] Calculating consistency ...")
    direct_map = load_direct_qa_map(qa_file)

    evaluator = ConsistencyEvaluator(debug_mode=False)
    missing_direct = 0

    for item in results_buffer:
        item_id = item.get("id")
        meta = item.get("meta", {})

        gt_val = None
        for t in item.get("conversations", []):
            if t["from"] == "gpt":
                gt_val = t["value"]
                break

        derived_val = item.get("predict")

        direct_row = direct_map.get(item_id)
        direct_val = None
        if direct_row:
            if "filtered_resps" in direct_row and direct_row["filtered_resps"]:
                direct_val = direct_row["filtered_resps"][0]
            elif "prediction" in direct_row:
                direct_val = direct_row["prediction"]
        else:
            missing_direct += 1

        if gt_val is None or derived_val is None:
            continue

        evaluator.evaluate_item(
            item_id=item_id,
            meta=meta,
            gt=gt_val,
            direct=direct_val,
            derived=derived_val,
        )

    if missing_direct > 0:
        print(f"  Warning: {missing_direct} items missing in Direct QA file.")

    accuracy_summary = evaluator.aggregate_and_print(
        evaluator.results_accuracy, f"Accuracy ({model_label} / {family})"
    )
    consistency_summary = evaluator.aggregate_and_print(
        evaluator.results_consistency, f"Consistency ({model_label} / {family})"
    )
    evaluator.save_results(results_path, accuracy_summary, consistency_summary)

    return {
        "model": model_label,
        "family": family,
        "accuracy_overall": accuracy_summary["overall_score"] if accuracy_summary else None,
        "consistency_overall": consistency_summary["overall_score"] if consistency_summary else None,
        "results_file": results_path,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(summaries: list[dict[str, Any]]) -> None:
    """Print a compact cross-model summary table."""
    if not summaries:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Model':<40} {'Family':<8} {'Accuracy':>10} {'Consistency':>12}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        acc = f"{s['accuracy_overall']:.4f}" if s["accuracy_overall"] is not None else "N/A"
        con = f"{s['consistency_overall']:.4f}" if s["consistency_overall"] is not None else "N/A"
        print(f"{s['model']:<40} {s['family']:<8} {acc:>10} {con:>12}")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for batch consistency evaluation across all models."""
    ap = argparse.ArgumentParser(
        description="Batch consistency evaluation across all models.",
    )
    ap.add_argument(
        "--logs_dir",
        required=True,
        help="Root of lmms-eval logs (e.g. lmms-eval/logs).",
    )
    ap.add_argument(
        "--output_dir",
        default="results/consistency_score",
        help="Directory for consistency result JSONs (default: results/consistency_score).",
    )
    ap.add_argument(
        "--qa_list",
        default=None,
        help="Path to QA dataset JSONL for the solver "
        "(default: $COMBINED_ROOT/QA_pairs/qa_data.jsonl).",
    )
    ap.add_argument(
        "--templates",
        default=None,
        help="Path to question templates JSON "
        "(default: qa/question_template.json).",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print discovered pairs, do not execute.",
    )
    args = ap.parse_args()

    # Resolve default paths relative to repo root
    repo_root = str(_REPO_ROOT)

    if args.qa_list is None:
        datasets_root = os.environ.get(
            "DATASETS_ROOT", os.path.join(repo_root, "data", "processed")
        )
        combined_root = os.environ.get(
            "COMBINED_ROOT", os.path.join(datasets_root, "combined")
        )
        args.qa_list = os.path.join(combined_root, "QA_pairs", "qa_data.jsonl")

    if args.templates is None:
        args.templates = os.path.join(repo_root, "qa", "question_template.json")

    # Validate required files
    for label, path in [("QA list", args.qa_list), ("Templates", args.templates)]:
        if not os.path.isfile(path):
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Discover & pair
    print(f"Scanning {args.logs_dir} ...")
    tree = _discover_log_files(args.logs_dir)
    pairs = _build_pairs(tree)

    if not pairs:
        print("No valid (SGC, QA) pairs found.")
        sys.exit(0)

    print(f"\nDiscovered {len(pairs)} evaluation job(s):\n")
    for i, p in enumerate(pairs, 1):
        print(f"  [{i:2d}] {p['model_label']} ({p['family']})")
        print(f"       SGC: {os.path.basename(p['sgc_file'])}  (ts: {p['sgc_ts']})")
        print(f"       QA:  {os.path.basename(p['qa_file'])}  (ts: {p.get('qa_ts', 'N/A')})")

    if args.dry_run:
        print("\n--dry_run: exiting without executing.")
        return

    # Execute
    summaries: list[dict[str, Any]] = []
    for p in pairs:
        try:
            summary = _run_pipeline(
                sgc_file=p["sgc_file"],
                qa_file=p["qa_file"],
                qa_list_path=args.qa_list,
                templates_path=args.templates,
                output_dir=args.output_dir,
                model_label=p["model_label"],
                family=p["family"],
            )
            summaries.append(summary)
        except Exception as exc:
            print(f"\n[ERROR] {p['model_label']} ({p['family']}): {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    _print_summary(summaries)

    # Save summary JSON
    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nBatch summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
