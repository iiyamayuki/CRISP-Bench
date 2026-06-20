#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.io import load_json as load_json_file
from utils.io import save_json

TARGET_TASKS = ["crisp_qa_sg"]

REQUIRED: set[tuple[str, str, str]] = {
    ("crisp_qa_sg", "multimodal", "gt"),
    ("crisp_qa_sg", "multimodal", "pred"),
    ("crisp_qa_sg", "text_only", "gt"),
    ("crisp_qa_sg", "text_only", "pred"),
}

TS_RE = re.compile(r"^(\d{8})_(\d{6})(?:_.*)?_results\.json$", re.IGNORECASE)
MRA_TAIL_RE = re.compile(r":\.5:\.95:\.05$")


@dataclass
class Row:
    suite: str
    model: str
    task: str
    variant: str
    sg_type: str     # gt or pred
    metric: str      # shortened metric name
    value: Any
    timestamp: str
    source_file: str


def load_json(path: str) -> dict[str, Any] | None:
    try:
        obj = load_json_file(path)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def find_key_recursive(obj: Any, key: str) -> Any | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = find_key_recursive(it, key)
            if found is not None:
                return found
    return None


def infer_variant(data: dict[str, Any]) -> str:
    v = find_key_recursive(data, "doc_to_visual")
    if isinstance(v, str) and "text_only" in v:
        return "text_only"
    return "multimodal"


def infer_sg_type(data: dict[str, Any]) -> str:
    v = find_key_recursive(data, "doc_to_text")
    if isinstance(v, str) and "pred" in v.lower():
        return "pred"
    return "gt"


def parse_timestamp_from_filename(fname: str) -> str | None:
    m = TS_RE.match(fname)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


def list_results_files(model_dir: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for fn in os.listdir(model_dir):
        if not fn.lower().endswith("_results.json"):
            continue
        ts = parse_timestamp_from_filename(fn)
        if ts is None:
            try:
                mt = os.path.getmtime(os.path.join(model_dir, fn))
                ts = datetime.fromtimestamp(mt).strftime("%Y%m%d_%H%M%S")
            except Exception:
                ts = "00000000_000000"
        out.append((ts, os.path.join(model_dir, fn)))
    return out


def shorten_metric_name(full_key: str) -> str:
    """
    Examples:
      vsibench_score,none/MCQ_deduction_accuracy -> MCQ_deduction
    vsibench_score,none/NA_distance_MRA:.5:.95:.05 -> NAQ_distance_MRA
      vsibench_score,none/difficulty_easy_mean -> difficulty_easy
    """
    k = full_key.split("/")[-1] if "/" in full_key else full_key

    if k.endswith("_accuracy"):
        k = k[:-len("_accuracy")]
    if k.endswith("_mean"):
        k = k[:-len("_mean")]

    k = MRA_TAIL_RE.sub("", k)
    if k.startswith("NA_"):
        k = f"NAQ_{k[len('NA_'):]}"
    if k.startswith("consistency_NA_"):
        k = f"consistency_NAQ_{k[len('consistency_NA_'):]}"
    return k


def flatten_task_metrics(task_obj: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in task_obj.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                short = shorten_metric_name(f"{k}/{subk}")
                if short in flat:
                    group = k.split(",")[0] if isinstance(k, str) else "metric"
                    short = f"{group}.{short}"
                flat[short] = subv
    return flat


def extract_from_file(path: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    data = load_json(path)
    if not data:
        return {}

    results = data.get("results")
    if not isinstance(results, dict):
        return {}

    variant = infer_variant(data)
    sg_type = infer_sg_type(data)
    extracted: dict[tuple[str, str, str], dict[str, Any]] = {}

    for task in TARGET_TASKS:
        obj = results.get(task)
        if not isinstance(obj, dict):
            continue
        extracted[(task, variant, sg_type)] = flatten_task_metrics(obj)

    return extracted


def detect_model_dirs(logs_root: str) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    logs_root = os.path.abspath(logs_root)

    for suite in os.listdir(logs_root):
        suite_dir = os.path.join(logs_root, suite)
        if not os.path.isdir(suite_dir):
            continue
        for model in os.listdir(suite_dir):
            model_dir = os.path.join(suite_dir, model)
            if not os.path.isdir(model_dir):
                continue
            has = any(fn.lower().endswith("_results.json") for fn in os.listdir(model_dir))
            if has:
                found.append((suite, model, model_dir))
    return found


def write_csv(rows: list[Row], out_path: str) -> None:
    fieldnames = ["suite", "model", "task", "variant", "sg_type", "metric", "value", "timestamp", "source_file"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            d["value"] = json.dumps(d["value"], ensure_ascii=False) if isinstance(d["value"], (dict, list)) else str(d["value"])
            w.writerow(d)


def write_json(obj: dict[str, Any], out_path: str) -> None:
    save_json(obj, out_path)


def metric_sort_key(m: str) -> tuple[int, str]:
    if m.startswith("MCQ_"):
        return (0, m)
    if m.startswith("NAQ_"):
        return (1, m)
    if m.startswith("difficulty_"):
        return (2, m)
    if m in ("overall", "SGC_Score"):
        return (3, m)
    return (9, m)


def make_markdown_tables_transposed(nested: dict[str, Any]) -> str:
    """
    For each (task, variant, sg_type):
      rows = metrics
      cols = models (display name only)
    """
    tvs_set: set[tuple[str, str, str]] = set()
    for model_key, model_obj in nested.items():
        for task, task_obj in model_obj.get("tasks", {}).items():
            for variant, variant_obj in task_obj.items():
                for sg_type in variant_obj.keys():
                    tvs_set.add((task, variant, sg_type))

    # display name: only model name; disambiguate duplicates by (suite)
    display_name: dict[str, str] = {}
    name_counts: dict[str, int] = {}
    for model_key, model_obj in nested.items():
        nm = str(model_obj.get("model", model_key))
        name_counts[nm] = name_counts.get(nm, 0) + 1

    for model_key, model_obj in nested.items():
        nm = str(model_obj.get("model", model_key))
        st = str(model_obj.get("suite", ""))
        if name_counts.get(nm, 0) > 1 and st:
            display_name[model_key] = f"{nm} ({st})"
        else:
            display_name[model_key] = nm

    parts: list[str] = []
    for task, variant, sg_type in sorted(tvs_set):
        table: dict[str, dict[str, Any]] = {}
        metrics_all: set[str] = set()

        for model_key, model_obj in nested.items():
            tasks = model_obj.get("tasks", {})
            if task not in tasks:
                continue
            if variant not in tasks[task]:
                continue
            if sg_type not in tasks[task][variant]:
                continue
            metrics = tasks[task][variant][sg_type].get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            table[model_key] = metrics
            metrics_all.update(metrics.keys())

        metrics_list = sorted(metrics_all, key=metric_sort_key)
        model_keys_sorted = sorted(table.keys(), key=lambda k: display_name.get(k, k).lower())

        parts.append(f"## {task} ({variant}, {sg_type})\n")
        if not table:
            parts.append("_No data._\n")
            continue

        header = ["metric"] + [display_name[k] for k in model_keys_sorted]
        parts.append("| " + " | ".join(header) + " |")
        parts.append("| " + " | ".join(["---"] * len(header)) + " |")

        for met in metrics_list:
            row = [met]
            for mk in model_keys_sorted:
                v = table[mk].get(met, "")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    row.append(f"{v * 100:.2f}".rstrip("0").rstrip("."))
                else:
                    row.append("" if v is None else str(v))
            parts.append("| " + " | ".join(row) + " |")

        parts.append("")

    return "\n".join(parts).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Path to lmms-eval logs root (contains suite/model folders).")
    ap.add_argument("--output_dir", default="aggregated_qa_sg_out", help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_dirs = detect_model_dirs(args.input_dir)
    if not model_dirs:
        print("No model dirs found. Check --input_dir.")
        return

    all_rows: list[Row] = []
    nested: dict[str, Any] = {}
    missing_report: dict[str, list[str]] = {}

    # Collect crisp_qa_sg task results
    for suite, model, model_dir in sorted(model_dirs):
        model_key = f"{suite}/{model}"

        files = list_results_files(model_dir)
        files.sort(key=lambda x: x[0], reverse=True)  # timestamp desc

        collected: dict[tuple[str, str, str], dict[str, Any]] = {}
        collected_src: dict[tuple[str, str, str], tuple[str, str]] = {}

        for ts, path in files:
            extracted = extract_from_file(path)
            if not extracted:
                continue

            for tvs, metrics in extracted.items():
                if tvs in collected:
                    continue
                collected[tvs] = metrics
                collected_src[tvs] = (ts, path)

            if REQUIRED.issubset(set(collected.keys())):
                break

        nested[model_key] = {"suite": suite, "model": model, "tasks": {}}

        for (task, variant, sg_type), metrics in collected.items():
            ts, src = collected_src[(task, variant, sg_type)]
            nested[model_key]["tasks"].setdefault(task, {})
            nested[model_key]["tasks"][task].setdefault(variant, {})
            nested[model_key]["tasks"][task][variant][sg_type] = {
                "timestamp": ts,
                "source_file": src,
                "metrics": metrics,
            }

            for metric_name, value in metrics.items():
                all_rows.append(Row(
                    suite=suite,
                    model=model,
                    task=task,
                    variant=variant,
                    sg_type=sg_type,
                    metric=metric_name,
                    value=value,
                    timestamp=ts,
                    source_file=src,
                ))

        missing = sorted(list(REQUIRED - set(collected.keys())))
        if missing:
            missing_report[model_key] = [f"{t}:{v}:{s}" for (t, v, s) in missing]

    # Write outputs
    csv_path = os.path.join(args.output_dir, "aggregated_qa_sg_long.csv")
    json_path = os.path.join(args.output_dir, "aggregated_qa_sg.json")
    md_path = os.path.join(args.output_dir, "aggregated_qa_sg.md")
    missing_path = os.path.join(args.output_dir, "missing_qa_sg.json")

    write_csv(all_rows, csv_path)
    write_json(nested, json_path)

    md = make_markdown_tables_transposed(nested)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    write_json(missing_report, missing_path)

    print(f"[OK] CSV  : {csv_path}  (rows={len(all_rows)})")
    print(f"[OK] JSON : {json_path}")
    print(f"[OK] MD   : {md_path}")
    if missing_report:
        print(f"[WARN] Missing experiments for some models. See: {missing_path}")
    else:
        print("[OK] All models have the full 4 experiments (crisp_qa_sg variants).")


if __name__ == "__main__":
    main()
