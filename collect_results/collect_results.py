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

TARGET_TASKS = ["crisp_qa", "crisp_sgc"]

REQUIRED: set[tuple[str, str]] = {
    ("crisp_qa", "multimodal"),
    ("crisp_qa", "text_only"),
    ("crisp_sgc", "multimodal"),
    ("crisp_sgc", "text_only"),
}

TS_RE = re.compile(r"^(\d{8})_(\d{6})_.*_results\.json$", re.IGNORECASE)
MRA_TAIL_RE = re.compile(r":\.5:\.95:\.05$")

# Consistency score filename heuristics:
# e.g. qwen3vl_8B_consistency_results.json -> base model token = qwen3vl_8B
CONSIST_NAME_RE = re.compile(r"^(?P<base>.+?)_consistency.*\.json$", re.IGNORECASE)


@dataclass
class Row:
    suite: str
    model: str
    task: str
    variant: str
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
      sgc_score,none/S_size -> S_size
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


def extract_from_file(path: str) -> dict[tuple[str, str], dict[str, Any]]:
    data = load_json(path)
    if not data:
        return {}

    results = data.get("results")
    if not isinstance(results, dict):
        return {}

    variant = infer_variant(data)
    extracted: dict[tuple[str, str], dict[str, Any]] = {}

    for task in TARGET_TASKS:
        obj = results.get(task)
        if not isinstance(obj, dict):
            continue
        extracted[(task, variant)] = flatten_task_metrics(obj)

    return extracted


def detect_model_dirs(logs_root: str) -> list[tuple[str, str, str]]:
    """
    logs_root/<suite>/<model>/*.json
    """
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
    fieldnames = ["suite", "model", "task", "variant", "metric", "value", "timestamp", "source_file"]
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
    # For consistency_score metrics we use prefixes:
    if m.startswith("accuracy_"):
        return (4, m)
    if m.startswith("consistency_"):
        return (5, m)
    return (9, m)


def make_markdown_tables_transposed(nested: dict[str, Any]) -> str:
    """
    For each (task, variant):
      rows = metrics
      cols = models (display name only)
    """
    tv_set: set[tuple[str, str]] = set()
    for model_key, model_obj in nested.items():
        for task, task_obj in model_obj.get("tasks", {}).items():
            for variant in task_obj.keys():
                tv_set.add((task, variant))

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
    for task, variant in sorted(tv_set):
        table: dict[str, dict[str, Any]] = {}
        metrics_all: set[str] = set()

        for model_key, model_obj in nested.items():
            tasks = model_obj.get("tasks", {})
            if task not in tasks:
                continue
            if variant not in tasks[task]:
                continue
            metrics = tasks[task][variant].get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            table[model_key] = metrics
            metrics_all.update(metrics.keys())

        metrics_list = sorted(metrics_all, key=metric_sort_key)
        model_keys_sorted = sorted(table.keys(), key=lambda k: display_name.get(k, k).lower())

        parts.append(f"## {task} ({variant})\n")
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


# -------------------------
# Consistency score support
# -------------------------

def normalize_name(s: str) -> str:
    """Lowercase and keep only [a-z0-9]."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_global_sgc_dir(input_dir: str, output_dir: str, provided: str | None) -> str | None:
    """
    Try to locate global_sgc folder containing evaluation results.
    Priority:
      1) --global_sgc_dir (if provided)
      2) output_dir/global_sgc
      3) dirname(output_dir)/global_sgc
      4) input_dir/global_sgc
    """
    candidates: list[str] = []
    if provided:
        candidates.append(provided)
    candidates.append(os.path.join(output_dir, "global_sgc"))
    candidates.append(os.path.join(os.path.dirname(output_dir), "global_sgc"))
    candidates.append(os.path.join(input_dir, "global_sgc"))

    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return None


def find_consistency_dir(input_dir: str, output_dir: str, provided: str | None) -> str | None:
    """
    Try to locate consistency_score folder.
    Priority:
      1) --consistency_dir (if provided)
      2) output_dir/consistency_score
      3) dirname(output_dir)/consistency_score
      4) input_dir/consistency_score
    """
    candidates: list[str] = []
    if provided:
        candidates.append(provided)
    candidates.append(os.path.join(output_dir, "consistency_score"))
    candidates.append(os.path.join(os.path.dirname(output_dir), "consistency_score"))
    candidates.append(os.path.join(input_dir, "consistency_score"))

    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return None


def parse_consistency_filename(path: str) -> tuple[str, str]:
    """
    Returns (base_token, timestamp).
    timestamp uses mtime fallback.
    """
    fn = os.path.basename(path)
    base = os.path.splitext(fn)[0]
    m = CONSIST_NAME_RE.match(fn)
    if m:
        base = m.group("base")

    try:
        mt = os.path.getmtime(path)
        ts = datetime.fromtimestamp(mt).strftime("%Y%m%d_%H%M%S")
    except Exception:
        ts = "00000000_000000"
    return base, ts


def flatten_global_sgc_json(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten Global SGC evaluation results into short metrics:
      S_size, S_dist_cam, S_dist, S_rel, SGC_Score, json_valid
    """
    out: dict[str, Any] = {}

    # Extract metrics from the nested structure
    metrics = obj.get("metrics", {})
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            out[key] = value

    return out


def collect_global_sgc_scores(sgc_dir: str, nested: dict[str, Any], all_rows: list[Row]) -> None:
    """
    Read all json in sgc_dir, pick latest per base token, and merge into nested/all_rows.
    Task name: crisp_sgc_pairwise (global)
    """
    # group by base token (extract from filename before _eval or similar suffixes)
    groups: dict[str, list[tuple[str, str]]] = {}  # base -> [(ts,path)]

    for fn in os.listdir(sgc_dir):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(sgc_dir, fn)
        if not os.path.isfile(path):
            continue

        # Extract base name (remove _eval.json or similar suffixes)
        base = os.path.splitext(fn)[0]
        # Try to extract model name from filename
        # Typically: <model>_eval.json or <model>_global_sgc.json
        base = re.sub(r"_(eval|global_sgc|results?)$", "", base, flags=re.IGNORECASE)

        try:
            mt = os.path.getmtime(path)
            ts = datetime.fromtimestamp(mt).strftime("%Y%m%d_%H%M%S")
        except Exception:
            ts = "00000000_000000"

        groups.setdefault(base, []).append((ts, path))

    for base, items in groups.items():
        # take latest by ts
        items.sort(key=lambda x: x[0], reverse=True)
        ts, path = items[0]

        obj = load_json(path)
        if not obj:
            continue

        metrics = flatten_global_sgc_json(obj)

        # map to existing model key if possible
        mk = best_match_model_key(base, nested)
        if mk is None:
            # create a new entry if no match
            mk = f"global_sgc/{base}"
            if mk not in nested:
                nested[mk] = {"suite": "global_sgc", "model": base, "tasks": {}}

        task_name = "crisp_sgc_pairwise (global)"
        nested[mk]["tasks"].setdefault(task_name, {})
        nested[mk]["tasks"][task_name]["multimodal"] = {
            "timestamp": ts,
            "source_file": path,
            "metrics": metrics,
        }

        suite = str(nested[mk].get("suite", "global_sgc"))
        model = str(nested[mk].get("model", base))

        for metric_name, value in metrics.items():
            all_rows.append(Row(
                suite=suite,
                model=model,
                task=task_name,
                variant="multimodal",
                metric=metric_name,
                value=value,
                timestamp=ts,
                source_file=path,
            ))


def flatten_consistency_json(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten into short metrics:

    accuracy:
      accuracy_overall
      accuracy_valid_rate
    accuracy_MCQ_count, accuracy_NAQ_distance, ...
      accuracy_difficulty_easy, ...

    consistency:
      consistency_overall
      consistency_valid_rate (usually null)
      consistency_MCQ_count, ...
      consistency_difficulty_easy, ...
    """
    out: dict[str, Any] = {}

    def handle_section(section_name: str) -> None:
        sec = obj.get(section_name)
        if not isinstance(sec, dict):
            return

        overall = sec.get("overall_score")
        if overall is not None:
            out[f"{section_name}_overall"] = overall

        valid_rate = sec.get("valid_rate")
        # keep even if None to show missing explicitly in json; md will render empty
        out[f"{section_name}_valid_rate"] = valid_rate

        by_cat = sec.get("by_category")
        if isinstance(by_cat, list):
            for it in by_cat:
                if not isinstance(it, dict):
                    continue
                t = it.get("type")
                cat = it.get("category")
                score = it.get("score_mean")
                if isinstance(t, str) and isinstance(cat, str):
                    out[f"{section_name}_{t}_{cat}"] = score

        by_diff = sec.get("by_difficulty")
        if isinstance(by_diff, list):
            for it in by_diff:
                if not isinstance(it, dict):
                    continue
                diff = it.get("difficulty")
                score = it.get("score")
                if isinstance(diff, str):
                    out[f"{section_name}_difficulty_{diff}"] = score

    handle_section("accuracy")
    handle_section("consistency")
    return out


def best_match_model_key(base_token: str, nested: dict[str, Any]) -> str | None:
    """
    Try to map a consistency file base token to an existing model_key in nested
    using normalized name containment.

    Returns the best matching model_key or None.
    """
    nb = normalize_name(base_token)
    if not nb:
        return None

    best_key = None
    best_score = -1

    for model_key, model_obj in nested.items():
        model_name = str(model_obj.get("model", model_key))
        nm = normalize_name(model_name)

        if not nm:
            continue

        # containment-based score
        if nb in nm:
            score = len(nb)
        elif nm in nb:
            score = len(nm)
        else:
            continue

        # prefer longer matches
        if score > best_score:
            best_score = score
            best_key = model_key

    return best_key


def collect_consistency_scores(cons_dir: str, nested: dict[str, Any], all_rows: list[Row]) -> None:
    """
    Read all json in cons_dir, pick latest per base token, and merge into nested/all_rows.
    """
    # group by base token
    groups: dict[str, list[tuple[str, str]]] = {}  # base -> [(ts,path)]
    for fn in os.listdir(cons_dir):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(cons_dir, fn)
        if not os.path.isfile(path):
            continue
        base, ts = parse_consistency_filename(path)
        groups.setdefault(base, []).append((ts, path))

    for base, items in groups.items():
        # take latest by ts
        items.sort(key=lambda x: x[0], reverse=True)
        ts, path = items[0]

        obj = load_json(path)
        if not obj:
            continue

        metrics = flatten_consistency_json(obj)

        # map to existing model key if possible
        mk = best_match_model_key(base, nested)
        if mk is None:
            # create a new entry if no match
            mk = f"consistency_score/{base}"
            if mk not in nested:
                nested[mk] = {"suite": "consistency_score", "model": base, "tasks": {}}

        nested[mk]["tasks"].setdefault("consistency_score", {})
        nested[mk]["tasks"]["consistency_score"]["all"] = {
            "timestamp": ts,
            "source_file": path,
            "metrics": metrics,
        }

        suite = str(nested[mk].get("suite", "consistency_score"))
        model = str(nested[mk].get("model", base))

        for metric_name, value in metrics.items():
            all_rows.append(Row(
                suite=suite,
                model=model,
                task="consistency_score",
                variant="all",
                metric=metric_name,
                value=value,
                timestamp=ts,
                source_file=path,
            ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Path to lmms-eval logs root (contains suite/model folders).")
    ap.add_argument("--output_dir", default="aggregated_out", help="Output directory.")
    ap.add_argument("--consistency_dir", default="", help="Optional: path to consistency_score folder.")
    ap.add_argument("--global_sgc_dir", default="", help="Optional: path to global_sgc evaluation results folder.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_dirs = detect_model_dirs(args.input_dir)
    if not model_dirs:
        print("No model dirs found. Check --input_dir.")
        return

    all_rows: list[Row] = []
    nested: dict[str, Any] = {}
    missing_report: dict[str, list[str]] = {}

    # 1) Collect lmms-eval task results
    for suite, model, model_dir in sorted(model_dirs):
        model_key = f"{suite}/{model}"

        files = list_results_files(model_dir)
        files.sort(key=lambda x: x[0], reverse=True)  # timestamp desc

        collected: dict[tuple[str, str], dict[str, Any]] = {}
        collected_src: dict[tuple[str, str], tuple[str, str]] = {}

        for ts, path in files:
            extracted = extract_from_file(path)
            if not extracted:
                continue

            for tv, metrics in extracted.items():
                if tv in collected:
                    continue
                collected[tv] = metrics
                collected_src[tv] = (ts, path)

            if REQUIRED.issubset(set(collected.keys())):
                break

        nested[model_key] = {"suite": suite, "model": model, "tasks": {}}

        for (task, variant), metrics in collected.items():
            ts, src = collected_src[(task, variant)]
            nested[model_key]["tasks"].setdefault(task, {})
            nested[model_key]["tasks"][task][variant] = {
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
                    metric=metric_name,
                    value=value,
                    timestamp=ts,
                    source_file=src,
                ))

        missing = sorted(list(REQUIRED - set(collected.keys())))
        if missing:
            missing_report[model_key] = [f"{t}:{v}" for (t, v) in missing]

    # 2) Collect optional pairwise/global SGC results and merge.
    if args.global_sgc_dir:
        sgc_dir = find_global_sgc_dir(args.input_dir, args.output_dir, args.global_sgc_dir)
        if sgc_dir:
            collect_global_sgc_scores(sgc_dir, nested, all_rows)
        else:
            print("[INFO] Provided global_sgc folder not found; skipping optional pairwise collection.")

    # 3) Collect consistency_score results and merge
    cons_dir = find_consistency_dir(args.input_dir, args.output_dir, args.consistency_dir or None)
    if cons_dir:
        collect_consistency_scores(cons_dir, nested, all_rows)
    else:
        # Not fatal
        print("[INFO] consistency_score folder not found; skipping consistency collection.")

    # 4) Write outputs
    csv_path = os.path.join(args.output_dir, "aggregated_long.csv")
    json_path = os.path.join(args.output_dir, "aggregated.json")
    md_path = os.path.join(args.output_dir, "aggregated.md")
    missing_path = os.path.join(args.output_dir, "missing.json")

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
        print("[OK] All models have the full 5 experiments (lmms-eval tasks).")


if __name__ == "__main__":
    main()
