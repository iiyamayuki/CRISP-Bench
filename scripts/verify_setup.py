#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

REQUIRED_ENV_KEYS = [
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "HF_HOME",
    "LMMS_EVAL_HOME",
    "VLLM_CACHE_ROOT",
    "NUSCENES_DATAROOT",
    "SCANNETPP_DATAROOT",
]


class Reporter:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.passes: list[str] = []

    def ok(self, message: str) -> None:
        self.passes.append(message)
        print(f"[OK] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.errors.append(message)
        print(f"[FAIL] {message}")

    def exit_code(self) -> int:
        return 1 if self.errors else 0


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def first_non_empty_jsonl(path: Path) -> dict[str, object] | None:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                return json.loads(line)
    return None


def first_record_from_json_or_jsonl(path: Path) -> dict[str, object] | None:
    if path.suffix.lower() == ".jsonl":
        return first_non_empty_jsonl(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            return None
        first = data[0]
        return first if isinstance(first, dict) else None
    if isinstance(data, dict):
        return data
    return None


def validate_env_file(path: Path, reporter: Reporter) -> dict[str, str]:
    if not path.exists():
        reporter.fail(f"Environment file not found: {path}")
        return {}
    env_values = read_env_file(path)
    missing = [key for key in REQUIRED_ENV_KEYS if key not in env_values]
    if missing:
        reporter.fail(f"Environment file is missing keys: {', '.join(missing)}")
    else:
        reporter.ok(f"Environment file contains all expected keys: {path}")
    empty_keys = [key for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY") if key in env_values and not env_values[key]]
    if empty_keys:
        reporter.warn(
            "API key placeholders are still empty: " + ", ".join(empty_keys)
        )
    return env_values


def validate_dir_markers(path: Path, markers: Iterable[str], label: str, reporter: Reporter) -> None:
    if not path.exists():
        reporter.fail(f"{label} root does not exist: {path}")
        return
    if not path.is_dir():
        reporter.fail(f"{label} root is not a directory: {path}")
        return
    present = [marker for marker in markers if (path / marker).exists()]
    if present:
        reporter.ok(f"{label} root looks plausible: {path}")
    else:
        reporter.warn(
            f"{label} root exists but does not contain the usual markers ({', '.join(markers)}): {path}"
        )


def validate_templates(path: Path, reporter: Reporter) -> None:
    if not path.exists():
        reporter.fail(f"QA template file not found: {path}")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        reporter.fail(f"QA template file is not valid JSON: {path} ({exc})")
        return
    if not isinstance(data, list) or not data:
        reporter.fail(f"QA template file must be a non-empty JSON list: {path}")
        return
    first = data[0]
    if not isinstance(first, dict):
        reporter.fail(f"First QA template entry is not an object: {path}")
        return
    missing = [key for key in ("question", "type", "category", "answer") if key not in first]
    if missing:
        reporter.fail(f"First QA template entry is missing keys: {', '.join(missing)}")
    else:
        reporter.ok(f"QA template file has the expected top-level schema: {path}")


def validate_scene_graph(path: Path, reporter: Reporter) -> None:
    if not path.exists():
        reporter.fail(f"Scene graph JSON not found: {path}")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        reporter.fail(f"Scene graph JSON is not valid JSON: {path} ({exc})")
        return
    if not isinstance(data, list) or not data:
        reporter.fail(f"Scene graph JSON must be a non-empty list: {path}")
        return
    first = data[0]
    if not isinstance(first, dict):
        reporter.fail(f"First scene graph entry is not an object: {path}")
        return
    missing = [key for key in ("image", "sample_data_token", "objects") if key not in first]
    if missing:
        reporter.fail(f"Scene graph entry is missing keys: {', '.join(missing)}")
        return
    objects = first.get("objects")
    if not isinstance(objects, list) or not objects:
        reporter.fail(f"Scene graph entry must contain a non-empty objects list: {path}")
        return
    first_object = objects[0]
    if not isinstance(first_object, dict):
        reporter.fail(f"First scene graph object is not an object: {path}")
        return
    if "node_id" not in first_object or "attributes" not in first_object:
        reporter.fail(f"First scene graph object is missing node_id or attributes: {path}")
        return
    attributes = first_object.get("attributes")
    if not isinstance(attributes, dict) or "category_name" not in attributes:
        reporter.fail(f"First scene graph object is missing attributes.category_name: {path}")
        return
    reporter.ok(f"Scene graph JSON has the expected minimal schema: {path}")


def validate_qa_jsonl(path: Path, reporter: Reporter) -> None:
    if not path.exists():
        reporter.fail(f"QA JSONL not found: {path}")
        return
    try:
        item = first_non_empty_jsonl(path)
    except json.JSONDecodeError as exc:
        reporter.fail(f"QA JSONL contains invalid JSON: {path} ({exc})")
        return
    if item is None:
        reporter.fail(f"QA JSONL does not contain any records: {path}")
        return
    missing = [key for key in ("id", "image", "conversations") if key not in item]
    if missing:
        reporter.fail(f"QA JSONL record is missing keys: {', '.join(missing)}")
        return
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        reporter.fail(f"QA JSONL record must contain at least two conversation turns: {path}")
        return
    reporter.ok(f"QA JSONL has the expected minimal schema: {path}")


def validate_sgc_jsonl(path: Path, reporter: Reporter) -> None:
    if not path.exists():
        reporter.fail(f"SGC JSONL not found: {path}")
        return
    try:
        item = first_non_empty_jsonl(path)
    except json.JSONDecodeError as exc:
        reporter.fail(f"SGC JSONL contains invalid JSON: {path} ({exc})")
        return
    if item is None:
        reporter.fail(f"SGC JSONL does not contain any records: {path}")
        return
    missing = [key for key in ("image", "conversations") if key not in item]
    if missing:
        reporter.fail(f"SGC JSONL record is missing keys: {', '.join(missing)}")
        return
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        reporter.fail(f"SGC JSONL record must contain at least two conversation turns: {path}")
        return
    reporter.ok(f"SGC JSONL has the expected minimal schema: {path}")


def validate_marks_jsonl(path: Path, label: str, reporter: Reporter) -> None:
    if not path.exists():
        reporter.fail(f"{label} not found: {path}")
        return
    try:
        item = first_non_empty_jsonl(path)
    except json.JSONDecodeError as exc:
        reporter.fail(f"{label} contains invalid JSON: {path} ({exc})")
        return
    if item is None:
        reporter.fail(f"{label} does not contain any records: {path}")
        return
    missing = [key for key in ("dataset", "sample_data_token", "source_image", "image_size", "marks") if key not in item]
    if missing:
        reporter.fail(f"{label} record is missing keys: {', '.join(missing)}")
        return
    image_size = item.get("image_size")
    if not isinstance(image_size, dict) or "width" not in image_size or "height" not in image_size:
        reporter.fail(f"{label} record must include image_size.width and image_size.height: {path}")
        return
    marks = item.get("marks")
    if not isinstance(marks, list) or not marks:
        reporter.fail(f"{label} record must contain a non-empty marks list: {path}")
        return
    first_mark = marks[0]
    if not isinstance(first_mark, dict):
        reporter.fail(f"First {label} mark is not an object: {path}")
        return
    missing_mark = [key for key in ("node_id", "local_id", "bbox_xyxy", "render") if key not in first_mark]
    if missing_mark:
        reporter.fail(f"First {label} mark is missing keys: {', '.join(missing_mark)}")
        return
    render = first_mark.get("render")
    if not isinstance(render, dict):
        reporter.fail(f"First {label} mark render block is not an object: {path}")
        return
    missing_render = [
        key
        for key in ("circle_center", "radius", "alpha", "font_scale", "thickness", "text_anchor")
        if key not in render
    ]
    if missing_render:
        reporter.fail(f"First {label} mark render block is missing keys: {', '.join(missing_render)}")
        return
    reporter.ok(f"{label} has the expected minimal schema: {path}")


def validate_qa_sg_master_scene_graph(path: Path, reporter: Reporter) -> None:
    if not path.exists():
        reporter.fail(f"QA-SG master scene graph source not found: {path}")
        return
    try:
        item = first_record_from_json_or_jsonl(path)
    except json.JSONDecodeError as exc:
        reporter.fail(f"QA-SG master scene graph source is not valid JSON/JSONL: {path} ({exc})")
        return
    if item is None:
        reporter.fail(f"QA-SG master scene graph source does not contain any records: {path}")
        return
    missing = [key for key in ("image", "objects", "edges") if key not in item]
    if missing:
        reporter.fail(f"QA-SG master scene graph record is missing keys: {', '.join(missing)}")
        return
    objects = item.get("objects")
    if not isinstance(objects, list):
        reporter.fail(f"QA-SG master scene graph record must contain an objects list: {path}")
        return
    edges = item.get("edges")
    if not isinstance(edges, list):
        reporter.fail(f"QA-SG master scene graph record must contain an edges list: {path}")
        return
    if objects:
        first_object = objects[0]
        if not isinstance(first_object, dict):
            reporter.fail(f"First QA-SG master scene graph object is not an object: {path}")
            return
        missing_object = [key for key in ("node_id", "local_id", "attributes") if key not in first_object]
        if missing_object:
            reporter.fail(f"First QA-SG master scene graph object is missing keys: {', '.join(missing_object)}")
            return
        attributes = first_object.get("attributes")
        if not isinstance(attributes, dict):
            reporter.fail(f"First QA-SG master scene graph object attributes block is not an object: {path}")
            return
        if "category_name" not in attributes:
            reporter.fail(f"First QA-SG master scene graph object is missing attributes.category_name: {path}")
            return
        translation = attributes.get("translation")
        if not isinstance(translation, dict) or "z_cam" not in translation:
            reporter.fail(f"First QA-SG master scene graph object is missing attributes.translation.z_cam: {path}")
            return
        size = attributes.get("size")
        if not isinstance(size, dict) or any(key not in size for key in ("w", "l", "h")):
            reporter.fail(f"First QA-SG master scene graph object is missing attributes.size.w, attributes.size.l, or attributes.size.h: {path}")
            return
    reporter.ok(f"QA-SG master scene graph source has the expected minimal schema: {path}")


def validate_results_json(path: Path, reporter: Reporter) -> None:
    if not path.exists():
        reporter.fail(f"Aggregated results JSON not found: {path}")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        reporter.fail(f"Aggregated results JSON is not valid JSON: {path} ({exc})")
        return
    if not isinstance(data, dict) or not data:
        reporter.fail(f"Aggregated results JSON must be a non-empty object: {path}")
        return
    reporter.ok(f"Aggregated results JSON is readable: {path}")


def resolve_optional_path(cli_value: str | None, env_values: dict[str, str], env_key: str) -> Path | None:
    if cli_value:
        return Path(cli_value)
    env_value = env_values.get(env_key) or os.environ.get(env_key)
    if env_value:
        return Path(env_value)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate current setup assumptions for CRISP Bench.")
    parser.add_argument("--env-file", default=".env.example", help="Path to .env or .env.example")
    parser.add_argument("--templates", default="qa/question_template.json", help="Path to QA template JSON")
    parser.add_argument("--nuscenes-root", default=None, help="Optional NuScenes root override")
    parser.add_argument("--scannetpp-root", default=None, help="Optional ScanNet++ root override")
    parser.add_argument(
        "--check-prepared-benchmark",
        action="store_true",
        help="Validate the standard prepared benchmark files used by the default evaluation path",
    )
    parser.add_argument(
        "--prepared-qa-file",
        default="data/processed/combined/QA_pairs/qa_data.jsonl",
        help="Path to the prepared benchmark QA JSONL",
    )
    parser.add_argument(
        "--prepared-sgc-file",
        default="data/processed/combined/SGC_task/sgc_task.jsonl",
        help="Path to the prepared benchmark SGC JSONL",
    )
    parser.add_argument(
        "--nuscenes-marks",
        default="data/processed/nuscenes/annotated_image/marks.jsonl",
        help="Path to the NuScenes marks manifest",
    )
    parser.add_argument(
        "--scannetpp-marks",
        default="data/processed/scannetpp/annotated_image/marks.jsonl",
        help="Path to the ScanNet++ marks manifest",
    )
    parser.add_argument(
        "--check-qa-sg-master",
        "--check-qa-sg-gt",
        dest="check_qa_sg_master",
        action="store_true",
        help="Validate the master scene graph source file used to derive crisp_qa_sg GT input",
    )
    parser.add_argument(
        "--qa-sg-master-file",
        "--qa-sg-gt-file",
        dest="qa_sg_master_file",
        default="data/processed/combined/scene_graph/combined_scene_graph.jsonl",
        help="Path to the master scene graph JSON/JSONL file used to derive crisp_qa_sg GT input",
    )
    parser.add_argument("--scene-graph", default=None, help="Optional scene graph JSON to validate")
    parser.add_argument("--qa-file", default=None, help="Optional QA JSONL to validate")
    parser.add_argument("--results-json", default=None, help="Optional aggregated results JSON to validate")
    parser.add_argument("--skip-data-roots", action="store_true", help="Skip validation of NuScenes and ScanNet++ roots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reporter = Reporter()

    env_values = validate_env_file(Path(args.env_file), reporter)
    validate_templates(Path(args.templates), reporter)

    if args.skip_data_roots:
        reporter.warn("Skipped raw dataset root validation by request")
    else:
        nuscenes_root = resolve_optional_path(args.nuscenes_root, env_values, "NUSCENES_DATAROOT")
        if nuscenes_root is not None:
            validate_dir_markers(
                nuscenes_root,
                ["samples", "sweeps", "v1.0-mini", "v1.0-trainval", "maps"],
                "NuScenes",
                reporter,
            )
        else:
            reporter.warn("NuScenes root was not provided and could not be inferred from the environment file")

        scannetpp_root = resolve_optional_path(args.scannetpp_root, env_values, "SCANNETPP_DATAROOT")
        if scannetpp_root is not None:
            validate_dir_markers(
                scannetpp_root,
                ["dslr", "scans", "scene_graph"],
                "ScanNet++",
                reporter,
            )
        else:
            reporter.warn("ScanNet++ root was not provided and could not be inferred from the environment file")

    if args.check_prepared_benchmark:
        validate_qa_jsonl(Path(args.prepared_qa_file), reporter)
        validate_sgc_jsonl(Path(args.prepared_sgc_file), reporter)
        validate_marks_jsonl(Path(args.nuscenes_marks), "NuScenes marks manifest", reporter)
        validate_marks_jsonl(Path(args.scannetpp_marks), "ScanNet++ marks manifest", reporter)

    if args.check_qa_sg_master:
        validate_qa_sg_master_scene_graph(Path(args.qa_sg_master_file), reporter)

    if args.scene_graph:
        validate_scene_graph(Path(args.scene_graph), reporter)
    if args.qa_file:
        validate_qa_jsonl(Path(args.qa_file), reporter)
    if args.results_json:
        validate_results_json(Path(args.results_json), reporter)

    print("\nSummary:")
    print(f"  passed: {len(reporter.passes)}")
    print(f"  warnings: {len(reporter.warnings)}")
    print(f"  errors: {len(reporter.errors)}")
    return reporter.exit_code()


if __name__ == "__main__":
    sys.exit(main())
