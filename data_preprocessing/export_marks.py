"""Export dataset-specific drawing metadata into a reusable marks.jsonl manifest."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from tqdm import tqdm

from data_preprocessing.marks_core import (
    NUSCENES_AVAILABLE,
    NuScenes,
    build_mark,
    clip_box_to_image,
    find_jsonl_for_image,
    get_box_2d_corners_nusc,
    load_local_id_lookup,
    load_scene_jsonl_lookup,
    resolve_entry_local_ids,
)
from utils.io import write_jsonl


def _load_input_data(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected top-level JSON list in {path}")
    return data


def _resolve_local_id_source(local_id_source: str | None, output_path: str, input_path: str | None = None) -> str | None:
    if local_id_source:
        resolved = os.path.abspath(local_id_source)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"local_id source not found: {resolved}")
        return resolved

    candidates = [os.path.join(os.path.dirname(os.path.abspath(output_path)), "nodes_with_2dbox.json")]
    if input_path:
        input_dir = os.path.dirname(os.path.abspath(input_path))
        candidates.append(os.path.join(os.path.dirname(input_dir), "annotated_image", "nodes_with_2dbox.json"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _all_objects_have_local_id(data: list[dict[str, Any]]) -> bool:
    for entry in data:
        for obj in entry.get("objects", []):
            if obj.get("node_id") and obj.get("local_id") is None:
                return False
    return True


def _load_scannetpp_image_size(dataroot: str, image_path: str) -> tuple[int, int]:
    """Load ScanNet++ image size from transforms_undistorted.json without opening the image."""
    parts = image_path.replace("\\", "/").split("/")
    try:
        dslr_index = parts.index("dslr")
    except ValueError as exc:
        raise FileNotFoundError(f"Cannot infer ScanNet++ scene root from image path: {image_path}") from exc

    scene_root = os.path.join(dataroot, *parts[:dslr_index])
    transforms_path = os.path.join(scene_root, "dslr", "nerfstudio", "transforms_undistorted.json")
    with open(transforms_path, encoding="utf-8") as handle:
        transforms = json.load(handle)
    return int(transforms["h"]), int(transforms["w"])


def export_scannetpp_marks(
    data: list[dict[str, Any]],
    original_jsonl_dir: str,
    dataroot: str,
) -> list[dict[str, Any]]:
    """Build a marks manifest for ScanNet++ from the original per-scene JSONL files."""
    print(f"[Info] Exporting ScanNet++ marks for {len(data)} frames...")

    cache = {"id": None, "data": {}}
    scene_size_cache: dict[str, tuple[int, int]] = {}
    stats = {"success": 0, "missing_file": 0, "missing_metadata": 0, "missing_bbox": 0}
    manifest_rows: list[dict[str, Any]] = []

    for entry in tqdm(data, desc="Exporting ScanNet++ marks"):
        image_path = entry.get("image")
        sample_data_token = entry.get("sample_data_token")
        if not image_path or not sample_data_token:
            continue

        jsonl_path, scene_id = find_jsonl_for_image(image_path, original_jsonl_dir)
        if not jsonl_path or not scene_id:
            stats["missing_file"] += 1
            continue

        if cache["id"] != scene_id:
            cache["data"] = load_scene_jsonl_lookup(jsonl_path) or {}
            cache["id"] = scene_id

        if scene_id not in scene_size_cache:
            try:
                scene_size_cache[scene_id] = _load_scannetpp_image_size(dataroot, image_path)
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
                print(f"[Error] Failed to load ScanNet++ metadata for {scene_id}: {exc}")
                stats["missing_metadata"] += 1
                continue
        image_height, image_width = scene_size_cache[scene_id]

        image_filename = image_path.replace("\\", "/").split("/")[-1]
        frame_lookup = cache["data"].get(image_filename, {})

        marks: list[dict[str, Any]] = []
        for index, obj in enumerate(entry.get("objects", []), start=1):
            node_id = str(obj.get("node_id", ""))
            bbox = frame_lookup.get(node_id)
            if not bbox:
                continue

            clipped_box = clip_box_to_image(bbox, image_width, image_height)
            if clipped_box is None:
                continue

            local_id = int(obj.get("local_id", index))
            marks.append(build_mark(clipped_box, local_id, node_id))

        if not marks:
            stats["missing_bbox"] += 1
            continue

        manifest_rows.append(
            {
                "dataset": "scannetpp",
                "scene_id": scene_id,
                "sample_data_token": sample_data_token,
                "camera_channel": entry.get("camera_channel"),
                "source_image": image_path,
                "image_size": {"width": image_width, "height": image_height},
                "marks": marks,
            }
        )
        stats["success"] += 1

    print(f"[Result] ScanNet++ marks export complete. Stats: {stats}")
    return manifest_rows


def export_nuscenes_marks(
    data: list[dict[str, Any]],
    nusc: Any,
    *,
    local_id_lookup: dict[str, dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Build a marks manifest for NuScenes from the SDK projection path."""
    print(f"[Info] Exporting NuScenes marks for {len(data)} frames...")

    stats = {"success": 0, "missing_metadata": 0, "missing_bbox": 0, "sdk_errors": 0}
    manifest_rows: list[dict[str, Any]] = []

    for entry in tqdm(data, desc="Exporting NuScenes marks"):
        sample_data_token = entry.get("sample_data_token")
        if not sample_data_token:
            continue

        try:
            sample_data = nusc.get("sample_data", sample_data_token)
            _, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token)
        except Exception as exc:
            print(f"[Error] Failed to load sample data {sample_data_token}: {exc}")
            stats["sdk_errors"] += 1
            continue

        try:
            image_width = int(sample_data["width"])
            image_height = int(sample_data["height"])
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[Error] Failed to read NuScenes metadata for {sample_data_token}: {exc}")
            stats["missing_metadata"] += 1
            continue

        source_image = entry.get("image") or sample_data.get("filename")
        if not source_image:
            stats["missing_metadata"] += 1
            continue
        user_objects = {str(obj["node_id"]): obj for obj in entry.get("objects", []) if obj.get("node_id")}
        entry_local_ids = resolve_entry_local_ids(entry, local_id_lookup)

        marks: list[dict[str, Any]] = []
        for box in boxes:
            try:
                annotation = nusc.get("sample_annotation", box.token)
                instance_token = annotation["instance_token"]
            except KeyError:
                continue

            if instance_token not in user_objects:
                continue

            bbox = get_box_2d_corners_nusc(box, camera_intrinsic, image_width, image_height)
            clipped_box = clip_box_to_image(bbox, image_width, image_height)
            if clipped_box is None:
                continue

            obj = user_objects[instance_token]
            local_id = obj.get("local_id")
            if local_id is None:
                local_id = entry_local_ids.get(instance_token)
            if local_id is None:
                local_id = len(marks) + 1
            marks.append(build_mark(clipped_box, local_id, instance_token))

        if not marks:
            stats["missing_bbox"] += 1
            continue

        manifest_rows.append(
            {
                "dataset": "nuscenes",
                "sample_data_token": sample_data_token,
                "camera_channel": entry.get("camera_channel"),
                "source_image": source_image,
                "image_size": {"width": image_width, "height": image_height},
                "marks": marks,
            }
        )
        stats["success"] += 1

    print(f"[Result] NuScenes marks export complete. Stats: {stats}")
    return manifest_rows


def main() -> None:
    """CLI entry point for exporting marks.jsonl."""
    parser = argparse.ArgumentParser(description="Export reusable mark metadata to marks.jsonl.")
    parser.add_argument("--dataset", choices=["nuscenes", "scannetpp"], required=True, help="Dataset mode.")
    parser.add_argument("--input", required=True, help="Path to the filtered scene/object JSON used for marking.")
    parser.add_argument("--output", required=True, help="Destination path for marks.jsonl.")
    parser.add_argument(
        "--dataroot",
        help="Dataset root used only for ScanNet++ metadata lookup (no images are read by this script).",
    )
    parser.add_argument(
        "--original_jsonl_dir",
        help="Directory containing original ScanNet++ per-scene JSONL files (required for scannetpp).",
    )
    parser.add_argument(
        "--local_id_source",
        help=(
            "Optional JSON/JSONL file that provides canonical node_id -> local_id mappings. "
            "For NuScenes, the script auto-detects a sibling nodes_with_2dbox.json next to --output."
        ),
    )
    parser.add_argument("--nusc_root", help="NuScenes dataroot (required for nuscenes).")
    parser.add_argument("--nusc_version", default="v1.0-trainval", help="NuScenes version.")

    args = parser.parse_args()

    data = _load_input_data(args.input)

    if args.dataset == "scannetpp":
        if not args.original_jsonl_dir:
            raise SystemExit("[Error] Argument --original_jsonl_dir is required for ScanNet++ mode.")
        if not args.dataroot:
            raise SystemExit("[Error] Argument --dataroot is required for ScanNet++ mode.")
        manifest_rows = export_scannetpp_marks(
            data,
            args.original_jsonl_dir,
            args.dataroot,
        )
    else:
        if not NUSCENES_AVAILABLE:
            raise SystemExit("[Error] nuscenes-devkit is not installed. Cannot export NuScenes marks.")
        if not args.nusc_root:
            raise SystemExit("[Error] Argument --nusc_root is required for NuScenes mode.")
        local_id_lookup = None
        local_id_source = _resolve_local_id_source(args.local_id_source, args.output, args.input)
        if local_id_source is not None:
            local_id_lookup = load_local_id_lookup(local_id_source)
            print(f"[Info] Using NuScenes local_id reference: {local_id_source}")
        elif not _all_objects_have_local_id(data):
            raise SystemExit(
                "[Error] NuScenes export requires canonical local_id information. "
                "Provide --local_id_source or place nodes_with_2dbox.json next to --output."
            )
        assert NuScenes is not None
        nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_root, verbose=True)
        manifest_rows = export_nuscenes_marks(data, nusc, local_id_lookup=local_id_lookup)

    write_jsonl(manifest_rows, args.output)
    print(f"[Info] Saved {len(manifest_rows)} mark records to {args.output}")


if __name__ == "__main__":
    main()