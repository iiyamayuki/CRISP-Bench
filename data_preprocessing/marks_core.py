"""Shared helpers for exporting and rendering benchmark image marks."""

from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_PATH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

NuScenes = None
_VIEW_POINTS = None
NUSCENES_AVAILABLE = False
try:
    from nuscenes.nuscenes import NuScenes as _NuScenes
    from nuscenes.utils.geometry_utils import view_points as _view_points

    NuScenes = _NuScenes
    _VIEW_POINTS = _view_points
    NUSCENES_AVAILABLE = True
except ImportError:
    pass


def ensure_dir(path: str) -> None:
    """Create *path* when it does not already exist."""
    os.makedirs(path, exist_ok=True)


def clip_box_to_image(box: list[int] | tuple[int, int, int, int], width: int, height: int) -> list[int] | None:
    """Clip an xyxy box to image bounds and return None when it collapses."""
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def build_output_image_path(
    dataset: str,
    output_img_dir: str,
    sample_data_token: str,
    scene_id: str | None = None,
) -> str:
    """Return the current marked-image filename used by the benchmark pipeline."""
    if dataset == "nuscenes":
        save_name = f"{sample_data_token}.jpg"
    elif dataset == "scannetpp":
        if not scene_id:
            raise ValueError("scene_id is required for ScanNet++ marked image names")
        save_name = f"{scene_id}_{sample_data_token}.jpg"
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return os.path.join(output_img_dir, save_name)


def compute_render_metadata(box: list[int] | tuple[int, int, int, int], label_id: int) -> dict[str, Any]:
    """Compute the exact drawing parameters used for one mark."""
    x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    width = x2 - x1
    height = y2 - y1
    center_x = int(x1 + width / 2)
    center_y = int(y1 + height / 2)

    radius = int(min(width, height) * 0.15)
    radius = max(15, min(radius, 40))

    circle_center = [center_x, center_y]
    alpha = 0.6
    if width < 30 or height < 30:
        radius = 5
        circle_center = [center_x, int(y1)]
        alpha = 0.8

    font_scale = max(0.4, radius / 20.0)
    thickness = max(1, int(font_scale * 2))
    text = str(label_id)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_anchor = [center_x - text_width // 2, center_y + text_height // 2]

    return {
        "circle_center": circle_center,
        "radius": radius,
        "alpha": alpha,
        "font_scale": font_scale,
        "thickness": thickness,
        "text_anchor": text_anchor,
    }


def build_mark(
    box: list[int] | tuple[int, int, int, int],
    label_id: int,
    node_id: str,
) -> dict[str, Any]:
    """Build a manifest record for one rendered mark."""
    box_xyxy = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    local_id = int(label_id)
    return {
        "node_id": str(node_id),
        "local_id": local_id,
        "bbox_xyxy": box_xyxy,
        "render": compute_render_metadata(box_xyxy, local_id),
    }


def _entry_lookup_key(entry: dict[str, Any]) -> str | None:
    sample_data_token = entry.get("sample_data_token")
    if sample_data_token:
        return str(sample_data_token)

    image_path = entry.get("source_image") or entry.get("image")
    if image_path:
        return os.path.basename(str(image_path))
    return None


def load_local_id_lookup(reference_path: str) -> dict[str, dict[str, int]]:
    """Load a sample-level node_id -> local_id lookup from a JSON or JSONL file."""
    if reference_path.endswith(".jsonl"):
        with open(reference_path, encoding="utf-8") as handle:
            entries = [json.loads(line) for line in handle if line.strip()]
    else:
        with open(reference_path, encoding="utf-8") as handle:
            entries = json.load(handle)

    if not isinstance(entries, list):
        raise TypeError(f"Expected a top-level list in {reference_path}")

    lookup: dict[str, dict[str, int]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        sample_key = _entry_lookup_key(entry)
        if not sample_key:
            continue

        objects = entry.get("objects")
        if objects is None:
            objects = entry.get("marks", [])

        node_id_to_local_id: dict[str, int] = {}
        for obj in objects:
            node_id = obj.get("node_id")
            local_id = obj.get("local_id")
            if node_id is None or local_id is None:
                continue
            node_id_to_local_id[str(node_id)] = int(local_id)

        if node_id_to_local_id:
            lookup[sample_key] = node_id_to_local_id

    return lookup


def resolve_entry_local_ids(
    entry: dict[str, Any],
    local_id_lookup: dict[str, dict[str, int]] | None,
) -> dict[str, int]:
    """Return the node_id -> local_id mapping for one manifest entry."""
    if not local_id_lookup:
        return {}

    sample_key = _entry_lookup_key(entry)
    if not sample_key:
        return {}

    return local_id_lookup.get(sample_key, {})


def remap_marks_local_ids(
    marks: list[dict[str, Any]],
    node_id_to_local_id: dict[str, int],
) -> list[dict[str, Any]]:
    """Rewrite mark local_ids from *node_id_to_local_id* while preserving bbox placement."""
    if not node_id_to_local_id:
        return marks

    remapped: list[dict[str, Any]] = []
    for mark in marks:
        node_id = str(mark.get("node_id"))
        local_id = node_id_to_local_id.get(node_id)
        if local_id is None:
            remapped.append(mark)
            continue

        current_local_id = mark.get("local_id")
        if current_local_id is not None and int(current_local_id) == int(local_id):
            remapped.append(mark)
            continue

        remapped.append(build_mark(mark["bbox_xyxy"], int(local_id), node_id))

    return remapped


def _coerce_bgr_image(img: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(img, Image.Image):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def render_mark(
    img: Image.Image | np.ndarray,
    mark: dict[str, Any],
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Render one mark record onto *img* using stored drawing parameters."""
    image = _coerce_bgr_image(img)

    render = mark.get("render")
    if render is None:
        render = compute_render_metadata(mark["bbox_xyxy"], int(mark["local_id"]))

    overlay = image.copy()
    circle_center = (int(render["circle_center"][0]), int(render["circle_center"][1]))
    radius = int(render["radius"])
    alpha = float(render["alpha"])
    cv2.circle(overlay, circle_center, radius, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    text = str(mark["local_id"])
    text_anchor = (int(render["text_anchor"][0]), int(render["text_anchor"][1]))
    font_scale = float(render["font_scale"])
    thickness = int(render["thickness"])
    cv2.putText(
        image,
        text,
        text_anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
    )
    cv2.putText(
        image,
        text,
        text_anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    return image


def draw_adaptive_centroid(
    img: Image.Image | np.ndarray,
    box: list[int] | tuple[int, int, int, int],
    label_id: int,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Backward-compatible wrapper used by the legacy mark_objects.py script."""
    return render_mark(img, build_mark(box, label_id, str(label_id)), color=color)


def load_scene_jsonl_lookup(jsonl_path: str) -> dict[str, dict[str, list[int]]] | None:
    """Build a per-image bbox lookup from one ScanNet++ source JSONL file."""
    lookup: dict[str, dict[str, list[int]]] = {}
    if not os.path.exists(jsonl_path):
        return None

    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            image_path = item.get("image")
            if not image_path:
                continue

            image_name = os.path.basename(image_path)
            lookup.setdefault(image_name, {})
            for obj in item.get("objects", []):
                node_id = obj.get("node_id")
                bbox = obj.get("attributes", {}).get("bbox_2d")
                if node_id and bbox:
                    lookup[image_name][str(node_id)] = [int(v) for v in bbox]
    return lookup


def find_jsonl_for_image(image_path: str, jsonl_dir: str) -> tuple[str | None, str | None]:
    """Find the ScanNet++ source JSONL file that owns *image_path*."""
    parts = image_path.replace("\\", "/").split("/")
    for part in parts:
        candidate = os.path.join(jsonl_dir, f"{part}.jsonl")
        if os.path.exists(candidate):
            return candidate, part
    return None, None


def get_box_2d_corners_nusc(box: Any, intrinsic: Any, width: int, height: int) -> list[int]:
    """Project a NuScenes 3D box into clipped 2D pixel coordinates."""
    if _VIEW_POINTS is None:
        raise ImportError("nuscenes-devkit is not installed")

    corners_3d = box.corners()
    corners_2d = _VIEW_POINTS(corners_3d, np.asarray(intrinsic), normalize=True)

    x_min = corners_2d[0, :].min()
    x_max = corners_2d[0, :].max()
    y_min = corners_2d[1, :].min()
    y_max = corners_2d[1, :].max()

    return [
        int(max(0, x_min)),
        int(max(0, y_min)),
        int(min(width - 1, x_max)),
        int(min(height - 1, y_max)),
    ]