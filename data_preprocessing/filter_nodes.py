#!/usr/bin/env python3
"""
Node visibility & bounding-box size filter for NuScenes.

Reads a node annotation JSON (list of per-image items with ``objects``),
projects each object's 3-D box into the camera frame, and drops objects
that fail either of two gates:

* **Size gate** - ``max(w_px, h_px) >= min_box_pixels``
* **Visibility gate** - nuScenes visibility level ``>= min_visibility``

The filtered list is written back as JSON (same schema, fewer objects).
No caption data is read or written.

This script consolidates the filtering logic that previously lived in
``get_object_image.py`` (B.1-B.3 projection + visibility check) and
``merge_caption.py`` (per-frame size/visibility gates driven by the devkit).

Typical usage::

    python data_preprocessing/filter_nodes.py \
        --nodes  "$SCENE_GRAPH_DIR/node_annotations.json" \
        --out    "$SCENE_GRAPH_DIR/filtered_nodes.json" \
        --dataroot "$NUSCENES_DATAROOT" \
        --version  v1.0-trainval \
        --min_box_pixels  40 \
        --min_visibility  0 \
        --drop_empty_images
"""

import argparse
import json
import os
from typing import Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

# ------------------------------------------------------------------
# Geometry helpers (ported from merge_caption.py)
# ------------------------------------------------------------------

def project_box_to_2d_size_px(
    box: Box,
    cam_intrinsic: np.ndarray,
    img_wh: tuple[int, int],
) -> tuple[float, float] | None:
    """Project a 3-D box (already in camera frame) and return *(w_px, h_px)*.

    Corners behind the camera (z ≤ 0) are discarded.  Returns ``None``
    when *all* corners are behind the camera.
    """
    corners_3d = box.corners()  # (3, 8)
    in_front = corners_3d[2, :] > 0.01
    if not np.any(in_front):
        return None

    pts = view_points(corners_3d[:, in_front], cam_intrinsic, normalize=True)
    u, v = pts[0, :], pts[1, :]

    W, H = img_wh
    u = np.clip(u, 0, max(1, W - 1))
    v = np.clip(v, 0, max(1, H - 1))

    return float(u.max() - u.min()), float(v.max() - v.min())


def build_sd_index(
    nusc: NuScenes,
    sd_token: str,
) -> dict[str, dict[str, Any]]:
    """Build a per-frame index mapping *instance_token* → filter metadata.

    Returns a dict whose values carry ``vis`` (int|None),
    ``w_px`` (float|None) and ``h_px`` (float|None).
    When two annotations share the same instance token in a single frame
    (rare), the one with the larger projected area is kept.
    """
    sd = nusc.get("sample_data", sd_token)
    W = int(sd.get("width", 0) or 0)
    H = int(sd.get("height", 0) or 0)

    _, boxes, cam_intrinsic = nusc.get_sample_data(sd_token)

    index: dict[str, dict[str, Any]] = {}
    for box in boxes:
        ann_token = box.token
        sa = nusc.get("sample_annotation", ann_token)
        inst = sa["instance_token"]

        vis_token = int(sa.get("visibility_token"))

        size_px = project_box_to_2d_size_px(
            box, np.asarray(cam_intrinsic), (W, H)
        )
        w_px, h_px = size_px if size_px is not None else (None, None)

        prev = index.get(inst)
        if prev is not None:
            prev_area = (prev["w_px"] or 0) * (prev["h_px"] or 0)
            area = (w_px or 0) * (h_px or 0)
            if area <= prev_area:
                continue
        index[inst] = {"vis": vis_token, "w_px": w_px, "h_px": h_px}

    return index


# ------------------------------------------------------------------
# Node identity helper (ported from merge_caption.py)
# ------------------------------------------------------------------

def get_instance_from_node_obj(obj: dict[str, Any]) -> str | None:
    """Extract the instance token from a node object."""
    return (
        obj.get("id")
        or obj.get("node_id")
        or (obj.get("attributes") or {}).get("instance_token")
    )


# ------------------------------------------------------------------
# Main filter routine
# ------------------------------------------------------------------

def filter_nodes(
    nodes_path: str,
    out_path: str,
    dataroot: str,
    version: str = "v1.0-trainval",
    min_box_pixels: int = 40,
    min_visibility: int = 0,
    strict_size: bool = False,
    assume_invisible_if_missing: bool = False,
    drop_empty_images: bool = False,
) -> None:
    """Apply visibility / bounding-box size filtering to *nodes_path*.

    The interface mirrors ``merge_caption.py``'s filtering parameters
    so that existing scripts can switch over with minimal changes.
    """
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    with open(nodes_path, encoding="utf-8") as f:
        nodes: list[dict[str, Any]] = json.load(f)

    sd_cache: dict[str, dict[str, Any]] = {}

    total_objs = 0
    kept_objs = 0
    deleted_small = 0
    deleted_lowvis = 0
    deleted_size_missing = 0
    deleted_vis_missing = 0
    deleted_no_instance = 0

    def pass_size(w_px: float | None, h_px: float | None) -> bool:
        if w_px is None or h_px is None:
            return not strict_size
        return max(float(w_px), float(h_px)) >= float(min_box_pixels)

    def pass_visibility(vis: int | None) -> bool:
        if vis is None:
            return not assume_invisible_if_missing
        return int(vis) >= int(min_visibility)

    out_nodes: list[dict[str, Any]] = []

    for item in nodes:
        sd_token = item.get("sample_data_token")
        if sd_token:
            if sd_token not in sd_cache:
                try:
                    sd_cache[sd_token] = build_sd_index(nusc, sd_token)
                except Exception as exc:
                    print(f"[warn] failed to build index for sd={sd_token}: {exc}")
                    sd_cache[sd_token] = {}
            sd_index = sd_cache[sd_token]
        else:
            sd_index = {}

        objs = item.get("objects", []) or []
        new_objs: list[dict[str, Any]] = []

        for obj in objs:
            total_objs += 1
            inst = get_instance_from_node_obj(obj)
            if not inst:
                deleted_no_instance += 1
                continue

            info = sd_index.get(inst)
            w_px = info.get("w_px") if info else None
            h_px = info.get("h_px") if info else None
            vis = info.get("vis") if info else None

            # -- size gate --
            if not pass_size(w_px, h_px):
                if w_px is None or h_px is None:
                    deleted_size_missing += 1
                else:
                    deleted_small += 1
                continue

            # -- visibility gate --
            if not pass_visibility(vis):
                if vis is None:
                    deleted_vis_missing += 1
                else:
                    deleted_lowvis += 1
                continue

            new_objs.append(obj)

        kept_objs += len(new_objs)
        item["objects"] = new_objs

        if drop_empty_images:
            if new_objs:
                out_nodes.append(item)
        else:
            out_nodes.append(item)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_nodes, f, ensure_ascii=False, indent=2)

    kept_imgs = sum(1 for it in out_nodes if it.get("objects"))

    print("-" * 40)
    print(f"[done] total input objects processed: {total_objs}")
    print(f"[done] deleted (no instance token):   {deleted_no_instance}")
    print(f"[done] deleted (too small < {min_box_pixels}px): {deleted_small}")
    print(f"[done] deleted (low visibility < {min_visibility}): {deleted_lowvis}")
    if strict_size:
        print(f"[done] deleted (missing size info): {deleted_size_missing}")
    if assume_invisible_if_missing:
        print(f"[done] deleted (missing vis info): {deleted_vis_missing}")
    print("-" * 40)
    print(f"[done] final objects kept: {kept_objs}")
    print(f"[done] final images kept:  {kept_imgs} / {len(out_nodes)}")
    print("-" * 40)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    """CLI entry point for filtering NuScenes nodes by visibility and box size."""
    ap = argparse.ArgumentParser(
        description="Filter NuScenes nodes by visibility and projected box size.",
    )
    ap.add_argument("--nodes", required=True,
                    help="Input node annotation JSON (list).")
    ap.add_argument("--out", required=True,
                    help="Output filtered JSON path.")
    ap.add_argument("--drop_empty_images", action="store_true",
                    help="Drop images whose object list becomes empty after filtering.")

    # NuScenes
    ap.add_argument("--dataroot", required=True,
                    help="NuScenes dataroot directory.")
    ap.add_argument("--version", default="v1.0-trainval",
                    help="NuScenes version (e.g., v1.0-trainval, v1.0-mini).")

    # Filter gates
    ap.add_argument("--min_box_pixels", type=int, default=40,
                    help="Drop object if max(w_px, h_px) < this (default: 40).")
    ap.add_argument("--min_visibility", type=int, default=0,
                    help="Drop object if visibility level < this (nuScenes 1..4, default: 0).")
    ap.add_argument("--strict_size", action="store_true",
                    help="If set, drop objects whose 2-D size cannot be determined.")
    ap.add_argument("--assume_invisible_if_missing", action="store_true",
                    help="If set, drop objects whose visibility info is missing.")

    args = ap.parse_args()

    filter_nodes(
        nodes_path=args.nodes,
        out_path=args.out,
        dataroot=args.dataroot,
        version=args.version,
        min_box_pixels=args.min_box_pixels,
        min_visibility=args.min_visibility,
        strict_size=args.strict_size,
        assume_invisible_if_missing=args.assume_invisible_if_missing,
        drop_empty_images=args.drop_empty_images,
    )


if __name__ == "__main__":
    main()
