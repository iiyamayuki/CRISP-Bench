#!/usr/bin/env python3
"""
Construct edges:
  1. Center Object -> Other Objects
  2. Ego (Camera) -> All Objects

Center selection:
  - centroid (mean of all (x,y,z) in camera frame),
  - choose the object closest to centroid,
  - if multiple within --centroid-tol (meters), choose the larger bbox (w*l*h).

Edge schema (under each image):
{
  "from": "<center_node_id>",
  "to":   "<other_node_id>",
  "distance": <float>,   # 3D Euclidean distance
  "angle": <float>,      # horizontal azimuth: atan2(dx, dz) in degrees (right positive)
  "relation": {
    "right": 0/1,
    "left": 0/1,
    "in front of": 0/1,
    "behind": 0/1,
    "up": 0/1,
    "down": 0/1
  }
}

Stats (computed on kept images only):
  - total images
  - average #objects per image
  - average #edges per image
  - easy/medium/hard counts and proportions (by object count)
  - relation distribution across all edges
"""

import argparse
import json
import math
from collections import Counter
from typing import Any

SPARSE_COUNTS = [2, 3]
COM_COUNTS  = [4, 5, 6, 7, 8]
DENSE_COUNTS = [9, 10, 11, 12]

# ---------------- helpers ---------------- #

def get_cam_xyz(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Returns (x, y, z_cam, z_world)."""
    tr = ((obj.get("attributes") or {}).get("translation")) or None
    if not tr:
        return None
    try:
        # z_world is needed for 'up/down' relation in absolute terms
        return float(tr["x"]), float(tr["y"]), float(tr["z_cam"]), float(tr.get("z_world", 0.0))
    except Exception:
        return None

def get_size_wlh(obj: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    """Extract (width, length, height) from an object's attributes."""
    sz = ((obj.get("attributes") or {}).get("size")) or None
    if not sz:
        return (None, None, None)
    try:
        return (
            float(sz["w"]) if "w" in sz else None,
            float(sz["l"]) if "l" in sz else None,
            float(sz["h"]) if "h" in sz else None
        )
    except Exception:
        return (None, None, None)

def bbox_volume(w: float | None, l: float | None, h: float | None) -> float:
    """Compute bounding box volume from width, length, and height."""
    if w is None or l is None or h is None:
        return 0.0
    return float(max(0.0, w) * max(0.0, l) * max(0.0, h))

def rel_flags(dx: float, dz_cam: float, cz_world: float, tz_world: float,
              csize: tuple[float | None, float | None, float | None],
              tsize: tuple[float | None, float | None, float | None], eps: float) -> dict[str, int]:
    """Compute binary directional relation flags between center and target objects."""
    # Camera frame: x right, y down, z forward
    right   = 1 if dx >  eps else 0
    left    = 1 if dx < -eps else 0
    infront = 1 if dz_cam < -eps else 0 # target is closer to camera (z smaller) than center
    behind  = 1 if dz_cam >  eps else 0 # target is further from camera (z larger) than center

    # Vertical relation (using World Z)
    # cz_min/max: Center object's vertical bounds in world coordinates
    cz_min = cz_world - (csize[2]/2 if csize[2] is not None else 0.0)
    cz_max = cz_world + (csize[2]/2 if csize[2] is not None else 0.0)

    # tz_min/max: Target object's vertical bounds in world coordinates
    tz_min = tz_world - (tsize[2]/2 if tsize[2] is not None else 0.0)
    tz_max = tz_world + (tsize[2]/2 if tsize[2] is not None else 0.0)

    # 'up': target bottom is higher than center top
    # 'down': target top is lower than center bottom
    up     = 1 if tz_min - cz_max > -eps else 0
    down   = 1 if tz_max - cz_min <  eps else 0

    return {
        "right": right,
        "left": left,
        "in front of": infront,
        "behind": behind,
        "up": up,
        "down": down
    }

def choose_center_node(objects: list[dict[str, Any]], centroid_tol: float) -> int | None:
    """Return index of center object (closest to centroid; tie -> larger volume)."""
    pts = []
    for idx, o in enumerate(objects or []):
        p = get_cam_xyz(o)
        if p is not None:
            pts.append((idx, (p[0], p[1], p[2])))
    if len(pts) == 0:
        return None
    if len(pts) == 1:
        return pts[0][0]

    sx = sum(p[0] for _, p in pts)
    sy = sum(p[1] for _, p in pts)
    sz = sum(p[2] for _, p in pts)
    n = float(len(pts))
    cx, cy, cz = sx/n, sy/n, sz/n

    dist_list = [(i, math.dist(p, (cx, cy, cz))) for i, p in pts]
    min_d = min(d for _, d in dist_list)
    cand_indices = [i for (i, d) in dist_list if abs(d - min_d) <= centroid_tol]
    if len(cand_indices) == 1:
        return cand_indices[0]

    best_i = cand_indices[0]
    best_vol = -1.0
    for i in cand_indices:
        w,l,h = get_size_wlh(objects[i])
        vol = bbox_volume(w,l,h)
        if vol > best_vol:
            best_vol = vol
            best_i = i
    return best_i

def build_edges_from_center(objects: list[dict[str, Any]],
                            center_idx: int,
                            eps: float,
                            max_distance: float | None,
                            float_precision: int) -> list[dict[str, Any]]:
    """Create edges from center object to all other objects."""
    edges: list[dict[str, Any]] = []
    center_obj = objects[center_idx]
    center_id = center_obj.get("node_id")
    pc = get_cam_xyz(center_obj)
    csize = get_size_wlh(center_obj)
    if center_id is None or pc is None:
        return edges

    cx, cy, cz_cam, cz_world = pc
    for j, obj in enumerate(objects or []):
        if j == center_idx:
            continue
        pj = get_cam_xyz(obj)
        if pj is None:
            continue
        nid_j = obj.get("node_id")
        if not nid_j:
            continue

        dx, dy, dz_cam = (pj[0]-cx), (pj[1]-cy), (pj[2]-cz_cam)
        dist = math.sqrt(dx*dx + dy*dy + dz_cam*dz_cam)
        if (max_distance is not None) and (dist > max_distance):
            continue

        tsize = get_size_wlh(obj)
        relation = rel_flags(dx, dz_cam, cz_world, pj[3], csize, tsize, eps)
        if sum(relation.values()) == 0:
            # all flags zero -> skip edge
            continue

        angle = math.degrees(math.atan2(dx, dz_cam))
        edges.append({
            "from": center_id,
            "to": nid_j,
            "distance": round(dist, float_precision),
            "angle": round(angle, float_precision),
            "relation": relation
        })
    return edges

def build_edges_from_ego(item: dict[str, Any],
                         objects: list[dict[str, Any]],
                         eps: float,
                         max_distance: float | None,
                         float_precision: int) -> list[dict[str, Any]]:
    """
    Create edges from 'ego' to all objects.
    Assumptions:
      - Ego Camera Coords: (0, 0, 0)
      - Ego Size: (0,0,0)
      - Ego World Z: Taken from item['ego']['translation'][2] if available, else 0.0
    """
    edges: list[dict[str, Any]] = []

    # 1. Ego Configuration
    cx, cy, cz_cam = 0.0, 0.0, 0.0
    csize = (0.0, 0.0, 0.0)

    cz_world = 0.0
    if "ego" in item and "translation" in item["ego"]:
        try:
            cz_world = float(item["ego"]["translation"][2])
        except (ValueError, TypeError, KeyError, IndexError):
            pass

    # 2. Iterate all objects
    for obj in objects:
        nid_j = obj.get("node_id")
        if not nid_j:
            continue

        pj = get_cam_xyz(obj)
        if pj is None:
            continue

        # pj = (x, y, z_cam, z_world)
        # Distance calculation (in Camera Frame)
        dx, dy, dz_cam = (pj[0]-cx), (pj[1]-cy), (pj[2]-cz_cam)
        dist = math.sqrt(dx*dx + dy*dy + dz_cam*dz_cam)

        tsize = get_size_wlh(obj)

        # Relation calculation
        relation = rel_flags(dx, dz_cam, cz_world, pj[3], csize, tsize, eps)
        relation["in front of"], relation["behind"] = relation["behind"], relation["in front of"]

        if sum(relation.values()) == 0:
            continue

        angle = math.degrees(math.atan2(dx, dz_cam))

        edges.append({
            "from": "ego",
            "to": nid_j,
            "distance": round(dist, float_precision),
            "angle": round(angle, float_precision),
            "relation": relation
        })

    return edges

# ---------------- CLI ---------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to merged_nodes_with_captions_cam.json')
    ap.add_argument('--output', required=True, help='Path to save JSON with edges')
    ap.add_argument('--eps', type=float, default=0.2, help='Tolerance (m) for relation flags near zero')
    ap.add_argument('--centroid-tol', type=float, default=0.5, help='Tie tolerance (m) for center selection')
    ap.add_argument('--max-distance', type=float, default=None, help='Max center-to-target distance (m) for edges')
    ap.add_argument('--float-precision', type=int, default=4, help='Decimal places for distance/angle')
    args = ap.parse_args()

    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)

    kept_images = []
    kept_edges_total = 0
    kept_objects_total = 0
    relation_counter = Counter()

    for item in data:
        objs = item.get("objects", []) or []

        # 1. Choose Center Object
        center_idx = choose_center_node(objs, args.centroid_tol)

        if center_idx is None:
            continue

        # 2. Build Object-to-Object Edges
        edges_center = build_edges_from_center(
            objects=objs,
            center_idx=center_idx,
            eps=args.eps,
            max_distance=args.max_distance,
            float_precision=args.float_precision
        )

        # 3. Build Ego-to-Object Edges
        edges_ego = build_edges_from_ego(
            item=item,
            objects=objs,
            eps=args.eps,
            max_distance=args.max_distance,
            float_precision=args.float_precision
        )

        # Combine
        all_edges = edges_center + edges_ego

        if not edges_center:
            continue

        # Update relation distribution stats
        for e in edges_center:
            rel = e.get("relation", {})
            for k in ["right", "left", "in front of", "behind", "up", "down"]:
                if rel.get(k, 0) == 1:
                    relation_counter[k] += 1

        new_item = dict(item)
        new_item["edges"] = all_edges
        kept_images.append(new_item)

        kept_edges_total += len(edges_center)
        kept_objects_total += len(objs)

    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(kept_images, f, ensure_ascii=False, indent=2)

    # ---- Stats ----
    total_images = len(kept_images)
    avg_objects_per_image = (kept_objects_total / total_images) if total_images > 0 else 0.0
    avg_edges_per_image = (kept_edges_total / total_images) if total_images > 0 else 0.0

    per_bucket = Counter()
    for img in kept_images:
        n_obj = len(img.get("objects", []) or [])
        if n_obj in SPARSE_COUNTS:
            per_bucket["sparse"] += 1
        elif n_obj in COM_COUNTS:
            per_bucket["common"] += 1
        elif n_obj in DENSE_COUNTS:
            per_bucket["dense"] += 1

    bucket_total = sum(per_bucket.values())
    bucket_pct = {k: (per_bucket[k] / bucket_total * 100.0 if bucket_total > 0 else 0.0)
                  for k in ["sparse", "common", "dense"]}

    print("=== Dataset summary (kept images) ===")
    print(f"Total images: {total_images}")
    print(f"Average #objects per image: {avg_objects_per_image:.2f}")
    print(f"Average #edges per image: {avg_edges_per_image:.2f}")
    print("Bucket counts:")
    print(f"  sparse   : {per_bucket['sparse']} ({bucket_pct['sparse']:.1f}%)")
    print(f"  common : {per_bucket['common']} ({bucket_pct['common']:.1f}%)")
    print(f"  dense   : {per_bucket['dense']} ({bucket_pct['dense']:.1f}%)")

    total_relation_flags = sum(relation_counter.values())
    print("\nRelation distribution (counts and % over all edges):")
    for k in ["right", "left", "in front of", "behind", "up", "down"]:
        cnt = relation_counter.get(k, 0)
        pct = (cnt / total_relation_flags * 100.0) if total_relation_flags > 0 else 0.0
        print(f"  {k:<12s}: {cnt:6d} ({pct:5.1f}%)")

    print(f"\nSaved filtered JSON to: {args.output}")

if __name__ == '__main__':
    main()
