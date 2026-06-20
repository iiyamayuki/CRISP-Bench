#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""scannetpp_sg_final_robust.py

**ROBUST FIX VERSION**:
1. Uses Modulo operator (%) to fix mesh face indices, preventing negative index errors.
2. Reads aligned poses from dslr/nerfstudio/transforms_undistorted.json.
3. Applies "Mode 2" coordinate fix.
4. Auto-scales intrinsics.
5. Includes advanced label filtering.
"""

from __future__ import annotations

import argparse
import json
import re
import cv2
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
import torch
import open3d as o3d

# scannetpp official utils
from common.utils import rasterize as spp_rast
from common.utils import anno as spp_anno

from utils.paths import make_relative

# --------------------------
# Advanced Filtering Logic
# --------------------------

DEFAULT_DROP = {
    "wall", "floor", "ceiling", "baseboard", "molding", "pillar", "column",
    "stairs", "stair", "step", "steps",
    "door frame", "window frame",
    "room", "background",
}

WHITELIST_EXCEPTIONS = {
    "ceiling lamp", "ceiling light", "ceiling fan",
    "floor lamp", "floor mat", "floor cushion",
    "wall cabinet", "wall mirror", "wall clock", "wall shelf",
    "door", "window", "window blind", "window sill",
    "pipe storage rack",
}

STUFF_BLACKLIST_KEYWORDS = {
    # Structure
    "wall", "floor", "ceiling", "roof", "pillar", "column", "beam", "post",
    "stair", "step", "railing", "handrail", "baseboard", "molding", "sill",
    # Frame
    "frame", "doorway", "threshold", "casing", "lintel",
    # Utility
    "pipe", "duct", "cable", "wire", "conduit", "junction", "raceway",
    "ventilation", "ventilator", "exhaust", "vent",
    # Electric (infrastructure)
    "electrical", "electric", "power panel", "control panel", "switch board",
    "fuse box", "circuit box", "electrical panel",
    # Surface
    "tile", "brick", "panel", "wallpaper", "backsplash",
    # Generic
    "structure", "structural", "support", "bracket", "mount",
    "background", "room", "space",
}

def contains_blacklist_keyword(label: str, blacklist: set) -> bool:
    """Check if label contains blacklist keyword (word boundary aware)."""
    label_lower = label.lower()
    for keyword in blacklist:
        keyword_lower = keyword.lower()
        if label_lower == keyword_lower:
            return True
        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
        if re.search(pattern, label_lower):
            return True
    return False

def should_keep_label(
    label_norm: str, 
    keep_set: Optional[set], 
    drop_set: set,
    use_blacklist_keywords: bool = True
) -> bool:
    if label_norm in WHITELIST_EXCEPTIONS:
        return True
    if keep_set is not None:
        return label_norm in keep_set
    if label_norm in drop_set:
        return False
    if use_blacklist_keywords:
        if contains_blacklist_keyword(label_norm, STUFF_BLACKLIST_KEYWORDS):
            return False
    return True

# --------------------------
# Coordinate System Fixes
# --------------------------

def opengl_to_opencv_c2w(c2w: np.ndarray) -> np.ndarray:
    S = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return c2w @ S

def fix_swapped_nerfstudio_c2w(c2w: np.ndarray) -> np.ndarray:
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    return opengl_to_opencv_c2w(c2w)

def world_to_camera(world_pos: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    world_pos = np.asarray(world_pos)
    is_single = (world_pos.ndim == 1)
    if is_single: world_pos = world_pos.reshape(1, 3)
    ones = np.ones((world_pos.shape[0], 1))
    world_pos_homo = np.concatenate([world_pos, ones], axis=1)
    camera_pos_homo = (w2c @ world_pos_homo.T).T
    camera_pos = camera_pos_homo[:, :3]
    if is_single: return camera_pos[0]
    return camera_pos

def bbox_xywh_to_xyxy_rowcol(bbox_xywh: List[float]) -> Tuple[float, float, float, float]:
    r0, c0, rh, cw = map(float, bbox_xywh)
    xmin, ymin = c0, r0
    xmax, ymax = c0 + cw, r0 + rh
    return xmin, ymin, xmax, ymax

def normalize_label(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

# --------------------------
# Main Processing
# --------------------------

def compute_frame_objects(
    mesh_o3d, faces_np, pix_to_face, vertex_obj_ids, obj_verts_index, objects_catalog,
    w2c, H, W, args,
    keep_set, drop_set
):
    valid_ratio = float(np.mean(pix_to_face != -1))
    if valid_ratio < 0.001: return []

    # Map mesh vertex IDs to 2D pixel map
    pix_obj_ids = spp_anno.get_vtx_prop_on_2d(pix_to_face, vertex_obj_ids, mesh_o3d)
    bboxes = spp_anno.get_bboxes_2d(pix_obj_ids)
    img_area = H * W
    out = []

    for obj_id, bbox_xywh in bboxes.items():
        if obj_id not in objects_catalog: continue
        
        obj_info = objects_catalog[obj_id]
        label = normalize_label(obj_info.get("label", ""))
        
        if not should_keep_label(label, keep_set, drop_set, use_blacklist_keywords=True):
            continue

        xmin, ymin, xmax, ymax = bbox_xywh_to_xyxy_rowcol(bbox_xywh)
        bbox_w, bbox_h = xmax - xmin, ymax - ymin
        
        # Filters
        if max(bbox_w, bbox_h) < args.min_bbox_px: continue
        if (bbox_w * bbox_h) / img_area < args.min_bbox_area_ratio: continue

        obj_pixel_mask = (pix_obj_ids == obj_id)
        num_visible_pixels = int(np.sum(obj_pixel_mask))
        
        visible_pixels_frac = num_visible_pixels / img_area
        if visible_pixels_frac < args.min_visible_pixels_frac: continue

        bbox_area = max(1, bbox_w * bbox_h)
        completion = min(1.0, num_visible_pixels / (bbox_area * 0.6)) 
        if completion < args.min_object_completion: continue

        obj_verts = obj_verts_index.get(obj_id, np.array([], dtype=np.int64))
        if len(obj_verts) == 0: continue
        
        face_ndx = pix_to_face[pix_to_face != -1].astype(np.int64)
        if len(face_ndx) == 0: continue
        
        faces_in_img = faces_np[face_ndx]
        img_verts = np.unique(faces_in_img)
        inter = np.intersect1d(obj_verts, img_verts)
        vis_vert_frac = len(inter) / max(1, len(obj_verts))
        
        if vis_vert_frac < args.min_visible_vertices_frac: continue

        obb = obj_info.get("obb", {})
        centroid_world = obb.get("centroid")
        axes_lengths = obb.get("axesLengths")
        if not centroid_world: continue
        
        centroid_cam = world_to_camera(centroid_world, w2c).tolist()

        out.append({
            "node_id": str(obj_id),
            "attributes": {
                "category_name": label,
                "bbox_2d": [xmin, ymin, xmax, ymax],
                "object_completion": round(completion, 4),
                "translation_world": centroid_world,
                "translation_camera": centroid_cam,
                "size_axesLengths": axes_lengths
            }
        })
    return out

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to scene dir")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4)
    # Filters
    parser.add_argument("--min_bbox_px", type=int, default=50)
    parser.add_argument("--min_bbox_area_ratio", type=float, default=0.001)
    parser.add_argument("--min_visible_vertices_frac", type=float, default=0.05)
    parser.add_argument("--min_visible_pixels_frac", type=float, default=0.001)
    parser.add_argument("--min_object_completion", type=float, default=0.0)
    parser.add_argument("--keep_labels", type=str, default="")
    parser.add_argument("--drop_labels", type=str, default="")
    parser.add_argument("--dataroot", type=str, required=True,
                        help="Dataset root for ScanNet++ (paths stored relative to this)")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spp_rast.device = dev
    print(f"Using device: {dev}")

    scene_dir = Path(args.input)
    scene_id = scene_dir.name
    img_dir = scene_dir / "dslr" / "resized_undistorted_images"
    tf_path = scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    mesh_path = scene_dir / "scans" / "mesh_aligned_0.05.ply"
    seg_anno_path = scene_dir / "scans" / "segments_anno.json"
    seg_path = scene_dir / "scans" / "segments.json"

    if not img_dir.exists(): raise FileNotFoundError(f"{img_dir} missing")
    if not tf_path.exists(): raise FileNotFoundError(f"{tf_path} missing")
    if not mesh_path.exists(): raise FileNotFoundError(f"{mesh_path} missing")

    print("Loading annotations...")
    anno = spp_anno.load_annotation(str(seg_anno_path), bboxes_only=True, segments_path=str(seg_path), return_vertex_obj_ids=True)
    vertex_obj_ids = anno["vertex_obj_ids"]
    objects_catalog = anno["objects"]
    u_ids = np.unique(vertex_obj_ids)
    u_ids = u_ids[u_ids > 0]
    obj_verts_index = {int(oid): np.where(vertex_obj_ids == oid)[0] for oid in u_ids}

    print("Loading mesh...")
    mesh_o3d = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh_o3d.vertices) == 0: raise ValueError("Mesh is empty!")
    
    verts, faces, meshes_single = spp_rast.prep_pt3d_inputs(mesh_o3d)
    meshes_single = meshes_single.to(dev)
    num_faces = int(meshes_single.num_faces_per_mesh().cpu().numpy()[0])
    faces_np = np.array(mesh_o3d.triangles, dtype=np.int64)

    with open(tf_path) as f:
        tf = json.load(f)
    
    frames = []
    poses_w2c = []
    
    if not tf.get("frames"): raise ValueError("No frames in transforms.json")
    first_img_path = img_dir / tf["frames"][0]["file_path"]
    if not first_img_path.exists():
        raise FileNotFoundError(f"Image not found: {first_img_path}")
    
    tmp_img = cv2.imread(str(first_img_path))
    H_real, W_real = tmp_img.shape[:2]
    
    W_tf, H_tf = int(tf["w"]), int(tf["h"])
    scale_x = W_real / W_tf
    scale_y = H_real / H_tf
    fx = float(tf["fl_x"]) * scale_x
    fy = float(tf["fl_y"]) * scale_y
    cx = float(tf["cx"]) * scale_x
    cy = float(tf["cy"]) * scale_y
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    print(f"[INFO] Image: {W_real}x{H_real}, K scaled: fx={fx:.1f}, fy={fy:.1f}")

    print("Preparing poses...")
    for fr in tf["frames"]:
        p = img_dir / fr["file_path"]
        if p.exists():
            c2w = np.array(fr["transform_matrix"], dtype=np.float32)
            c2w = fix_swapped_nerfstudio_c2w(c2w)
            w2c = np.linalg.inv(c2w)
            frames.append(fr)
            poses_w2c.append(w2c)
            
    poses_w2c = np.stack(poses_w2c)
    
    drop_set = set(DEFAULT_DROP)
    if args.drop_labels.strip():
        drop_set |= {normalize_label(x) for x in args.drop_labels.split(",") if x.strip()}
    keep_set = None
    if args.keep_labels.strip():
        keep_set = {normalize_label(x) for x in args.keep_labels.split(",") if x.strip()}

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_id}.jsonl"
    
    print(f"Processing {len(frames)} frames...")
    
    with open(out_path, "w", encoding='utf-8') as f:
        for start in tqdm(range(0, len(frames), args.batch_size)):
            end = min(start + args.batch_size, len(frames))
            batch_poses = poses_w2c[start:end]
            batch_frames = frames[start:end]
            bsize = len(batch_poses)
            
            poses_tensor = torch.from_numpy(batch_poses).float()
            cameras = spp_rast.get_opencv_cameras_batch(poses_tensor, H_real, W_real, K)
            
            # Extend mesh for batch
            meshes_batch = meshes_single.extend(bsize)
            
            raster = spp_rast.rasterize_mesh(meshes_batch, H_real, W_real, cameras)
            pix_batch = raster["pix_to_face"].cpu().numpy()
            
            for i in range(bsize):
                pix = pix_batch[i].squeeze()
                
                # [CRITICAL FIX] Use Modulo to handle index offsets robustly
                valid = pix != -1
                if valid.any():
                    pix[valid] = pix[valid] % num_faces
                
                objs = compute_frame_objects(
                    mesh_o3d, faces_np, pix, vertex_obj_ids, obj_verts_index, objects_catalog,
                    batch_poses[i], H_real, W_real, args, keep_set, drop_set
                )
                
                if objs:
                    image_abs = str(img_dir / batch_frames[i]["file_path"])
                    record = {
                        "scene_id": scene_id,
                        "image": make_relative(image_abs, args.dataroot),
                        "objects": objs
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Saved to {out_path}")

if __name__ == "__main__":
    main()