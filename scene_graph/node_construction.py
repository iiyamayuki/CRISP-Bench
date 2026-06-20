import argparse
import json
import os
from typing import Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from tqdm.auto import tqdm

from utils.paths import make_relative


def observed_annos_for_image(nusc: NuScenes,
                             sd_token: str,
                             vis_level=BoxVisibility.ANY,
                             min_box_h_px: int = 0,
                             min_inside_ratio: float = 0.0) -> dict[str, Any]:
    """Extract visible object annotations for a single camera sample_data token."""
    sd = nusc.get('sample_data', sd_token)
    img_path, boxes, K = nusc.get_sample_data(sd_token,
                                              box_vis_level=vis_level,
                                              use_flat_vehicle_coordinates=False)
    W, H = sd['width'], sd['height']
    K = np.array(K)

    objects = []
    for box in boxes:
        # bbox
        corners = box.corners()                           # (3,8) in this camera coord
        pixels = view_points(corners, K, normalize=True)  # (3,8)
        xs, ys = pixels[0], pixels[1]
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

        # inside_ratio
        bx1, by1 = np.clip([x1, y1], [0, 0], [W - 1, H - 1])
        bx2, by2 = np.clip([x2, y2], [0, 0], [W - 1, H - 1])
        w, h = bx2 - bx1, by2 - by1
        if w <= 1 or h <= 1:
            continue
        orig_w, orig_h = (x2 - x1), (y2 - y1)
        inside_ratio = (w * h) / max(orig_w * orig_h, 1e-6)

        if h < min_box_h_px or inside_ratio < min_inside_ratio:
            continue

        # get sample_annotation
        ann = nusc.get('sample_annotation', box.token)

        wlh = ann['size']  # [w, l, h] in nuScenes devkit
        obj = {
            "node_id": ann["instance_token"],
            "attributes": {
                "category_name": ann["category_name"],
                "translation": {
                    "x": float(ann["translation"][0]),
                    "y": float(ann["translation"][1]),
                    "z": float(ann["translation"][2]),
                },
                "size": {
                    "w": float(wlh[0]),
                    "l": float(wlh[1]),
                    "h": float(wlh[2]),
                },
                "caption": None
            }
        }
        objects.append(obj)

    return {
        "image": img_path,
        "sample_data_token": sd_token,
        "camera_channel": sd["channel"],
        "objects": objects
    }

def export_dataset(dataroot: str,
                   version: str,
                   out_json: str,
                   camera_channels: list[str] = None,
                   vis_level=BoxVisibility.ANY,
                   min_box_h_px: int = 0,
                   min_inside_ratio: float = 0.0,
                   keyframes_only: bool = True,
                   show_progress: bool = True) -> None:
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    if camera_channels is None:
        camera_channels = [
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
        ]

    sd_list = [
        sd for sd in nusc.sample_data
        if sd["channel"] in camera_channels and (sd["is_key_frame"] or not keyframes_only)
    ]

    results = []
    total_objs = 0

    iterator = tqdm(sd_list, desc="Exporting camera-observed annotations",
                    total=len(sd_list), dynamic_ncols=True, disable=not show_progress)

    for sd in iterator:
        sd_token = sd["token"]
        sample_entry = observed_annos_for_image(
            nusc,
            sd_token,
            vis_level=vis_level,
            min_box_h_px=min_box_h_px,
            min_inside_ratio=min_inside_ratio
        )
        # Store image path relative to dataroot
        sample_entry["image"] = make_relative(sample_entry["image"], dataroot)
        results.append(sample_entry)
        total_objs += len(sample_entry.get("objects", []))
        if show_progress:
            iterator.set_postfix({
                "images": len(results),
                "objects": total_objs,
                "channel": sd["channel"]
            })

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} images / {total_objs} objects to {out_json}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', required=True, help='NuScenes dataset root directory')
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--out-json', required=True, help='Output JSON path')
    ap.add_argument('--min-box-h-px', type=int, default=40)
    ap.add_argument('--min-inside-ratio', type=float, default=0.2)
    args = ap.parse_args()

    export_dataset(
        dataroot=args.dataroot,
        version=args.version,
        out_json=args.out_json,
        camera_channels=None,                 # None = all 6 cameras
        vis_level=BoxVisibility.ANY,          # clip only fully/partially visible boxes
        min_box_h_px=args.min_box_h_px,
        min_inside_ratio=args.min_inside_ratio,
        keyframes_only=True                   # only traverse keyframes (recommended)
    )
