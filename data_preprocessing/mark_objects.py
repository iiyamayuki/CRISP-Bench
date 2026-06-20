import argparse
import json
import os

import cv2
from tqdm import tqdm

from data_preprocessing.marks_core import (
    DEFAULT_PATH_ROOT,
    NUSCENES_AVAILABLE,
    NuScenes,
    build_output_image_path,
    clip_box_to_image,
    draw_adaptive_centroid,
    ensure_dir,
    find_jsonl_for_image,
    get_box_2d_corners_nusc,
    load_scene_jsonl_lookup,
)
from utils.paths import make_relative, resolve_path

# ==========================================
# Logic for ScanNet++ (Lookup Mode)
# ==========================================

def process_scannetpp(data, output_img_dir, original_jsonl_dir, dataroot, path_root):
    """
    Process ScanNet++ data by looking up 2D bboxes from original source files.
    """
    print(f"[Info] Starting ScanNet++ processing for {len(data)} frames...")

    cache = {"id": None, "data": {}}

    stats = {"success": 0, "missing_file": 0, "missing_bbox": 0}

    for entry in tqdm(data, desc="Processing Images"):
        image_path = entry.get('image')
        if not image_path: continue
        # Resolve relative path to absolute for file access
        abs_image_path = resolve_path(image_path, dataroot)

        # 1. Identify Scene and Load Data
        jsonl_path, scene_id = find_jsonl_for_image(image_path, original_jsonl_dir)

        if not jsonl_path:
            stats["missing_file"] += 1
            continue

        # Lazy Loading: Only reload if the scene ID changes
        if cache["id"] != scene_id:
            cache["data"] = load_scene_jsonl_lookup(jsonl_path) or {}
            cache["id"] = scene_id

        if not cache["data"]:
            continue

        # 2. Load Image
        if not os.path.exists(abs_image_path):
            print(f"[Warning] Image not found: {abs_image_path}")
            continue

        try:
            img = cv2.imread(abs_image_path)
            if img is None: continue
            im_h, im_w = img.shape[:2]
        except Exception as e:
            print(f"[Error] Failed to read {abs_image_path}: {e}")
            continue

        # 3. Match Objects and Draw
        image_filename = os.path.basename(image_path)
        frame_lookup = cache["data"].get(image_filename, {})

        drawn_count = 0
        objects = entry.get('objects', [])

        for idx, obj in enumerate(objects):
            node_id = obj.get('node_id')

            # Retrieve bbox from original source data
            bbox = frame_lookup.get(node_id)

            if not bbox:
                continue # Skip objects without 2D visibility

            clipped_box = clip_box_to_image(bbox, im_w, im_h)
            if clipped_box is None:
                continue

            # Determine ID to display (prefer 'local_id' from merged json, else index)
            local_id = obj.get('local_id', idx + 1)

            img = draw_adaptive_centroid(img, clipped_box, local_id)
            drawn_count += 1

        # 4. Save Result
        if drawn_count > 0:
            stats["success"] += 1
            token = entry.get('sample_data_token', 'unknown')
            save_path = build_output_image_path('scannetpp', output_img_dir, token, scene_id=scene_id)
            cv2.imwrite(save_path, img)

            # Store generated benchmark images relative to the repository root.
            entry['image_with_2dbox'] = make_relative(save_path, path_root)
        else:
            stats["missing_bbox"] += 1

    print(f"[Result] Processing complete. Stats: {stats}")
    return data

def process_nuscenes(data, output_img_dir, nusc, dataroot, path_root):
    """
    Process NuScenes data by projecting 3D boxes onto images.
    """
    print(f"[Info] Starting NuScenes processing for {len(data)} frames...")

    for entry in tqdm(data, desc="Processing NuScenes"):
        sd_token = entry.get('sample_data_token')
        if not sd_token: continue

        # 1. Get Geometry via NuScenes SDK
        try:
            data_path, boxes, camera_intrinsic = nusc.get_sample_data(sd_token)
        except Exception as e:
            print(f"[Error] Failed to load sample data {sd_token}: {e}")
            continue

        if not os.path.exists(data_path): continue

        img = cv2.imread(data_path)
        if img is None: continue
        im_h, im_w = img.shape[:2]

        # 2. Map JSON objects to NuScenes Instances
        user_obj_map = {obj['node_id']: obj for obj in entry.get('objects', [])}
        relevant_boxes = []

        for box in boxes:
            try:
                # NuScenes box.token is annotation token, we need instance token
                ann = nusc.get('sample_annotation', box.token)
                inst_token = ann['instance_token']
                if inst_token in user_obj_map:
                    relevant_boxes.append({'box': box, 'instance_token': inst_token})
            except KeyError:
                continue

        # 3. Draw
        if not relevant_boxes: continue

        for idx, item in enumerate(relevant_boxes, 1):
            box = item['box']
            inst_token = item['instance_token']

            bbox = get_box_2d_corners_nusc(box, camera_intrinsic, im_w, im_h)
            clipped_box = clip_box_to_image(bbox, im_w, im_h)
            if clipped_box is None:
                continue

            # Update 'local_id' in the JSON object for consistency
            user_obj_map[inst_token]['local_id'] = idx

            img = draw_adaptive_centroid(img, clipped_box, idx)

        # 4. Save
        save_path = build_output_image_path('nuscenes', output_img_dir, sd_token)
        cv2.imwrite(save_path, img)
        # Store generated benchmark images relative to the repository root.
        entry['image_with_2dbox'] = make_relative(save_path, path_root)

    return data

# ==========================================
# Main Execution
# ==========================================

def main():
    """CLI entry point for drawing 2D bounding boxes on scene images."""
    parser = argparse.ArgumentParser(description="Visualize scene graphs with 2D bounding boxes.")
    parser.add_argument('--dataset', type=str, choices=['nuscenes', 'scannetpp'], required=True,
                        help="Select dataset mode.")
    parser.add_argument('--input', type=str, required=True, help="Path to the merged scene graph JSON.")
    parser.add_argument('--output', type=str, required=True, help="Output directory for images and JSON.")
    parser.add_argument('--dataroot', type=str, required=True, help="Dataset root directory for resolving/storing relative paths.")
    parser.add_argument('--path_root', type=str, default=DEFAULT_PATH_ROOT,
                        help="Base directory used when storing generated image paths (default: repo root).")

    # ScanNet++ specific argument
    parser.add_argument('--original_jsonl_dir', type=str,
                        help="Directory containing original .jsonl files (Required for scannetpp).")

    # NuScenes specific arguments
    parser.add_argument('--nusc_root', type=str, help="NuScenes root directory (Required for nuscenes).")
    parser.add_argument('--nusc_version', type=str, default='v1.0-trainval', help="NuScenes version.")

    args = parser.parse_args()

    # Load input JSON
    print(f"[Info] Loading input JSON: {args.input}")
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)

    # Prepare output directories
    ensure_dir(args.output)
    img_out_dir = os.path.join(args.output, "images_with_bbox")
    ensure_dir(img_out_dir)
    json_out_path = os.path.join(args.output, "nodes_with_2dbox.json")

    # Dispatch Logic
    updated_data = []

    if args.dataset == 'scannetpp':
        if not args.original_jsonl_dir:
            print("[Error] Argument --original_jsonl_dir is required for ScanNet++ mode.")
            return
        updated_data = process_scannetpp(data, img_out_dir, args.original_jsonl_dir, args.dataroot, args.path_root)

    elif args.dataset == 'nuscenes':
        if not NUSCENES_AVAILABLE:
            print("[Error] 'nuscenes-devkit' is not installed. Cannot run NuScenes mode.")
            return
        if not args.nusc_root:
            print("[Error] Argument --nusc_root is required for NuScenes mode.")
            return

        print(f"[Info] Initializing NuScenes ({args.nusc_version})...")
        assert NuScenes is not None
        nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_root, verbose=True)
        updated_data = process_nuscenes(data, img_out_dir, nusc, args.dataroot, args.path_root)

    # Save final JSON
    print(f"[Info] Saving updated annotations to {json_out_path}...")
    with open(json_out_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=2)
    print("[Info] Done.")

if __name__ == "__main__":
    main()
