import json
import uuid
import os
import argparse
import glob
from pathlib import Path

# ==========================================
# Configuration: Quality Filters
# ==========================================
# ScanNet++ Resized Undistorted images typically are 1752x1168
IMG_W = 1752
IMG_H = 1168

def is_high_quality_object(obj_attrs):
    """
    Applies 4 strict filters to determine if an object is high-quality.
    Returns True if it passes all filters, False otherwise.
    """
    # 0. Data Validity Check
    bbox = obj_attrs.get("bbox_2d")
    trans_cam = obj_attrs.get("translation_camera")
    
    if not bbox or not trans_cam:
        return False

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # =========================================================
    # Filter 1: Depth & Distance 
    # =========================================================
    # z_cam: Depth in camera coordinate system.
    # > 0.3: Avoid objects too close (clipping) or behind camera.
    # < 5.0: Avoid objects too far (lacking detail).
    z_cam = trans_cam[2]
    if z_cam <= 0.2 or z_cam > 5.0:
        return False

    # =========================================================
    # Filter 2: Minimum Relative Area 
    # =========================================================
    # Filter out tiny objects that are hard for VLMs to recognize.
    # Threshold: 0.5% of total image area.
    img_area = IMG_W * IMG_H
    box_area = w * h
    if (box_area / img_area) < 0.005: 
        return False

    # =========================================================
    # Filter 3: Aspect Ratio 
    # =========================================================
    # Filter out extreme slivers (e.g., seen through a crack).
    # Avoid w/h > 5 (too wide/flat) or < 0.2 (too thin/tall).
    if h > 0:
        aspect_ratio = w / h
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            return False
    else:
        return False

    # =========================================================
    # Filter 4: Edge Truncation 
    # =========================================================
    # Check if object touches the image border.
    # Margin of 5 pixels to account for float/int conversion errors.
    margin = 5
    touching_border = (x1 < margin) or (y1 < margin) or \
                      (x2 > IMG_W - margin) or (y2 > IMG_H - margin)
    
    # If touching border, we require high object completion.
    # If it touches border AND completion is low (< 0.6), it's likely heavily truncated.
    completion = obj_attrs.get("object_completion", 1.0)
    if touching_border and completion < 0.6:
        return False

    return True

def convert_entry(source_data):
    """
    Transforms a single line (scene frame) from ScanNet++ JSONL to Target JSON format.
    """
    # Generate a consistent token based on image name
    image_name = source_data.get("image_name", str(uuid.uuid4()))
    generated_token = uuid.uuid5(uuid.NAMESPACE_DNS, image_name).hex

    target_entry = {
        "image": source_data.get("image", ""),
        "sample_data_token": generated_token,
        "camera_channel": "CAM_FRONT", 
        "objects": []
    }

    source_objects = source_data.get("objects", [])
    
    idx = 0
    for obj in source_objects:
        attrs = obj.get("attributes", {})

        # --- [CRITICAL] Apply Quality Filters ---
        if not is_high_quality_object(attrs):
            continue
        # ----------------------------------------
        
        # Defaulting to [0,0,0] if keys are missing
        trans_cam = attrs.get("translation_camera", [0.0, 0.0, 0.0])
        trans_world = attrs.get("translation_world", [0.0, 0.0, 0.0])
        sizes = attrs.get("size_axesLengths", [0.0, 0.0, 0.0])
        
        new_obj = {
            "node_id": obj.get("node_id", ""),
            "attributes": {
                "category_name": attrs.get("category_name", "unknown"),
                "translation": {
                    "x": trans_cam[0],
                    "y": trans_cam[1],
                    "z_cam": trans_cam[2],
                    "z_world": trans_world[2]
                },
                "size": {
                    # Swapped indices 0 and 1 based on typical [w, l, h] vs [l, w, h] needs
                    # Using min/max logic from your snippet to ensure consistency
                    "w": min(sizes[0], sizes[2]), 
                    "l": max(sizes[0], sizes[2]), 
                    "h": sizes[1] 
                },
                # Note: Adding bbox_2d to output is helpful for debugging, 
                # even if not strictly in target format. Optional.
                # "bbox_2d": attrs.get("bbox_2d"), 
                
                "caption": f"A {attrs.get('category_name', 'object')}"
            },
            "local_id": idx + 1
        }
        target_entry["objects"].append(new_obj)
        idx += 1
        
    return target_entry

def process_folder(input_dir, output_file):
    """
    Reads all .jsonl files in input_dir, converts them, and saves to output_file.
    """
    merged_data = []
    
    search_path = os.path.join(input_dir, "*.jsonl")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"[Warning] No .jsonl files found in {input_dir}")
        return

    print(f"[Info] Found {len(files)} files. Starting conversion with Quality Filters...")

    total_frames = 0
    total_objects_kept = 0
    
    for file_path in files:
        print(f"Processing: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue 
                    
                    try:
                        source_obj = json.loads(line)
                        converted_obj = convert_entry(source_obj)
                        
                        # Only add frames that still have objects after filtering
                        if converted_obj["objects"]:
                            merged_data.append(converted_obj)
                            total_frames += 1
                            total_objects_kept += len(converted_obj["objects"])
                            
                    except json.JSONDecodeError:
                        print(f"[Error] Invalid JSON at {file_path} line {line_number + 1}")
                        
        except Exception as e:
            print(f"[Error] Failed to read file {file_path}: {e}")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[Info] Writing {total_frames} valid frames (Total {total_objects_kept} objects) to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2)
        
    print("[Success] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ScanNet++ JSONL to Target JSON with Quality Filters.")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder with .jsonl files")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output .json file")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"[Error] Directory '{args.input_dir}' not found.")
    else:
        process_folder(args.input_dir, args.output_file)