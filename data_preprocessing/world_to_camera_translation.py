#!/usr/bin/env python3
"""
Convert world-frame translations in merged_nodes_with_captions.json to camera-frame
using nuScenes devkit. node_id is instance_token. Only process objects present in the input
(list is partial). Optionally verify/sync with official sample_annotation of that instance.

Also adds 'ego' key to each item containing the ego vehicle's world state.

Usage:
  python world_to_cam_translation_plus.py \
      --input merged_nodes_with_captions.json \
      --output merged_nodes_with_captions_cam.json \
      --dataroot /path/to/nuscenes \
      --version v1.0-trainval \
      [--verify-instance] [--sync-attrs] [--tolerance 0.5] \
      [--float-precision 6]
"""

import argparse
import json

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def world_to_cam_point(x_w, t_we, R_we, t_ec, R_ec):
    """ global(world) -> ego -> camera """
    x_w = np.asarray(x_w, dtype=float)
    t_we = np.asarray(t_we, dtype=float)
    t_ec = np.asarray(t_ec, dtype=float)
    # Step 1: World -> Ego
    x_e = R_we.T @ (x_w - t_we)
    # Step 2: Ego -> Camera
    x_c = R_ec.T @ (x_e - t_ec)
    return x_c

def get_pose_chain_from_sample_data(nusc: NuScenes, sd_token: str):
    """
    Return transformation matrices and raw ego_pose record for a camera sample_data token.
    Returns: (t_we, R_we, t_ec, R_ec, sample_token, ego_pose_record)
    """
    sd = nusc.get('sample_data', sd_token)
    ego_pose = nusc.get('ego_pose', sd['ego_pose_token'])
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

    # World -> Ego translation and rotation matrix
    t_we = np.array(ego_pose['translation'], dtype=float)
    R_we = Quaternion(ego_pose['rotation']).rotation_matrix

    # Ego -> Camera translation and rotation matrix
    t_ec = np.array(cs['translation'], dtype=float)
    R_ec = Quaternion(cs['rotation']).rotation_matrix

    return t_we, R_we, t_ec, R_ec, sd['sample_token'], ego_pose

def build_instance_index_for_sample(nusc: NuScenes, sample_token: str):
    """Build a dict mapping instance_token -> annotation record for a given sample."""
    sample = nusc.get('sample', sample_token)
    idx = {}
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        idx[ann['instance_token']] = ann
    return idx

def maybe_verify_and_sync(attrs_in, ann_official, do_sync=False, tol=0.5):
    """Optionally verify node attributes against official annotations and sync if requested."""
    notes = []
    if ann_official is None:
        notes.append("no_official_ann")
        return attrs_in, notes

    if 'translation' in attrs_in:
        tw_user = np.array([attrs_in['translation']['x'],
                            attrs_in['translation']['y'],
                            attrs_in['translation']['z']], dtype=float)
        tw_off = np.array(ann_official['translation'], dtype=float)
        d = np.linalg.norm(tw_user - tw_off)
        if d > tol:
            notes.append(f"translation_mismatch_{d:.3f}m")

    if do_sync:
        if 'category_name' in ann_official:
            attrs_in['category_name'] = ann_official['category_name']
        if 'size' in ann_official:
            w, l, h = ann_official['size']
            attrs_in['size'] = {'w': w, 'l': l, 'h': h}
        if 'rotation' in ann_official:
            attrs_in['rotation'] = {
                'qw': ann_official['rotation'][0],
                'qx': ann_official['rotation'][1],
                'qy': ann_official['rotation'][2],
                'qz': ann_official['rotation'][3],
            }
        notes.append("synced_attrs")
    return attrs_in, notes

def main():
    """CLI entry point for translating world-frame coordinates to camera frame."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--dataroot', required=True)
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--float-precision', type=int, default=6)
    ap.add_argument('--verify_instance', action='store_true',
                    help='verify the translation against official annotation if found.')
    ap.add_argument('--sync-attrs', action='store_true',
                    help='sync category/size/rotation to output if official annotation is found.')
    ap.add_argument('--tolerance', type=float, default=0.5,
                    help='tolerance (m) for world coordinates vs official annotation.')
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)

    # Updated cache structure:
    # sd_token -> (t_we, R_we, t_ec, R_ec, sample_token, ego_pose_rec)
    sd_pose_cache = {}
    sample_inst_index_cache = {}

    stats = {
        'images': 0,
        'objects': 0,
        'behind_cam': 0,
        'no_official_ann': 0,
        'translation_mismatch': 0,
        'synced_attrs': 0
    }

    for item in data:
        sd_token = item.get('sample_data_token')
        if not sd_token:
            continue

        if sd_token not in sd_pose_cache:
            t_we, R_we, t_ec, R_ec, sample_token, ego_pose_rec = get_pose_chain_from_sample_data(nusc, sd_token)
            sd_pose_cache[sd_token] = (t_we, R_we, t_ec, R_ec, sample_token, ego_pose_rec)
        else:
            t_we, R_we, t_ec, R_ec, sample_token, ego_pose_rec = sd_pose_cache[sd_token]

        # nuScenes rotation format is quaternion [w, x, y, z]
        item['ego'] = {
            'translation': ego_pose_rec['translation'], # [x, y, z]
            'rotation': ego_pose_rec['rotation'],       # [w, x, y, z]
        }

        inst_index = None
        if args.verify_instance or args.sync_attrs:
            if sample_token not in sample_inst_index_cache:
                sample_inst_index_cache[sample_token] = build_instance_index_for_sample(nusc, sample_token)
            inst_index = sample_inst_index_cache[sample_token]

        for obj in item.get('objects', []):
            attrs = obj.get('attributes', {})
            inst_token = obj.get('node_id')
            trans = attrs.get('translation')

            if not trans or not all(k in trans for k in ('x', 'y', 'z')):
                continue

            official_ann = None
            if inst_index is not None and inst_token in inst_index:
                official_ann = inst_index[inst_token]
            elif args.verify_instance:
                stats['no_official_ann'] += 1

            notes = []
            if args.verify_instance or args.sync_attrs:
                attrs, notes = maybe_verify_and_sync(attrs, official_ann,
                                                     do_sync=args.sync_attrs,
                                                     tol=args.tolerance)
                if any(n.startswith('translation_mismatch') for n in notes):
                    stats['translation_mismatch'] += 1
                if 'no_official_ann' in notes:
                    stats['no_official_ann'] += 1
                if 'synced_attrs' in notes:
                    stats['synced_attrs'] += 1


            x_w = np.array([attrs['translation']['x'],
                            attrs['translation']['y'],
                            attrs['translation']['z']], dtype=float)
            x_c = world_to_cam_point(x_w, t_we, R_we, t_ec, R_ec)

            if x_c[2] <= 0:
                stats['behind_cam'] += 1

            attrs['translation'] = {
                'x': round(float(x_c[0]), args.float_precision),
                'y': round(float(x_c[1]), args.float_precision),
                'z_cam': round(float(x_c[2]), args.float_precision),
                'z_world': round(float(x_w[2]), args.float_precision),
            }

            obj['attributes'] = attrs
            stats['objects'] += 1

        stats['images'] += 1

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "processed_images": stats['images'],
        "processed_objects": stats['objects'],
        "objects_behind_camera": stats['behind_cam'],
        "no_official_ann_found": stats['no_official_ann'],
        "world_vs_official_translation_mismatch": stats['translation_mismatch'],
        "synced_attrs_count": stats['synced_attrs'],
        "output": args.output
    }, indent=2))

if __name__ == '__main__':
    main()
