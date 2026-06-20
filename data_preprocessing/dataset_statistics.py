#!/usr/bin/env python3
"""
Statistics script for merged_nodes_with_captions_cam.json (or similar).

It reports:
1. Total number of images.
2. Average number of unique categories per image.
3. Total sample count per category.
4. Distribution of object counts per image
   (e.g., how many images contain N objects).
"""

import argparse
import json
from collections import Counter


def main():
    """CLI entry point for computing per-category and per-image object statistics."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to merged_nodes_with_captions_cam.json')
    args = ap.parse_args()

    # Load the merged annotations file
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)

    num_images = len(data)
    category_counter = Counter()      # Counts total objects per category
    per_image_object_counts = []      # Total number of objects per image

    for item in data:
        objects = item.get('objects', []) or []
        cats_in_image = set()

        for obj in objects:
            attrs = obj.get('attributes', {}) or {}
            cat = attrs.get('category_name')
            if cat:
                cats_in_image.add(cat)
                category_counter[cat] += 1

        per_image_object_counts.append(len(objects))

    # Compute averages and distributions
    avg_objects_per_image = (
        sum(per_image_object_counts) / num_images if num_images > 0 else 0
    )

    # Count how many images have N objects
    object_count_distribution = Counter(per_image_object_counts)

    # ---- Print results ----
    print(f"Total number of images: {num_images}")
    print(f"Average number of objects per image: {avg_objects_per_image:.2f}\n")

    print("Total sample count per category:")
    for cat, cnt in sorted(category_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:<40s} {cnt}")

    print("\nDistribution of object counts per image:")
    for obj_count, img_count in sorted(object_count_distribution.items()):
        print(f"  {obj_count:>3d} objects: {img_count} images")

if __name__ == '__main__':
    main()

