import json
import os
from collections import Counter
from itertools import combinations

import numpy as np
from PIL import Image
from tqdm import tqdm

############################################################
#                    Helper Functions
############################################################

def load_scene_graphs(path):
    """Load scene graphs from a JSON or JSONL file."""
    if path.endswith(".jsonl"):
        data = [json.loads(line) for line in open(path)]
    else:
        data = json.load(open(path))
    return data


def extract_category(obj_list):
    """Return a mapping from node_id to category_name."""
    return {obj["node_id"]: obj["attributes"]["category_name"] for obj in obj_list}


def relation_vector(rel_dict):
    """Extract a directional relation tuple from a relation dict."""
    keys = ["in front of", "behind", "left", "right", "up", "down"]
    return tuple(rel_dict[k] for k in keys)


def distance_bin(d):
    """Discretize distance into your three bins + fallback."""
    if d < 5:
        return "d0_5"
    elif d < 15:
        return "d5_15"
    elif d < 25:
        return "d15_25"
    else:
        return "d25_plus"


def build_edge_signature(scene):
    """
    Build enhanced edge signature:
    (from_category, to_category, relation_vector_bin, distance_bin)
    """
    node2cat = extract_category(scene["objects"])
    signature = set()

    for e in scene["edges"]:
        if e["from"] == "ego":
            continue
        from_cat = node2cat[e["from"]]
        to_cat   = node2cat[e["to"]]
        rel_vec = relation_vector(e["relation"])
        dist_bin = distance_bin(e["distance"])

        signature.add((from_cat, to_cat, rel_vec, dist_bin))

    return signature


def jaccard_set(A, B):
    """Compute Jaccard similarity between two sets."""
    if len(A) == 0 and len(B) == 0:
        return 1.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union


############################################################
#         CLIP embeddings & similarity
############################################################

def compute_clip_embeddings(image_paths, clip_model, preprocess, device, batch_size=32):
    """Encode all images into a single (N, D) tensor with CLIP."""
    all_feats = []
    clip_model.eval()

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(preprocess(img))
                except Exception:
                    continue
            if not imgs:
                continue
            batch = torch.stack(imgs, dim=0).to(device)
            feats = clip_model.encode_image(batch)          # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
            all_feats.append(feats.cpu())

    if not all_feats:
        return None

    feats = torch.cat(all_feats, dim=0)   # (N, D)
    return feats


############################################################
#                Edge-based Filtering
############################################################

def edge_filter(scenes, jaccard_threshold=0.9):
    """Remove near-duplicate scenes based on edge signature Jaccard similarity."""
    N = len(scenes)
    signatures = [build_edge_signature(s) for s in scenes]
    edge_counts = [len(sig) for sig in signatures]
    keep = [True] * N

    print("\nComputing edge Jaccard similarities...")
    for i, j in tqdm(list(combinations(range(N), 2))):
        if not (keep[i] and keep[j]):
            continue

        J = jaccard_set(signatures[i], signatures[j])
        if J >= jaccard_threshold:
            # delete the one with fewer edges
            if edge_counts[i] >= edge_counts[j]:
                keep[j] = False
            else:
                keep[i] = False

    filtered = [scenes[i] for i in range(N) if keep[i]]
    return filtered


############################################################
#                  Statistics Functions
############################################################

def get_object_number_bins(num_objects):
    """Classify object count into sparse/common/dense bins."""
    if 2 <= num_objects <= 3:
        return "sparse"
    elif 4 <= num_objects <= 8:
        return "common"
    elif 9 <= num_objects <= 12:
        return "dense"
    else:
        return "ignored"

def get_difficulty(num_edges):
    """Classify edge count into easy/medium/hard difficulty levels."""
    if 1 <= num_edges <= 2:
        return "easy"
    elif 3 <= num_edges <= 7:
        return "medium"
    elif 8 <= num_edges <= 11:
        return "hard"
    else:
        return "ignored"

def collect_statistics(scenes):
    """Compute aggregate statistics over a list of filtered scene graphs."""
    object_num_bins = Counter()
    diff_counter = Counter()
    relation_counter = Counter()
    distances = []
    angles = []
    sizes_w = []
    sizes_l = []
    sizes_h = []

    for s in scenes:
        num_obj = len(s["objects"])
        num_edges = len(s["edges"]) - num_obj
        object_num_bins[get_object_number_bins(num_obj)] += 1
        diff_counter[get_difficulty(num_edges)] += 1

        for e in s["edges"]:
            if e["from"] == "ego":
                continue
            rel = relation_vector(e["relation"])

            # relation stats
            rel_names = ["front", "behind", "left", "right", "up", "down"]
            for name, flag in zip(rel_names, rel):
                if flag == 1:
                    relation_counter[name] += 1

            distances.append(e["distance"])
            angles.append(e["angle"])

        # size stats
        for obj in s["objects"]:
            size = obj["attributes"]["size"]
            sizes_w.append(size["w"])
            sizes_l.append(size["l"])
            sizes_h.append(size["h"])

    stats = {
        "object_num_bins": object_num_bins,
        "difficulty_counts": diff_counter,
        "relation_counts": relation_counter,
        "distances": distances,
        "angles": angles,
        "sizes_w": sizes_w,
        "sizes_l": sizes_l,
        "sizes_h": sizes_h,
    }
    return stats


############################################################
#                       Main
############################################################

def main(json_path, out_path, jaccard_threshold=0.9, compute_clip_avg=False, dataroot=None):
    """CLI entry point: load, filter, save, and print statistics."""
    print("Loading scene graphs...")
    scenes = load_scene_graphs(json_path)

    # Step 1: Edge filter
    filtered_scenes = edge_filter(scenes, jaccard_threshold=jaccard_threshold)

    print(f"\nOriginal: {len(scenes)}")
    print(f"Filtered: {len(filtered_scenes)}")

    # Step 2: Save results
    with open(out_path, "w") as f:
        json.dump(filtered_scenes, f, indent=2)
    print(f"\nFiltered results saved to {out_path}")

    # Step 3: Statistics
    stats = collect_statistics(filtered_scenes)

    print("\n========== STATISTICS ==========")

    total = len(filtered_scenes)
    print(f"Total samples: {total}")

    # Object number bins
    print("\n--- Object numbers Distribution ---")
    for k in ["sparse", "common", "dense"]:
        cnt = stats["object_num_bins"].get(k, 0)
        pct = cnt / total * 100
        print(f"{k:6s}: {cnt:4d} ({pct:5.2f}%)")

    # Difficulty
    print("\n--- Difficulty Distribution ---")
    for k in ["easy", "medium", "hard"]:
        cnt = stats["difficulty_counts"].get(k, 0)
        pct = cnt / total * 100
        print(f"{k:6s}: {cnt:4d} ({pct:5.2f}%)")

    # Relations
    print("\n--- Spatial Relation Distribution ---")
    relation_total = sum(stats["relation_counts"].values())
    for k in ["front", "behind", "left", "right", "up", "down"]:
        cnt = stats["relation_counts"].get(k, 0)
        pct = (cnt / relation_total * 100) if relation_total > 0 else 0
        print(f"{k:8s}: {cnt:6d} ({pct:5.2f}%)")

    # Numeric stats helper
    def summary(arr):
        arr = np.array(arr)
        return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)

    # Distance stats
    mean_d, std_d, min_d, max_d = summary(stats["distances"])
    print("\n--- Distance Stats ---")
    print(f"mean={mean_d:.3f}, std={std_d:.3f}, min={min_d:.3f}, max={max_d:.3f}")

    # Angle stats
    mean_a, std_a, min_a, max_a = summary(stats["angles"])
    print("\n--- Angle Stats ---")
    print(f"mean={mean_a:.3f}, std={std_a:.3f}, min={min_a:.3f}, max={max_a:.3f}")

    # Size stats
    mean_w, std_w, min_w, max_w = summary(stats["sizes_w"])
    mean_l, std_l, min_l, max_l = summary(stats["sizes_l"])
    mean_h, std_h, min_h, max_h = summary(stats["sizes_h"])

    print("\n--- Object Size Stats ---")
    print(f"Width : mean={mean_w:.3f}, std={std_w:.3f}, min={min_w:.3f}, max={max_w:.3f}")
    print(f"Length: mean={mean_l:.3f}, std={std_l:.3f}, min={min_l:.3f}, max={max_l:.3f}")
    print(f"Height: mean={mean_h:.3f}, std={std_h:.3f}, min={min_h:.3f}, max={max_h:.3f}")


    # Optional CLIP similarity
    if compute_clip_avg:
        print("\nComputing CLIP similarities...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, preprocess = clip.load("ViT-B/16", device=device)

        image_paths = [s["image"] for s in filtered_scenes]
        # Resolve relative paths if dataroot is provided
        if dataroot:
            image_paths = [os.path.join(dataroot, p) for p in image_paths]

        feats = compute_clip_embeddings(image_paths, clip_model, preprocess, device)
        if feats is None or feats.shape[0] < 2:
            print("Not enough valid embeddings to compute CLIP similarity.")
        else:
            feats = feats.to(device)
            with torch.no_grad():
                sim_matrix = feats @ feats.t()  # (N, N) cosine similarity

            N = sim_matrix.shape[0]
            idx = torch.triu_indices(N, N, offset=1)
            pair_sims = sim_matrix[idx[0], idx[1]]
            avg_sim = pair_sims.mean().item()
            std_sim = pair_sims.std().item()
            min_sim = pair_sims.min().item()
            max_sim = pair_sims.max().item()

            print(f"Average CLIP similarity after filtering (all pairs): {avg_sim:.4f}")
            print(f"Std CLIP similarity after filtering (all pairs): {std_sim:.4f}")
            print(f"Min CLIP similarity after filtering (all pairs): {min_sim:.4f}")
            print(f"Max CLIP similarity after filtering (all pairs): {max_sim:.4f}")

    return filtered_scenes, stats


############################################################
#                  CLI Interface
############################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to scene graph JSON")
    parser.add_argument("--output", required=True, help="Output filtered JSON file")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--dataroot", type=str, default=None, help="Dataset root for resolving relative image paths")

    args = parser.parse_args()

    if args.clip:
        try:
            import clip
            import torch
        except ImportError:
            print("CLIP library not found. Please install it to use --clip option.")
            exit(1)

    main(args.input, args.output, args.threshold, compute_clip_avg=args.clip, dataroot=args.dataroot)
