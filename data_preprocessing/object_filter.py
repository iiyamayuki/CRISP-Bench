#!/usr/bin/env python3
"""
Filter a merged nuScenes-style annotations JSON with a two-stage pipeline:

Stage 1 (Diversity):
  - Filter based on object counts and Jaccard similarity.

Stage 2 (Balancing & Capping):
  - Logic A (Small Dataset): If valid samples <= 600, keep ALL Dense, Common, and Sparse (capped at 50% of Common).
  - Logic B (Large Dataset): If valid samples > 600, enforce strict cap of 600 images.
    - Ratio Sparse:Common:Dense = 1:2:2.
"""

import argparse
import json
import random
from collections import Counter, defaultdict

# ---- Buckets configuration ----
SPARSE_COUNTS  = [2, 3]
COMMON_COUNTS  = [4, 5, 6, 7, 8]
DENSE_COUNTS   = [9, 10, 11, 12]

ALL_COUNTS     = SPARSE_COUNTS + COMMON_COUNTS + DENSE_COUNTS

# ------------------ Utilities ------------------

def object_count(item: dict) -> int:
    """Return the number of objects in a scene item."""
    return len(item.get("objects", []) or [])

def build_instance_sets(data: list[dict]) -> list[set[str]]:
    """Precompute per-image sets of instance ids (node_id)."""
    inst_sets = []
    for item in data:
        s = set()
        for obj in (item.get("objects", []) or []):
            nid = obj.get("node_id")
            if nid:
                s.add(nid)
        inst_sets.append(s)
    return inst_sets

def jaccard(a: set[str], b: set[str]) -> float:
    """Compute the Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 1.0
    i = len(a & b)
    return i / float(u)

def build_pools(indices: list[int], data: list[dict]) -> tuple[dict[int, list[int]], dict[str, list[int]]]:
    """
    Build pools on a given subset of indices.
    """
    count_to_indices = defaultdict(list)
    for idx in indices:
        n = object_count(data[idx])
        count_to_indices[n].append(idx)

    bucket_to_indices = {
        "sparse" : [i for n in SPARSE_COUNTS for i in count_to_indices.get(n, [])],
        "common" : [i for n in COMMON_COUNTS for i in count_to_indices.get(n, [])],
        "dense"  : [i for n in DENSE_COUNTS  for i in count_to_indices.get(n, [])],
    }
    return count_to_indices, bucket_to_indices

def compute_targets_hybrid(bucket_to_indices: dict[str, list[int]], max_total_cap: int) -> tuple[int, dict[str, int]]:
    """
    Hybrid Logic based on dataset size:
    1. Calculate 'Natural Yield': All Dense + All Common + (0.5 * Common) Sparse.
    2. If Natural Yield <= max_total_cap (600):
       - Strategy: Maximize retention.
       - Dense: Keep All.
       - Common: Keep All.
       - Sparse: 50% of Common (1:2 ratio).
    3. If Natural Yield > max_total_cap:
       - Strategy: Cap and rebalance (High Dense).
       - Total: 600.
       - Ratio: Sparse:Common:Dense = 1:2:2.
    """
    avail = {k: len(v) for k, v in bucket_to_indices.items()}

    # --- Step 1: Calculate "Natural Yield" (Scenario A) ---
    # Logic: Keep all valid Dense/Common, restrict Sparse to avoid domination.
    s1_dense = avail["dense"]
    s1_common = avail["common"]
    s1_sparse_target = int(s1_common * 0.5)
    s1_sparse = min(avail["sparse"], s1_sparse_target)

    natural_total = s1_dense + s1_common + s1_sparse

    targets = {}

    if natural_total <= max_total_cap:
        print(f"[Logic] Natural yield ({natural_total}) <= {max_total_cap}. Using 'Max Retention' strategy.")
        targets["dense"] = s1_dense
        targets["common"] = s1_common
        targets["sparse"] = s1_sparse
    else:
        print(f"[Logic] Natural yield ({natural_total}) > {max_total_cap}. Using 'Capped 1:2:2' strategy.")
        # Ratio 1:2:2 -> Total 5 parts
        # 1 part = 600 / 5 = 120
        base_unit = max_total_cap // 5 # 120

        target_sparse = base_unit * 1  # 120
        target_common = base_unit * 2  # 240
        target_dense  = base_unit * 2  # 240

        # We must take min() because we can't invent data if it doesn't exist
        # (Though per your description, you likely have enough Dense now)
        targets["sparse"] = min(avail["sparse"], target_sparse)
        targets["common"] = min(avail["common"], target_common)
        targets["dense"]  = min(avail["dense"],  target_dense)

    total = sum(targets.values())
    return total, targets

def allocate_evenly_no_diversity(desired_total: int,
                                 counts: list[int],
                                 count_to_indices: dict[int, list[int]],
                                 rng: random.Random) -> list[int]:
    """
    Even allocation across the given counts WITHOUT diversity checks.
    """
    for n in counts:
        rng.shuffle(count_to_indices[n])

    k = len(counts)
    if k == 0 or desired_total <= 0:
        return []

    base = desired_total // k
    rem  = desired_total % k

    # Sort counts by availability (descending) to fill holes easier
    counts_by_avail = sorted(counts, key=lambda n: len(count_to_indices[n]), reverse=True)

    target_per_count = {n: base for n in counts}
    for i in range(rem):
        target_per_count[counts_by_avail[i % k]] += 1

    assigned = {n: min(target_per_count[n], len(count_to_indices[n])) for n in counts}

    # Redistribution logic for shortages
    total_assigned = sum(assigned.values())
    leftover = desired_total - total_assigned

    if leftover > 0:
        while leftover > 0:
            # Try to add to counts that still have capacity
            candidates = sorted(
                counts, key=lambda n: (len(count_to_indices[n]) - assigned[n]), reverse=True
            )
            progressed = False
            for n in candidates:
                if len(count_to_indices[n]) - assigned[n] > 0:
                    assigned[n] += 1
                    leftover -= 1
                    progressed = True
                    if leftover == 0:
                        break
            if not progressed:
                break

    selected = []
    for n in counts:
        selected.extend(count_to_indices[n][:assigned[n]])
    return selected

# ------------------ Main pipeline ------------------

def main():
    """CLI entry point for filtering and sampling scene images by object count."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to the input merged JSON')
    parser.add_argument('--output', required=True, help='Path to save the filtered JSON')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--diversity-threshold', type=float, default=0.3,
                        help='Jaccard similarity threshold in [0,1].')
    parser.add_argument('--image_cap', type=int, default=600,
                        help='Maximum total number of images to keep after filtering.')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    max_total_cap = args.image_cap

    # Load data
    with open(args.input, encoding='utf-8') as f:
        data = json.load(f)

    N = len(data)
    inst_sets = build_instance_sets(data)
    counts = [object_count(it) for it in data]

    # ------------------ Stage 0: Report initial availability ------------------
    initial_indices = [i for i, c in enumerate(counts) if c in ALL_COUNTS]
    _, initial_bucket_to_indices = build_pools(initial_indices, data)
    print("=== Pre-diversity availability (eligible counts only) ===")
    print(f"Available per bucket: sparse={len(initial_bucket_to_indices['sparse'])}, "
          f"common={len(initial_bucket_to_indices['common'])}, "
          f"dense={len(initial_bucket_to_indices['dense'])}")

    # ------------------ Stage 1: Diversity with lazy deletion ------------------
    deleted = [False] * N
    for i, c in enumerate(counts):
        if c not in ALL_COUNTS:
            deleted[i] = True  # ineligible

    # [CRITICAL UPDATE] Randomized Candidates Logic
    # 1. Get all valid indices.
    valid_indices = [i for i in range(N) if not deleted[i]]
    # 2. Shuffle them FIRST. This breaks the temporal link (index 0 is no more likely to be kept than index 100).
    rng.shuffle(valid_indices)
    # 3. Sort by count descending (we still prefer Dense images, but within the same count, the order is now random).
    candidates = sorted(valid_indices, key=lambda i: -counts[i])

    kept: list[int] = []
    diversity_rejects = 0

    for i in candidates:
        if deleted[i]: continue

        is_similar = False
        for j in kept:
            if jaccard(inst_sets[i], inst_sets[j]) >= args.diversity_threshold:
                deleted[i] = True
                is_similar = True
                diversity_rejects += 1
                break
        if is_similar:
            continue
        kept.append(i)

    surviving_indices = [i for i in range(N) if not deleted[i]]

    # ------------------ Stage 2: Balancing after diversity ------------------
    count_to_indices, bucket_to_indices = build_pools(surviving_indices, data)

    # Compute targets using the new "Hybrid" logic
    total_selected, targets = compute_targets_hybrid(bucket_to_indices, max_total_cap)

    # Allocate evenly within each bucket
    selected_sparse = allocate_evenly_no_diversity(targets.get("sparse", 0), SPARSE_COUNTS, count_to_indices, rng)
    selected_common = allocate_evenly_no_diversity(targets.get("common", 0), COMMON_COUNTS, count_to_indices, rng)
    selected_dense  = allocate_evenly_no_diversity(targets.get("dense", 0),  DENSE_COUNTS,  count_to_indices, rng)

    selected_indices = list(set(selected_sparse + selected_common + selected_dense))
    rng.shuffle(selected_indices)
    filtered = [data[i] for i in selected_indices]

    # ------------------ Reporting ------------------
    post_sparse = len(bucket_to_indices['sparse'])
    post_common = len(bucket_to_indices['common'])
    post_dense  = len(bucket_to_indices['dense'])

    per_count = Counter(object_count(data[i]) for i in selected_indices)
    per_bucket = {
        "sparse": sum(per_count[n] for n in SPARSE_COUNTS),
        "common": sum(per_count[n] for n in COMMON_COUNTS),
        "dense" : sum(per_count[n] for n in DENSE_COUNTS),
    }
    total_after = len(filtered)
    pct = {k: (per_bucket[k] / total_after if total_after > 0 else 0.0) for k in per_bucket}

    print("\n=== Diversity stage summary ===")
    print(f"Input images: {N}")
    print(f"Diversity threshold: {args.diversity_threshold}")
    print(f"Diversity rejections: {diversity_rejects}")
    print(f"Post-diversity availability: sparse={post_sparse}, common={post_common}, dense={post_dense}")

    print("\n=== Final selection summary (Hybrid Balancing) ===")
    print(f"Targets per bucket: sparse={targets.get('sparse',0)}, "
          f"common={targets.get('common',0)}, dense={targets.get('dense',0)}")
    print(f"Total selected: {total_after} (Max Cap: {max_total_cap})")
    print("Final proportions:")
    print(f"  sparse : {per_bucket['sparse']} ({pct['sparse']*100:.1f}%)")
    print(f"  common : {per_bucket['common']} ({pct['common']*100:.1f}%)")
    print(f"  dense  : {per_bucket['dense']}  ({pct['dense']*100:.1f}%)")
    print("\nPer-count distribution:")
    for n in sorted(per_count.keys()):
        print(f"  {n:>2d} -> {per_count[n]}")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"\nSaved filtered dataset to: {args.output}")

if __name__ == '__main__':
    main()
