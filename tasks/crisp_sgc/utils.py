import json
import os
import re
from typing import Any

import datasets
import json_repair
import pandas as pd
from loguru import logger as eval_logger
from PIL import Image

from utils.paths import resolve_image_path

# ========================================
# 1. Data Loading & Formatting
# ========================================

def doc_to_visual(doc: dict[str, Any]):
    p = resolve_image_path(doc)
    img = Image.open(p).convert("RGB")
    return [img]

def doc_to_visual_text_only(doc: dict[str, Any]):
    return []

def doc_to_text(doc):
    # Extract the human instruction
    for turn in doc["conversations"]:
        if turn["from"] == "human":
            return turn["value"]
    return ""

def doc_to_target(doc: dict[str, Any]) -> str:
    # Extract the ground truth JSON string
    for turn in doc["conversations"]:
        if turn["from"] == "gpt":
            return turn["value"]
    return ""

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        return dataset.shuffle(seed=42)
    return dataset

# ========================================
# 2. Metric Implementation
# ========================================

def parse_json_output(output_str):
    """
    Robust JSON parser handling Markdown blocks, mixed text, and syntax errors.
    Prioritizes extraction -> standard parse -> repair parse.
    """
    if not output_str:
        return None

    output_str = output_str.strip()

    # Step 1: Isolate the JSON string candidate
    candidate = output_str

    # Strategy 1: Extract from Markdown blocks (e.g., ```json ... ```)
    if "```" in output_str:
        pattern = r"```(?:json)?(.*?)```"
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            candidate = match.group(1).strip()

    # Strategy 2: If no Markdown, attempt to find the outermost braces
    # This helps when the model outputs "Here is the JSON: {...}"
    else:
        start_idx = output_str.find("{")
        end_idx = output_str.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = output_str[start_idx : end_idx + 1]

    # Step 2: Attempt Parsing (Strict -> Loose)
    # Attempt 1: Standard strict JSON parsing (Fastest & most accurate if valid)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Use json_repair library (if available)
    # This fixes missing quotes, trailing commas, unclosed brackets, etc.
    try:
        return json_repair.loads(candidate)
    except Exception:
        # Attempt 3: Desperate fallback
        # Sometimes extracting the substring breaks things (if braces were malformed).
        # Try repairing the ORIGINAL full string.
        try:
            return json_repair.loads(output_str)
        except Exception:
            pass

    return None

def normalize_relations(rel_list: list[str]) -> dict[str, int]:
    mapping = {
        # Standard
        "left": "left",
        "right": "right",
        "in front of": "in front of",
        "behind": "behind",
        "up": "up",
        "down": "down",

        # Common Hallucinations / Synonyms
        "front": "in front of",
        "in front": "in front of",
        "ahead": "in front of",
        "back": "behind",
        "to the left": "left",
        "to the right": "right",
        "on the left": "left",
        "on the right": "right",
        "above": "up",
        "below": "down",
        "under": "down"
    }

    standard_keys = ["left", "right", "in front of", "behind", "up", "down"]
    result = {k: 0 for k in standard_keys}

    if isinstance(rel_list, list):
        for raw_rel in rel_list:
            if not isinstance(raw_rel, str): continue

            clean_rel = raw_rel.lower().strip()

            if clean_rel in mapping:
                standard_key = mapping[clean_rel]
                result[standard_key] = 1
            else:
                pass

    return result

def calculate_distance_score(pred_val, gt_val):
    """
    Computes score based on relative error: max(0, 1 - |p - g| / g).
    Used for both 'dist_to_cam' (Node) and 'distance' (Edge).
    """
    try:
        p = float(pred_val)
        g = float(gt_val)
    except (TypeError, ValueError):
        return 0.0

    epsilon = 1e-6
    # If GT is 0 (e.g. ego center), relative error is undefined.
    # Handle strictly: if pred is also near 0, score 1, else 0.
    if g < epsilon:
        return 1.0 if p < epsilon else 0.0

    return max(0.0, 1.0 - abs(p - g) / (g + epsilon))

def calculate_node_score_single(pred_obj, gt_obj):
    """
    Computes S_node as the average of:
      1. Size Score (Min/Max Ratio) - Attribute consistency
      2. Dist_to_Cam Score (Rel Error) - Localization consistency
    """
    scores = {"size": 0.0, "dist_to_cam": 0.0}
    # --- Part A: Size Score (Symmetric Min/Max) ---
    pred_size = pred_obj.get("size", {})
    gt_size = gt_obj.get("size", {})

    dims = ['w', 'l', 'h']
    size_ratios = []
    for d in dims:
        p_val = float(pred_size.get(d, 0.0))
        g_val = float(gt_size.get(d, 0.0))
        if p_val <= 0 or g_val <= 0:
            size_ratios.append(0.0)
        else:
            size_ratios.append(min(p_val, g_val) / max(p_val, g_val))

    scores["size"] = sum(size_ratios) / 3.0

    # --- Part B: Distance to Camera Score (Bounded Rel Error) ---
    # We use the same strict logic as edges
    scores["dist_to_cam"] = calculate_distance_score(
        pred_obj.get("dist_to_cam", 0.0),
        gt_obj.get("dist_to_cam", 0.0)
    )

    return scores

def calculate_dist_scores_single(pred_edge, gt_edge):
    """
    Computes S_dist for a single edge.
    """
    scores = {"dist": 0.0}

    # --- Distance Score: Use helper ---
    if "distance" in gt_edge and "distance" in pred_edge:
        scores["dist"] = calculate_distance_score(pred_edge["distance"], gt_edge["distance"])

    return scores

def calculate_relation_score_single(pred_rel_raw, gt_rel):
    if isinstance(pred_rel_raw, list):
        pred_rel_dict = normalize_relations(pred_rel_raw)
    elif isinstance(pred_rel_raw, dict):
        pred_rel_dict = pred_rel_raw
    else:
        pred_rel_dict = normalize_relations([])

    if isinstance(gt_rel, list):
        gt_rel_dict = normalize_relations(gt_rel)
    else:
        gt_rel_dict = gt_rel

    if not gt_rel_dict: return 0.0

    pairs = [("left", "right"), ("in front of", "behind"), ("up", "down")]
    correct = 0
    total_pairs = 3.0

    for k1, k2 in pairs:
        g1, g2 = gt_rel_dict.get(k1, 0), gt_rel_dict.get(k2, 0)
        p1, p2 = pred_rel_dict.get(k1, 0), pred_rel_dict.get(k2, 0)

        if p1 == g1 and p2 == g2:
            correct += 1

    return correct / total_pairs

def compute_metrics_per_sample(pred_json, gt_json):
    """
    Calculates S_node (avg of Size & DistCam), S_dist, S_rel, and strict json_valid.
    """
    # Define all metric keys for initialization
    metric_keys = ["S_size", "S_dist_cam", "S_dist", "S_est", "S_rel", "SGC_Score", "json_valid"]

    if pred_json is None or not isinstance(pred_json, dict):
        eval_logger.warning(f"Invalid pred_json type: {type(pred_json)}. Content: {pred_json}")
        return {k: 0.0 for k in metric_keys}

    # 1. Node Metrics
    # Map objects by id for matching.
    pred_objs = {str(obj.get("id", "")): obj for obj in pred_json.get("objects", [])}
    gt_objs = {str(obj.get("id", "")): obj for obj in gt_json.get("objects", [])}

    size_scores = []
    dist_cam_scores = []
    missing_nodes_count = 0

    for node_id, gt_obj in gt_objs.items():
        if node_id in pred_objs:
            res = calculate_node_score_single(pred_objs[node_id], gt_obj)
            size_scores.append(res["size"])
            dist_cam_scores.append(res["dist_to_cam"])
        else:
            # Missing node: penalty is 0.0
            size_scores.append(0.0)
            dist_cam_scores.append(0.0)
            missing_nodes_count += 1

    # Compute averages for sub-metrics
    S_size = sum(size_scores) / len(size_scores) if size_scores else 0.0
    S_dist_cam = sum(dist_cam_scores) / len(dist_cam_scores) if dist_cam_scores else 0.0

    # 2. Edge Metrics
    pred_edges = {edge.get("to", ""): edge for edge in pred_json.get("edges", [])}
    gt_edges = {edge.get("to", ""): edge for edge in gt_json.get("edges", [])}

    dist_scores, rel_scores = [], []
    missing_edges_count = 0

    for target_cap, gt_edge in gt_edges.items():
        if target_cap in pred_edges:
            p_edge = pred_edges[target_cap]

            # Geometric scores (Distance & Angle)
            dist_scores.append(calculate_dist_scores_single(p_edge, gt_edge)["dist"])

            # Relation score
            s_rel = calculate_relation_score_single(p_edge.get("relation", []), gt_edge.get("relation", []))
            rel_scores.append(s_rel)
        else:
            # Missing edge: penalty is 0.0
            dist_scores.append(0.0)
            rel_scores.append(0.0)
            missing_edges_count += 1

    S_dist = sum(dist_scores) / len(dist_scores) if dist_scores else 0.0
    S_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0.0
    S_est = (S_size + S_dist_cam + S_dist) / 3

    # 3. Final Aggregation
    SGC_Score = (S_est + S_rel) / 2

    # Strict JSON Validity: Valid only if parsed AND no missing nodes/edges
    is_valid = 1.0 if (missing_nodes_count == 0 and missing_edges_count == 0) else 0.0

    return {
        "S_size": S_size,
        "S_dist_cam": S_dist_cam,
        "S_dist": S_dist,
        "S_est": S_est,
        "S_rel": S_rel,
        "SGC_Score": SGC_Score,
        "json_valid": is_valid
    }

# ========================================
# 3. Process & Aggregate
# ========================================

def process_results(doc, results):
    """
    Parses LLM output, compares with target, and computes sample-level metrics.
    """
    pred_str = results[0]
    target_str = doc_to_target(doc)

    pred_json = parse_json_output(pred_str)
    target_json = json.loads(target_str)

    scores = compute_metrics_per_sample(pred_json, target_json)

    doc["prediction_json"] = pred_json
    doc["scores"] = scores

    # Return dict for aggregation
    return {"sgc_score": doc}

def aggregate_results(results):
    """
    Aggregates metrics across the entire dataset.
    """
    score_list = [res["scores"] for res in results]
    df = pd.DataFrame(score_list)

    output = {}
    metrics_to_report = ["S_size", "S_dist_cam", "S_dist", "S_est", "S_rel", "SGC_Score", "json_valid"]

    for metric in metrics_to_report:
        if metric in df.columns:
            output[metric] = df[metric].mean()

    eval_logger.info(f"Final Scene Graph Metrics: {output}")
    return output
