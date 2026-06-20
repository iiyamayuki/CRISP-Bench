import os
import re
from collections import defaultdict
from functools import partial
from typing import Any

import datasets
import numpy as np
import pandas as pd
from loguru import logger as eval_logger
from PIL import Image

from utils.paths import resolve_image_path

def doc_to_visual(doc: dict[str, Any]):
    p = resolve_image_path(doc)
    img = Image.open(p).convert("RGB")
    return [img]

def doc_to_visual_text_only(doc: dict[str, Any]):
    return []

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["conversations"][0]["value"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")

    if doc["meta"]["type"] == "NA":
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please respond with a single numeric value only. Do not include units, words, symbols, or explanations."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc["meta"]["type"] == "MCQ":
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['meta']['type']}")

def doc_to_target(doc: dict[str, Any]) -> str:
    return doc["conversations"][1]["value"]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS", None):
        eval_logger.info("Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset


def fuzzy_matching(pred):
    """
    Robustly extracts the predicted answer (A, B, C, D, etc.) from model output.
    """
    if not isinstance(pred, str):
        pred = str(pred)

    # Pre-clean: Remove common markdown formatting like **A** or `A`
    clean_pred = pred.replace('*', '').replace('`', '').strip()

    # For JSON-like outputs, try to extract the answer field first
    json_pattern = re.search(r'["\']answer["\']\s*[:=]\s*["\']([A-E])["\']', clean_pred, re.IGNORECASE)
    if json_pattern:
        return json_pattern.group(1).upper()

    # For textual outputs, try to find patterns like "The answer is A" or "Correct option: B"
    text_pattern = re.search(r'(?:answer|choice|option|correct)\s*(?:is|:)?\s*\(?([A-E])\)?\b', clean_pred, re.IGNORECASE)
    if text_pattern:
        return text_pattern.group(1).upper()

    # For outputs starting directly with the option letter
    start_pattern = re.search(r'^\s*\(?([A-E])\)?(?:[.)]|$)', clean_pred)
    if start_pattern:
        return start_pattern.group(1).upper()

    # For outputs that don't match above patterns, take the first token and clean it
    first_token = clean_pred.split(" ")[0]
    final_clean = first_token.strip('."\'()[]_,')

    return final_clean


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


METRICS_FOR_MCA = {
    "accuracy": exact_match,
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": partial(mean_relative_accuracy, start=.5, end=.95, interval=.05),
}


WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    "MRA:.5:.95:.05": 0.0,
}


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException:
        pred = None
    return pred


def process_results(doc, results):
    doc["prediction"] = results[0]
    if doc["meta"]["type"] == "MCQ":
        for key, metric_fn in METRICS_FOR_MCA.items():
            doc[key] = metric_fn(fuzzy_matching(doc["prediction"]), doc["conversations"][1]["value"])
    elif doc["meta"]["type"] == "NA":
        for key, metric_fn in METRICS_FOR_NA.items():
            try:
                doc[key] = metric_fn(to_float(fuzzy_matching(doc["prediction"])), to_float(doc["conversations"][1]["value"]))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['meta']['type']}")
    return {"vsibench_score": doc}


def aggregate_results(results):
    """
    Args:
        results: a list of dicts returned by process_results, each containing:
            - "meta": {"category": ..., "difficulty": ..., "type": ...}
            - metric fields corresponding to METRICS_FOR_MCA / METRICS_FOR_NA

    Returns:
        A dictionary containing:
            - metrics grouped by (type, category)
            - overall: mean over all (type, category) metric values
            - metrics grouped by (difficulty, type)
            - per-difficulty mean values (averaging over all types)
    """
    df = pd.DataFrame(results)

    if "meta" not in df.columns:
        raise ValueError("'meta' field is missing from results")

    # Expand metadata
    meta_df = df["meta"].apply(pd.Series)
    df = pd.concat([df.drop(columns=["meta"]), meta_df], axis=1)

    required_meta = {"category", "difficulty", "type"}
    if not required_meta.issubset(df.columns):
        raise ValueError(f"'meta' must contain keys: {required_meta}")

    output = {}

    # Group by (type, category)
    type_category_values = []  # used for computing the overall score

    for (q_type, category), group in df.groupby(["type", "category"]):
        prefix = f"{q_type}_{category}"

        if q_type == "MCQ":
            metrics = METRICS_FOR_MCA.keys()
        elif q_type == "NA":
            metrics = METRICS_FOR_NA.keys()
        else:
            raise ValueError(f"Unknown question type: {q_type}")

        for metric in metrics:
            if metric not in group.columns:
                raise ValueError(
                    f"Metric '{metric}' is missing for type={q_type}, category={category}"
                )
            value = group[metric].mean()
            output[f"{prefix}_{metric}"] = value
            type_category_values.append(value)

    # Compute overall score
    if len(type_category_values) == 0:
        raise ValueError("No type+category metric values collected; cannot compute overall score")

    output["overall"] = sum(type_category_values) / len(type_category_values)

    # Group by (difficulty, type)
    difficulty_to_values = defaultdict(list)

    for (difficulty, q_type), group in df.groupby(["difficulty", "type"]):
        prefix = f"{difficulty}_{q_type}"

        if q_type == "MCQ":
            metrics = METRICS_FOR_MCA.keys()
        elif q_type == "NA":
            metrics = METRICS_FOR_NA.keys()
        else:
            raise ValueError(f"Unknown question type: {q_type}")

        for metric in metrics:
            if metric not in group.columns:
                raise ValueError(
                    f"Metric '{metric}' is missing for difficulty={difficulty}, type={q_type}"
                )
            value = group[metric].mean()
            output[f"{prefix}_{metric}"] = value
            difficulty_to_values[difficulty].append(value)

    # Per-difficulty mean score
    for difficulty, vals in difficulty_to_values.items():
        if len(vals) > 0:
            output[f"difficulty_{difficulty}_mean"] = sum(vals) / len(vals)

    eval_logger.info(f"Evaluation results: {output}")
    return output
