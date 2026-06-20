import argparse
import json
import re

import numpy as np
import pandas as pd

# ==============================================================================
# 1. Helper Functions (Strictly following utils.py logic)
# ==============================================================================

def fuzzy_matching(pred):
    """
    Robustly extracts the predicted answer (MCQ or NAQ) from model output.
    Prioritizes explicit <answer> tags and falls back to reverse token search for CoT.
    """
    if not isinstance(pred, str):
        pred = str(pred)

    # Priority 1: CoT <answer> tag extraction
    # Uses DOTALL to handle potential line breaks inside the tag
    cot_pattern = re.search(r'<answer>(.*?)</answer>', pred, re.IGNORECASE | re.DOTALL)
    if cot_pattern:
        extracted = cot_pattern.group(1).strip()
        # Clean up any residual markdown or punctuation inside the tag
        return extracted.replace('*', '').replace('`', '').strip('."\'()[]_,')

    # Pre-clean: Remove common markdown formatting for fallback parsing
    clean_pred = pred.replace('*', '').replace('`', '').strip()

    # Define regex for valid answers: A-E for MCQ, or numbers/floats for NAQ
    # Example matches: "A", "C", "42", "-1.5", "0.75"
    valid_ans_regex = r'([A-E]|-?\d+(?:\.\d+)?)'

    # Priority 2: JSON-like outputs
    json_pattern = re.search(r'["\']answer["\']\s*[:=]\s*["\']?' + valid_ans_regex + r'["\']?', clean_pred, re.IGNORECASE)
    if json_pattern:
        return json_pattern.group(1).upper()

    # Priority 3: Textual patterns (e.g., "The answer is A" or "value is 1.5")
    text_pattern = re.search(r'(?:answer|choice|option|correct|value)\s*(?:is|:)?\s*\(?' + valid_ans_regex + r'\)?\b', clean_pred, re.IGNORECASE)
    if text_pattern:
        return text_pattern.group(1).upper()

    # Priority 4: Outputs starting directly with the option letter or number
    start_pattern = re.search(r'^\s*\(?' + valid_ans_regex + r'\)?(?:[.)\s]|$)', clean_pred, re.IGNORECASE)
    if start_pattern:
        return start_pattern.group(1).upper()

    # Fallback for missing CoT tags: Search backwards
    # Rationale: In zero-shot CoT, the final conclusion is usually at the end.
    tokens = clean_pred.split()
    for token in reversed(tokens):
        clean_token = token.strip('."\'()[]_,:')

        # Check if the token is exactly A-E
        if re.fullmatch(r'[A-E]', clean_token, re.IGNORECASE):
            return clean_token.upper()

        # Check if the token is a valid number
        if re.fullmatch(r'-?\d+(?:\.\d+)?', clean_token):
            return clean_token

    # Absolute fallback (Legacy behavior): Return the first cleaned token
    if tokens:
        return tokens[0].strip('."\'()[]_,')

    return ""
def to_float(pred):
    """Attempt to convert a prediction to float, returning None on failure."""
    try:
        return float(pred)
    except (ValueError, TypeError):
        return None

def exact_match(pred, target):
    """Return 1.0 if pred and target match (case-insensitive), else 0.0."""
    return 1.0 if str(pred).lower() == str(target).lower() else 0.0

def abs_dist_norm(pred, target):
    """Return the absolute relative error between pred and target."""
    if target == 0:
        return 0.0 if pred == 0 else 1.0
    return abs(pred - target) / abs(target)

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    """
    Computes MRA. Checks if relative error is within confidence intervals.
    """
    if pred is None or target is None:
        return 0.0

    # Logic from utils.py
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    rel_error = abs_dist_norm(pred, target)
    accuracy_bools = rel_error <= (1 - conf_intervs)
    return accuracy_bools.mean()

# ==============================================================================
# 2. Evaluation Logic
# ==============================================================================

class ConsistencyEvaluator:
    def __init__(self, debug_mode: bool = False):
        """Initialize evaluator with optional debug logging."""
        self.results_accuracy = []    # Derived vs GT
        self.results_consistency = [] # Derived vs Direct
        self.debug_mode = debug_mode
        self.mismatches = []  # Store mismatch details for debugging

    def evaluate_item(self, item_id: str, meta: dict, gt: str, direct: str, derived: str,
                     question: str = None, options: list[str] = None):
        """Score a single QA item for accuracy (derived vs GT) and consistency (derived vs direct)."""
        q_type = meta.get("type", "unknown")
        category = meta.get("category", "unknown")
        difficulty = meta.get("difficulty", "unknown")

        # 1. Handle Solver Failures
        # If solver returned "FAILED...", valid=False
        derived_is_valid = not str(derived).startswith("FAILED")

        # --- Metric A: Accuracy (Derived vs GT) ---
        score_acc = 0.0
        parsed_derived = None
        parsed_gt = None

        if derived_is_valid:
            if q_type == "MCQ":
                parsed_derived = fuzzy_matching(derived)
                parsed_gt = gt
                score_acc = exact_match(parsed_derived, parsed_gt)
            elif q_type in {"NA", "NAQ"}:
                parsed_derived = to_float(fuzzy_matching(derived))
                parsed_gt = to_float(gt)
                score_acc = mean_relative_accuracy(parsed_derived, parsed_gt)

        # Debug Mode: Record mismatches
        if self.debug_mode and score_acc < 1.0:
            mismatch_info = {
                "id": item_id,
                "type": q_type,
                "category": category,
                "difficulty": difficulty,
                "question": question,
                "options": options,
                "gt": gt,
                "derived": derived,
                "parsed_gt": parsed_gt,
                "parsed_derived": parsed_derived,
                "score": score_acc,
                "valid": derived_is_valid
            }
            self.mismatches.append(mismatch_info)

        self.results_accuracy.append({
            "id": item_id,
            "type": q_type,
            "category": category,
            "difficulty": difficulty,
            "score": score_acc,
            "valid": derived_is_valid
        })

        # --- Metric B: Consistency (Derived vs Direct) ---
        score_cons = 0.0
        # Only calc consistency if solver succeeded AND direct answer exists
        if derived_is_valid and direct is not None:
            if q_type == "MCQ":
                # Direct answer might be "A" or "A. Left". Fuzzy match extracts "A".
                score_cons = exact_match(fuzzy_matching(derived), fuzzy_matching(direct))
            elif q_type in {"NA", "NAQ"}:
                score_cons = mean_relative_accuracy(
                    to_float(fuzzy_matching(derived)),
                    to_float(fuzzy_matching(direct))
                )

        self.results_consistency.append({
            "id": item_id,
            "type": q_type,
            "category": category,
            "difficulty": difficulty,
            "score": score_cons
        })

    def print_mismatches(self):
        """Print detailed mismatch information in debug mode."""
        if not self.debug_mode or not self.mismatches:
            return

        print(f"\n{'='*80}")
        print(f"DEBUG: Found {len(self.mismatches)} Mismatches (Derived vs GT)")
        print(f"{'='*80}\n")

        # Group by category
        by_category = {}
        for m in self.mismatches:
            cat = m['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(m)

        for category, items in sorted(by_category.items()):
            print(f"\n{'─'*80}")
            print(f"Category: {category.upper()} ({len(items)} mismatches)")
            print(f"{'─'*80}")

            for i, m in enumerate(items[:10], 1):  # Show first 10 per category
                print(f"\n[{i}] ID: {m['id']}")
                print(f"    Type: {m['type']} | Difficulty: {m['difficulty']}")

                if m['question']:
                    q_preview = m['question'][:120] + "..." if len(m['question']) > 120 else m['question']
                    print(f"    Question: {q_preview}")

                if m['options']:
                    print(f"    Options: {m['options']}")

                if m['type'] == "MCQ":
                    print(f"    GT Answer:      {m['gt']}")
                    print(f"    Derived Answer: {m['parsed_derived']}")
                    print(f"    Raw Derived:    {m['derived'][:80]}...")
                    print(f"    Match: {'✓' if m['score'] == 1.0 else '✗'}")
                else:  # NAQ
                    print(f"    GT Value:      {m['parsed_gt']}")
                    print(f"    Derived Value: {m['parsed_derived']}")
                    print(f"    Raw Derived:   {m['derived'][:80]}...")
                    print(f"    MRA Score: {m['score']:.4f}")
                    if m['parsed_gt'] and m['parsed_derived']:
                        rel_error = abs_dist_norm(m['parsed_derived'], m['parsed_gt'])
                        print(f"    Relative Error: {rel_error:.2%}")

                if not m['valid']:
                    print(f"    ⚠️  Solver Failed: {m['derived']}")

            if len(items) > 10:
                print(f"\n    ... and {len(items) - 10} more mismatches in this category")

    def aggregate_and_print(self, results_list: list[dict], title: str):
        """Aggregate per-item scores, print summary, and return statistics dict."""
        df = pd.DataFrame(results_list)
        if df.empty:
            print(f"\n[{title}] No results.")
            return None

        print(f"\n{'='*20} {title} {'='*20}")

        type_category_means = (
            df.groupby(["type", "category"])["score"]
            .mean()
            .reset_index()
            .rename(columns={"score": "score_mean"})
        )

        overall_score = type_category_means["score_mean"].mean()
        print(f"Overall Score: {overall_score:.4f}")

        print("\n--- By Category ---")
        print(type_category_means.to_string(index=False))

        print("\n--- By Difficulty ---")
        diff_means = (
            df.groupby("difficulty")["score"]
            .mean()
            .reset_index()
        )
        print(diff_means.to_string(index=False))

        if "valid" in df.columns:
            valid_rate = df['valid'].mean()
            print(f"\nSolver Valid Rate: {valid_rate:.2%}")
        else:
            valid_rate = None

        return {
            "overall_score": float(overall_score),
            "by_category": type_category_means.to_dict('records'),
            "by_difficulty": diff_means.to_dict('records'),
            "valid_rate": float(valid_rate) if valid_rate is not None else None
        }

    def save_results(self, output_path: str, accuracy_summary: dict, consistency_summary: dict):
        """Save accuracy and consistency results to a JSON file."""
        results = {
            "accuracy": accuracy_summary,
            "consistency": consistency_summary,
            "detailed_accuracy": self.results_accuracy,
            "detailed_consistency": self.results_consistency
        }

        if self.debug_mode:
            results["debug_mismatches"] = self.mismatches

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_path}")

# ==============================================================================
# 3. Main Script
# ==============================================================================

def load_direct_qa_map(path: str) -> dict[str, dict]:
    """
    Loads the Direct QA output JSONL.
    PRIORITY: vsibench_score['id'] -> root 'id' -> root 'question_id'
    """
    data_map = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)

            # Key Logic: Look inside vsibench_score first
            key = None
            if "vsibench_score" in item and "id" in item["vsibench_score"]:
                key = item["vsibench_score"]["id"]
            else:
                # Fallback keys
                key = item.get("id") or item.get("question_id") or str(item.get("doc_id"))

            if key:
                data_map[key] = item
    return data_map

def main():
    """CLI entry point for calculating consistency between derived and direct QA answers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-file", required=True, help="Output from VLM Direct QA")
    parser.add_argument("--output", default="consistency_results.json", help="Output file for results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print mismatches")
    args = parser.parse_args()

    # 1. Load Data
    print("Loading Derived Answers...")
    derived_data = []
    with open(args.derived_file, encoding='utf-8') as f:
        for line in f:
            if line.strip(): derived_data.append(json.loads(line))

    print("Loading Direct Answers...")
    direct_map = load_direct_qa_map(args.direct_file)

    evaluator = ConsistencyEvaluator(debug_mode=args.debug)
    print(f"Evaluating {len(derived_data)} items...")

    missing_direct = 0

    for item in derived_data:
        # Derived file structure (ShareGPT-like)
        item_id = item.get("id")
        meta = item.get("meta", {})

        question_text = None
        options_list = None
        for t in item.get("conversations", []):
            if t["from"] == "human":
                question_text = t["value"]
                # Try to extract options from question
                options_match = re.search(r'Options:\s*\n(.+)', question_text, re.DOTALL)
                if options_match:
                    options_text = options_match.group(1)
                    options_list = [line.strip() for line in options_text.split('\n') if line.strip()]
                break

        # Get GT from conversation history
        gt_val = None
        for t in item.get("conversations", []):
            if t["from"] == "gpt":
                gt_val = t["value"]
                break

        # Get Derived Prediction
        derived_val = item.get("predict") # Must exist from solver

        # Get Direct Prediction
        direct_row = direct_map.get(item_id)
        direct_val = None
        if direct_row:
            # Logic: filtered_resps[0] > prediction > text
            if "filtered_resps" in direct_row and direct_row["filtered_resps"]:
                direct_val = direct_row["filtered_resps"][0]
            elif "prediction" in direct_row:
                direct_val = direct_row["prediction"]
        else:
            missing_direct += 1

        if gt_val is None or derived_val is None:
            continue

        evaluator.evaluate_item(
            item_id=item_id,
            meta=meta,
            gt=gt_val,
            direct=direct_val,
            derived=derived_val,
            question=question_text,
            options=options_list
        )

    if missing_direct > 0:
        print(f"Warning: {missing_direct} items missing in Direct QA file (ID mismatch?).")

    accuracy_summary = evaluator.aggregate_and_print(evaluator.results_accuracy, "1. Derived vs GT (Accuracy)")
    consistency_summary = evaluator.aggregate_and_print(evaluator.results_consistency, "2. Derived vs Direct (Consistency)")

    if args.debug:
        evaluator.print_mismatches()
        evaluator.save_results(args.output, accuracy_summary, {})  # debug mode only for accuracy
    else:
        evaluator.save_results(args.output, accuracy_summary, consistency_summary)

if __name__ == "__main__":
    main()
