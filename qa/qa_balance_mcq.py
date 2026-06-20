#!/usr/bin/env python

import argparse
import json
from typing import Any

# ===========================================================
# Parsing helpers
# ===========================================================

def parse_item(item: dict[str, Any]) -> dict[str, Any]:
    """
    Parse a ShareGPT-style item into a more convenient internal structure.

    Returns a dict:
      {
        "raw": original_item_dict,
        "question_text": str,
        "options": List[str] or None,
        "answer_label": str or None,
        "num_options": int (0 for NAQ),
        "is_mcq": bool,
        "category": str
      }
    """
    convs = item.get("conversations", [])
    if len(convs) != 2:
        raise ValueError(f"Expected exactly 2 conversation turns, got {len(convs)}")

    human = convs[0]["value"]
    gpt = convs[1]["value"].strip()

    # Split question and options
    question_text = human
    options: list[str] | None = None

    if "\nOptions:\n" in human:
        question_text, opt_block = human.split("\nOptions:\n", 1)
        opt_lines = [line for line in opt_block.split("\n") if line.strip()]
        if opt_lines:
            options = opt_lines

    is_mcq = options is not None and len(gpt) == 1 and gpt in "ABCD"
    num_options = len(options) if options is not None else 0
    answer_label = gpt if is_mcq else None

    meta = item.get("meta", {})
    category = meta.get("category", "unknown")

    return {
        "raw": item,
        "question_text": question_text,
        "options": options,
        "answer_label": answer_label,
        "num_options": num_options,
        "is_mcq": is_mcq,
        "category": category,
    }


# ===========================================================
# Stats (same as in qa_stats)
# ===========================================================

def compute_stats(parsed_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute MCQ/NAQ distribution statistics over parsed QA items."""
    total = len(parsed_items)
    mcq_count = 0
    na_count = 0

    counts2 = {"A": 0, "B": 0}
    counts4 = {"A": 0, "B": 0, "C": 0, "D": 0}
    category_counts: dict[str, int] = {}

    for it in parsed_items:
        cat = it["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

        if it["is_mcq"]:
            mcq_count += 1
            nopt = it["num_options"]
            label = it["answer_label"]
            if nopt == 2 and label in counts2:
                counts2[label] += 1
            elif nopt == 4 and label in counts4:
                counts4[label] += 1
        else:
            na_count += 1

    def ratio(x: int, base: int) -> float:
        return float(x) / base if base > 0 else 0.0

    stats = {
        "total_questions": total,
        "mcq_count": mcq_count,
        "na_count": na_count,
        "mcq_ratio": ratio(mcq_count, total),
        "na_ratio": ratio(na_count, total),
        "two_option_counts": counts2,
        "four_option_counts": counts4,
        "category_counts": category_counts,
    }
    return stats


def print_stats(stats: dict[str, Any]) -> None:
    """Print a human-readable summary of QA dataset statistics."""
    print("===== Dataset Statistics =====")
    total = stats["total_questions"]
    print(f"Total questions: {total}")
    print(f"MCQ: {stats['mcq_count']} ({stats['mcq_ratio']:.3f})")
    print(f"NAQ: {stats['na_count']} ({stats['na_ratio']:.3f})")
    print()

    c2 = stats["two_option_counts"]
    total2 = c2["A"] + c2["B"]
    if total2 > 0:
        print("2-option MCQ label distribution (A/B):")
        for label in ["A", "B"]:
            cnt = c2[label]
            ratio = cnt / total2 if total2 > 0 else 0.0
            print(f"  {label}: {cnt} ({ratio:.3f})")
    else:
        print("No 2-option MCQ questions.")
    print()

    c4 = stats["four_option_counts"]
    total4 = sum(c4.values())
    if total4 > 0:
        print("4-option MCQ label distribution (A/B/C/D):")
        for label in ["A", "B", "C", "D"]:
            cnt = c4[label]
            ratio = cnt / total4 if total4 > 0 else 0.0
            print(f"  {label}: {cnt} ({ratio:.3f})")
    else:
        print("No 4-option MCQ questions.")
    print()

    cat_counts = stats["category_counts"]
    if cat_counts:
        print("Category distribution:")
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[0]):
            ratio = cnt / total if total > 0 else 0.0
            print(f"  {cat}: {cnt} ({ratio:.3f})")
    else:
        print("No category info found.")
    print("==============================")


# ===========================================================
# Balancing helpers
# ===========================================================

def balance_two_option_mcq(
    items: list[dict[str, Any]],
    tolerance_ratio: float = 0.05,
    max_iterations: int = 10000,
) -> None:
    """
    Balance 2-option MCQ answers so that:

        max_count - min_count <= tolerance_ratio * total_2

    by swapping options and flipping labels for some questions.
    """
    indices_2 = [
        i for i, it in enumerate(items)
        if it["is_mcq"] and it["num_options"] == 2
    ]
    if not indices_2:
        return

    counts = {"A": 0, "B": 0}
    for i in indices_2:
        label = items[i]["answer_label"]
        if label in counts:
            counts[label] += 1
    total2 = counts["A"] + counts["B"]
    if total2 == 0:
        return

    def get_max_min(c: dict[str, int]):
        if c["A"] >= c["B"]:
            return "A", "B"
        else:
            return "B", "A"

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        max_label, min_label = get_max_min(counts)
        diff = counts[max_label] - counts[min_label]
        if diff <= tolerance_ratio * total2:
            break

        # find a question with label = max_label and swap its options
        flipped = False
        for idx in indices_2:
            it = items[idx]
            if it["answer_label"] != max_label:
                continue
            opts = it["options"]
            if not opts or len(opts) != 2:
                continue
            # swap options[0] and options[1]
            opts[0], opts[1] = opts[1], opts[0]
            # flip label
            it["answer_label"] = "B" if max_label == "A" else "A"
            counts[max_label] -= 1
            counts[min_label] += 1
            flipped = True
            break

        if not flipped:
            break


def balance_four_option_mcq(
    items: list[dict[str, Any]],
    tolerance_ratio: float = 0.05,
    max_iterations: int = 20000,
) -> None:
    """
    Balance 4-option MCQ answers so that:

        max_count - min_count <= tolerance_ratio * total_4

    by swapping options indices and moving answer labels.
    """
    indices_4 = [
        i for i, it in enumerate(items)
        if it["is_mcq"] and it["num_options"] == 4
    ]
    if not indices_4:
        return

    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for i in indices_4:
        label = items[i]["answer_label"]
        if label in counts:
            counts[label] += 1
    total4 = sum(counts.values())
    if total4 == 0:
        return

    def get_max_min_labels(c: dict[str, int]):
        labels = sorted(c.keys())
        max_label = max(labels, key=lambda x: c[x])
        min_label = min(labels, key=lambda x: c[x])
        return max_label, min_label

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        max_label, min_label = get_max_min_labels(counts)
        diff = counts[max_label] - counts[min_label]
        if diff <= tolerance_ratio * total4:
            break

        # find an item with label = max_label and move it to min_label
        moved = False
        for idx in indices_4:
            it = items[idx]
            if it["answer_label"] != max_label:
                continue
            opts = it["options"]
            if not opts or len(opts) != 4:
                continue

            idx_max = ord(max_label) - ord("A")
            idx_min = ord(min_label) - ord("A")

            # swap options at idx_max and idx_min
            opts[idx_max], opts[idx_min] = opts[idx_min], opts[idx_max]
            # update label
            it["answer_label"] = min_label

            counts[max_label] -= 1
            counts[min_label] += 1
            moved = True
            break

        if not moved:
            break


# ===========================================================
# Reconstruction
# ===========================================================

def apply_changes_back(parsed_items: list[dict[str, Any]]) -> None:
    """
    After balancing (options + answer_label updated in parsed_items),
    write the changes back into the raw item structure.

    - For MCQ:
        conversations[0]['value'] = question_text + "\\nOptions:\\n" + "\\n".join(options)
        conversations[1]['value'] = answer_label
    - For NAQ:
        leave raw untouched.
    """
    for it in parsed_items:
        raw = it["raw"]
        if not it["is_mcq"]:
            continue

        question_text = it["question_text"]
        options = it["options"]
        answer_label = it["answer_label"]

        # Rebuild human text
        if options:
            human_value = question_text + "\nOptions:\n" + "\n".join(options)
        else:
            human_value = question_text

        raw_convs = raw.get("conversations", [])
        if len(raw_convs) != 2:
            continue

        raw_convs[0]["value"] = human_value
        raw_convs[1]["value"] = answer_label


# ===========================================================
# Main
# ===========================================================

def balance_and_save(
    input_path: str,
    output_path: str,
    tolerance_ratio: float = 0.05,
) -> None:
    """Load a QA JSONL, balance MCQ answer distributions, and save the result."""
    parsed_items: list[dict[str, Any]] = []

    # Load
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            parsed = parse_item(obj)
            parsed_items.append(parsed)

    # Balance
    balance_two_option_mcq(parsed_items, tolerance_ratio=tolerance_ratio)
    balance_four_option_mcq(parsed_items, tolerance_ratio=tolerance_ratio)

    # Apply changes to raw
    apply_changes_back(parsed_items)

    # Stats after balancing
    stats = compute_stats(parsed_items)
    print("After balancing:")
    print_stats(stats)

    # Save
    with open(output_path, "w", encoding="utf-8") as out_f:
        for it in parsed_items:
            out_f.write(json.dumps(it["raw"], ensure_ascii=False) + "\n")


def parse_args():
    """Parse command-line arguments for QA balancing."""
    ap = argparse.ArgumentParser(
        description="Balance MCQ answer distributions and save a new ShareGPT dataset."
    )
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input ShareGPT JSONL file.",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output balanced ShareGPT JSONL file.",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Allowed difference between max and min label frequencies (as a ratio).",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    balance_and_save(
        input_path=args.input,
        output_path=args.output,
        tolerance_ratio=args.tolerance,
    )
