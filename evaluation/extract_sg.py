import argparse
import json
import os
import re
import sys
from typing import Any

import json_repair


def _to_relative_image_path(path: str) -> str:
    """Best-effort conversion of an absolute image path to a relative one.

    The canonical relative form used throughout the project is
    ``annotated_image/images_with_bbox/<hash>.jpg`` (dataroot-relative).
    If the path is already relative it is returned unchanged.
    """
    if not os.path.isabs(path):
        return path
    # Try to find a known anchor directory in the path
    for anchor in ("annotated_image", "images"):
        idx = path.find(f"/{anchor}/")
        if idx != -1:
            return path[idx + 1:]  # strip leading '/'
    # Fallback: basename only
    return os.path.basename(path)

def parse_json_output(output_str: str) -> dict[str, Any] | None:
    """
    Robust JSON parser handling Markdown blocks, mixed text, and syntax errors.
    Prioritizes extraction -> standard parse -> repair parse.
    (Logic provided by user)
    """
    if not output_str:
        return None

    output_str = output_str.strip()

    # Step 1: Isolate the JSON string candidate
    candidate = output_str

    # Strategy 1: Extract from Markdown blocks (e.g., ```json ... ```)
    if "```" in output_str:
        pattern = r"```(?:json)?(.*?)```"
        match = re.search(pattern, output_str, re.DOTALL | re.IGNORECASE)
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

    # Attempt 2: Use json_repair library
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

def process_vlm_output(input_path: str, output_path: str):
    """Extract structured scene graphs from VLM text output in a JSONL log."""
    extracted_data = []
    success_count = 0
    total_count = 0

    print(f"Processing {input_path}...")

    with open(input_path, encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            total_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Error] Skipped invalid JSONL line at {line_idx + 1}", file=sys.stderr)
                continue

            doc_id = row.get("doc_id", f"line_{line_idx}")

            image_path = None
            if "sgc_score" in row and isinstance(row["sgc_score"], dict):
                image_path = row["sgc_score"].get("image")

            if not image_path:
                print(f"[Warning] No image path in 'sgc_score' for doc_id {doc_id}. Skipping...", file=sys.stderr)
                continue

            raw_response = ""
            if "filtered_resps" in row and isinstance(row["filtered_resps"], list) and row["filtered_resps"]:
                raw_response = str(row["filtered_resps"][0])
            else:
                print(f"[Warning] No filtered_resps found for doc_id {doc_id}", file=sys.stderr)
                continue

            sg_json = parse_json_output(raw_response)

            if sg_json and isinstance(sg_json, dict):
                objects = sg_json.get("objects", [])
                edges = sg_json.get("edges", [])

                if not isinstance(objects, list): objects = []
                if not isinstance(edges, list): edges = []

                final_obj = {
                    "doc_id": doc_id,
                    "image": _to_relative_image_path(image_path),
                    "objects": objects,
                    "edges": edges
                }

                extracted_data.append(final_obj)
                success_count += 1
            else:
                print(f"[Fail] Could not parse/repair JSON for doc_id {doc_id}", file=sys.stderr)

    print(f"Saving {len(extracted_data)} valid Scene Graphs to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(extracted_data, f_out, indent=2, ensure_ascii=False)

    print(f"Done. Success rate: {success_count}/{total_count} ({success_count/total_count*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and Repair Scene Graphs from VLM output JSONL.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output .json file")

    args = parser.parse_args()
    process_vlm_output(args.input, args.output)
