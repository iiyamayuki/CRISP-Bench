import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


class IDTracker:
    """
    Helper class to track ID uniqueness across multiple files.
    """
    def __init__(self):
        # Stores all seen IDs to ensure global uniqueness
        self.seen_ids = set()
        # Stores collision details: {duplicated_id: [file_where_it_occurred, ...]}
        self.collisions = defaultdict(list)
        self.total_collisions = 0

    def check_and_record(self, uid, current_filename):
        """
        Checks if the ID has been seen. Records collision if true.
        Returns True if collision detected, False otherwise.
        """
        # Critical check: Ensure ID is valid (not None or empty)
        if not uid:
            return False # Skip empty IDs, handled by other logic if needed

        if uid in self.seen_ids:
            self.collisions[uid].append(current_filename)
            self.total_collisions += 1
            return True
        else:
            self.seen_ids.add(uid)
            return False

    def print_report(self):
        """
        Prints a summary of ID collisions.
        """
        if self.total_collisions == 0:
            print("\n[ID Check] No ID collisions detected. Good to go.")
        else:
            print(f"\n[ID Check] CRITICAL WARNING: Found {self.total_collisions} ID collisions!")
            print("[ID Check] Sample collisions (ID -> Found in files):")
            # Show top 5 collisions to avoid flooding the console
            for uid, files in list(self.collisions.items())[:5]:
                print(f"   - ID '{uid}': detected in {files}")
            if len(self.collisions) > 5:
                print(f"   ... and {len(self.collisions) - 5} more IDs.")

def extract_dataset_name(image_path):
    """
    Extracts dataset name from the image path.
    Supports both legacy absolute paths (with Datasets/ prefix) and
    relative paths (falls back to top-level directory name).
    """
    match = re.search(r"Datasets/([^/]+)", image_path)
    if match:
        return match.group(1)
    # For relative paths, use the top-level directory component
    parts = Path(image_path).parts
    if parts:
        return parts[0]
    return "unknown"

def process_files(input_files, output_file, dataset_names=None):
    """Aggregate multiple scene-graph JSON/JSONL files into a single JSONL output."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = IDTracker()
    count = 0

    # Using 'w' mode. Be careful: this overwrites existing files without asking.
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_idx, file_path in enumerate(input_files):
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found. Skipping.")
                continue

            try:
                print(f"Processing: {file_path}...")
                with open(file_path, encoding='utf-8') as f_in:
                    content = f_in.read().strip()
                    if content.startswith('['):
                        data_list = json.loads(content)
                    else:
                        data_list = [json.loads(line) for line in content.splitlines() if line.strip()]

                current_file_records = 0
                for item in data_list:
                    # 1. ID Collision Check
                    current_id = item.get("id")
                    if tracker.check_and_record(current_id, os.path.basename(file_path)):
                        # CRITICAL: Currently we just log it.
                        # If you want to skip duplicates, add 'continue' here.
                        pass

                    # 2. Add Metadata
                    if dataset_names and file_idx < len(dataset_names):
                        dataset_name = dataset_names[file_idx]
                    else:
                        img_path = item.get("image", "")
                        dataset_name = extract_dataset_name(img_path)

                    if "meta" not in item:
                        item["meta"] = {}
                    item["meta"]["dataset"] = dataset_name

                    # 3. Write to output
                    f_out.write(json.dumps(item) + "\n")
                    count += 1
                    current_file_records += 1

                print(f"  -> Added {current_file_records} records.")

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_path}: {e}")
            except Exception as e:
                print(f"Unexpected error processing {file_path}: {e}")

    print(f"\nDone. Total records merged: {count}. Saved to {output_file}")

    # Print the collision statistics at the end
    tracker.print_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge benchmark files with ID collision check.")
    parser.add_argument("--inputs", nargs='+', required=True, help="List of input JSON/JSONL files.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--dataset-names", nargs='*', default=None,
                        help="Dataset name for each input file (same order as --inputs). "
                             "If omitted, names are inferred from image paths.")

    args = parser.parse_args()
    process_files(args.inputs, args.output, dataset_names=args.dataset_names)
