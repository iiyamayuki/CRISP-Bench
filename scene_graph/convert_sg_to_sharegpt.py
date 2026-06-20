import argparse
import json
import math
import os
import sys
from typing import Any


def build_prompt(object_ids, center_id):
    """
    Constructs the instruction prompt for the model in the id-only setting.
    Each object in the image is marked with an integer id (Object 1, Object 2, ...).
    The model must build a 3D scene graph using only these ids.
    """
    # Expected output schema
    schema_example = """
    {
      "objects": [
        {
          "id": int,             
          "dist_to_cam": float, 
          "size": {
            "w": float,
            "l": float,
            "h": float
          }
        }
      ],
      "edges": [
        {
          "from": int,           
          "to": int,             
          "distance": float,     
          "relation": ["string"] 
        }
      ]
    }
    """

    # Bulleted list of all object ids in the scene
    # (These ids are assumed to be shown in the image as boxes with numbers.)
    object_list_str = "\n".join([f"- Object {oid}" for oid in object_ids])

    prompt = (
        "You are given a real world image where each object is marked "
        "with a numeric id (Object 1, Object 2, ...).\n\n"

        "**Task:**\n"
        "Generate a 3D scene graph JSON that describes the spatial layout of the scene.\n\n"

        "**Requirements:**\n"
        "1. Create exactly one entry in `objects` for **every** id listed in 'Scene Objects'.\n"
        "2. Create edges from the **Center Object** to **all other** objects.\n"
        "3. Analyze the spatial relationship from the center object to each target object.\n"
        "4. For `relation`, list **ALL** applicable terms (e.g., `['left', 'in front of']`).\n"
        "5. For `relation`, use ONLY these words: `['left', 'right', 'in front of', 'behind', 'up', 'down']`.\n\n"

        "**Output Format:**\n"
        f"```json\n{schema_example}\n```\n\n"

        "**Input Data:**\n"
        f"Center Object id: {center_id}\n"
        f"Scene Objects (ids shown in the image):\n{object_list_str}\n\n"

        "**Response:**\n"
        "Output the valid JSON only, without any extra text."
    )

    return prompt


def distance_to_camera(obj: dict[str, Any]) -> float:
    """
    Euclidean distance in camera coordinate frame:
      sqrt(x^2 + y^2 + z_cam^2)
    """
    t = obj["attributes"]["translation"]
    x, y, z = t["x"], t["y"], t["z_cam"]
    return math.sqrt(x * x + y * y + z * z)

def convert_sg_to_sharegpt(input_path, output_path):
    """
    Reads the GT JSON file, converts it to ShareGPT format, and saves it.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    print(f"Loading data from {input_path}...")
    try:
        with open(input_path, encoding='utf-8') as f:
            gt_data_list = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)

    sharegpt_data = []
    processed_count = 0
    skipped_count = 0

    for sample in gt_data_list:
        image_path = sample.get("image_with_2dbox", "")

        # -------------------------------------------------
        # 1. Parse Objects
        # -------------------------------------------------
        if "objects" not in sample:
            skipped_count += 1
            continue

        node_id_to_local_id = {}  # Mapping: node_id -> local_id
        all_local_ids = []        # List of local_ids for the prompt
        output_objects = []       # List for the GPT output (contains all objects)

        for obj in sample["objects"]:
            node_id = obj["node_id"]
            local_id = obj.get("local_id")
            attrs = obj.get("attributes", {})

            # Skip objects without a local_id
            if local_id is None:
                continue

            obj_data = {
                "id": local_id,
                "dist_to_cam": distance_to_camera(obj),
                "size": attrs.get("size", {})
            }

            node_id_to_local_id[node_id] = local_id
            all_local_ids.append(local_id)
            output_objects.append(obj_data)

        # -------------------------------------------------
        # 2. Identify Center Object and Parse Edges
        # -------------------------------------------------
        output_edges = []
        center_node_id = None
        center_local_id = None

        # We assume there is exactly one center object per sample,
        # defined by the 'from' field of the first edge.
        if "edges" in sample and len(sample["edges"]) > 0:
            first_edge = sample["edges"][0]
            center_node_id = first_edge["from"]

            # Validate if center_id exists in the parsed objects
            if center_node_id in node_id_to_local_id:
                center_local_id = node_id_to_local_id[center_node_id]

                # Process all edges for this center object
                for edge in sample["edges"]:
                    src = edge["from"]
                    dst = edge["to"]

                    # Only process edges starting from the center and pointing to a valid object
                    if src == center_node_id and dst in node_id_to_local_id:
                        dst_local_id = node_id_to_local_id[dst]

                        new_edge = {
                            "from": center_local_id,
                            "to": dst_local_id,
                            "distance": edge.get("distance"),
                            "relation": edge.get("relation")
                        }
                        output_edges.append(new_edge)
            else:
                # Center object ID not found in object list, skip sample
                skipped_count += 1
                continue
        else:
            # No edges found, cannot define a scene graph task, skip sample
            skipped_count += 1
            continue

        # -------------------------------------------------
        # 3. Construct ShareGPT Entry
        # -------------------------------------------------

        # Construct Human Prompt
        human_input = build_prompt(all_local_ids, center_local_id)

        # Construct GPT Output (JSON)
        gpt_response_obj = {
            "objects": output_objects, # Contains all objects in the scene
            "edges": output_edges      # Contains edges from the center object
        }
        gpt_response_str = json.dumps(gpt_response_obj, indent=2)

        conversation_entry = {
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": human_input
                },
                {
                    "from": "gpt",
                    "value": gpt_response_str
                }
            ]
        }
        sharegpt_data.append(conversation_entry)
        processed_count += 1

    print("Conversion complete.")
    print(f"Processed: {processed_count} samples.")
    print(f"Skipped: {skipped_count} samples (due to missing edges or invalid IDs).")
    print(f"Saving to {output_path}...")

    # Save to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Ground Truth Scene Graph JSON to ShareGPT format for evaluation.")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSON file containing ground truth scene graphs."
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSON file to save the ShareGPT formatted data."
    )

    args = parser.parse_args()

    convert_sg_to_sharegpt(args.input, args.output)
