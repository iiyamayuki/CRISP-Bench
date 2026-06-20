#!/usr/bin/env python3
"""
Script for converting NuScenes scene graphs to a simplified format.
"""
import argparse
import json
import math
from typing import Any


def calculate_distance_to_camera(xyz_cam: dict[str, float]) -> float:
    """
    Calculate the Euclidean distance to the camera from xyz_cam coordinates.
    
    Args:
        xyz_cam: Dictionary containing x, y, and z_cam.
        
    Returns:
        float: Euclidean distance to the camera.
    """
    x = xyz_cam.get('x', 0)
    y = xyz_cam.get('y', 0)
    z_cam = xyz_cam.get('z_cam', 0)

    distance = math.sqrt(x**2 + y**2 + z_cam**2)
    return distance


def convert_objects(objects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Convert the objects format and build a mapping from node_id to local_id.
    
    Args:
        objects: Original objects list.
        
    Returns:
        Converted objects list and node_id-to-local_id mapping dictionary.
    """
    converted_objects = []
    node_id_to_local_id = {}

    for obj in objects:
        local_id = obj['local_id']
        node_id = obj['node_id']
        attributes = obj['attributes']

        # Build the mapping.
        node_id_to_local_id[node_id] = local_id

        # Compute the distance to the camera.
        dist_to_cam = calculate_distance_to_camera(attributes['translation'])

        # Build the new object format.
        converted_obj = {
            'id': local_id,
            'dist_to_cam': round(dist_to_cam, 6),
            'size': {
                'w': attributes['size']['w'],
                'l': attributes['size']['l'],
                'h': attributes['size']['h']
            }
        }

        converted_objects.append(converted_obj)

    return converted_objects, node_id_to_local_id


def convert_edges(edges: list[dict[str, Any]], node_id_to_local_id: dict[str, int]) -> list[dict[str, Any]]:
    """
    Convert the edges format, map node_id to local_id, and extract active relations.
    
    Args:
        edges: Original edges list.
        node_id_to_local_id: Mapping from node_id to local_id.
        
    Returns:
        Converted edges list.
    """
    converted_edges = []

    for edge in edges:
        from_node_id = edge['from']
        to_node_id = edge['to']

        # Resolve local IDs.
        from_local_id = node_id_to_local_id.get(from_node_id)
        to_local_id = node_id_to_local_id.get(to_node_id)

        if from_local_id is None or to_local_id is None:
            # Skip edges that cannot be mapped.
            continue

        # Extract active relations (relations with value 1).
        relations = edge.get('relation', {})
        active_relations = [rel_name for rel_name, value in relations.items() if value == 1]

        # Build the new edge format.
        converted_edge = {
            'from': from_local_id,
            'to': to_local_id,
            'distance': round(edge['distance'], 4),
            'relation': active_relations
        }

        converted_edges.append(converted_edge)

    return converted_edges


def convert_scene_graph(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a single scene graph.
    
    Args:
        input_data: Original scene graph data.
        
    Returns:
        Converted scene graph data.
    """
    # Convert objects.
    converted_objects, node_id_to_local_id = convert_objects(input_data['objects'])

    # Convert edges.
    converted_edges = convert_edges(input_data['edges'], node_id_to_local_id)

    # Build the result.
    result = {
        'image': input_data.get('image_with_2dbox', input_data.get('image', '')),
        'objects': converted_objects,
        'edges': converted_edges
    }

    return result


def main():
    """CLI entry point for converting ground-truth scene graphs to evaluation format."""
    parser = argparse.ArgumentParser(description='Convert NuScenes scene graph format')
    parser.add_argument(
        '--input',
        type=str,
        default='filtered_scene_graph.json',
        help='Path to the input JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='converted_scene_graph.json',
        help='Path to the output JSON file'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print the output JSON'
    )

    args = parser.parse_args()

    # Read the input file. Supports both JSON and JSONL formats.
    print(f"Reading input file: {args.input}")
    input_data = []

    # Detect the file format.
    if args.input.endswith('.jsonl'):
        # JSONL format: one JSON object per line.
        with open(args.input, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines.
                    try:
                        input_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: failed to parse line {line_num}: {e}")
    else:
        # Standard JSON format.
        with open(args.input, encoding='utf-8') as f:
            input_data = json.load(f)

    print(f"Found {len(input_data)} scene graphs to convert")

    # Convert all scene graphs.
    converted_data = []
    for i, scene_graph in enumerate(input_data):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(input_data)} scene graphs...")

        converted_sg = convert_scene_graph(scene_graph)
        converted_data.append(converted_sg)

    # Save the output file.
    print(f"Writing output file: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        if args.pretty:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(converted_data, f, ensure_ascii=False)

    print(f"Conversion complete. Converted {len(converted_data)} scene graphs")

    # # Print an example.
    # if converted_data:
    #     print("\nExample output (first scene graph):")
    #     print(json.dumps(converted_data[0], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
