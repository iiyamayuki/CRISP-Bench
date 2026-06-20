# Data Format

This page lists the main user-facing artifact formats used by the pipeline.

## Path Convention

Raw scene-graph `image` fields and `marks.jsonl` `source_image` fields are stored relative to the dataset root.
Generated `image_with_2dbox` fields and evaluation-task `image` fields are stored relative to the repository root.
Resolve raw-image fields with the dataroot you configured in:

- [configs/nuscenes.yaml](../configs/nuscenes.yaml)
- [configs/scannetpp.yaml](../configs/scannetpp.yaml)

The `render_marks.py` script regenerates benchmark images from `marks.jsonl`.
It chooses the output filename from dataset metadata and writes into the `--output_dir` you provide.

## 1. Scene Graph JSON

Scene graph files are JSON lists.
Each item represents one image.

```json
[
  {
    "image": "samples/CAM_FRONT/example.jpg",
    "image_with_2dbox": "data/processed/nuscenes/annotated_image/images_with_bbox/example.jpg",
    "objects": [],
    "edges": []
  }
]
```

Common keys:

- `image`: raw image path, relative to the dataset root
- `image_with_2dbox`: annotated image path, relative to the repository root
- `objects`: object list
- `edges`: spatial relation list

Typical object entry:

```json
{
  "node_id": "instance-token",
  "local_id": 3,
  "attributes": {
    "category_name": "vehicle.car",
    "translation": {"x": 1.2, "y": 0.5, "z_cam": 8.4},
    "size": {"w": 1.8, "l": 4.4, "h": 1.6}
  }
}
```

Typical edge entry:

```json
{
  "from": "ego",
  "to": "instance-token",
  "distance": 8.52,
  "relation": {
    "left": 0,
    "right": 1,
    "in front of": 1,
    "behind": 0,
    "up": 0,
    "down": 0
  }
}
```

## 2. Marks JSONL

`marks.jsonl` is the portable manifest used to regenerate benchmark images from raw dataset images.
Each line represents one source image.

Minimal item:

```json
{
  "dataset": "nuscenes",
  "sample_data_token": "example-token",
  "camera_channel": "CAM_FRONT",
  "source_image": "samples/CAM_FRONT/example.jpg",
  "image_size": {"width": 1600, "height": 900},
  "marks": [
    {
      "node_id": "instance-token",
      "local_id": 3,
      "bbox_xyxy": [100, 200, 260, 420],
      "render": {
        "circle_center": [180, 310],
        "radius": 24,
        "alpha": 0.6,
        "font_scale": 1.2,
        "thickness": 2,
        "text_anchor": [168, 324]
      }
    }
  ]
}
```

Common keys:

- `dataset`: `nuscenes` or `scannetpp`
- `scene_id`: present for ScanNet++ and used in the rendered filename
- `sample_data_token`: frame identifier used in the rendered filename
- `source_image`: raw image path, relative to the dataset root
- `image_size`: expected source image size for validation
- `marks`: ordered list of object ids, clipped boxes, and exact render parameters

Current rendered filename convention:

- NuScenes: `{sample_data_token}.jpg`
- ScanNet++: `{scene_id}_{sample_data_token}.jpg`

## 3. QA JSONL

The benchmark QA file is ShareGPT-style JSONL.
Each line is one question.

```json
{
  "id": "sample-token_0",
  "image": "data/processed/nuscenes/annotated_image/images_with_bbox/example.jpg",
  "conversations": [
    {"from": "human", "value": "Question text"},
    {"from": "gpt", "value": "A"}
  ],
  "meta": {
    "category": "direction",
    "difficulty": "easy",
    "type": "MCQ"
  }
}
```

Common keys:

- `id`: item id
- `image`: annotated image path, relative to the repository root
- `conversations`: two-turn QA conversation
- `meta.category`: question category
- `meta.difficulty`: question difficulty
- `meta.type`: `MCQ` or `NAQ`

## 4. QA-SG Master Scene Graph Source

`data/processed/combined/scene_graph/combined_scene_graph.jsonl` is the shipped master scene-graph source file for the optional `crisp_qa_sg` workflow.
It keeps the full object metadata used by `evaluation/convert_GT_scene_graph.py`.

Minimal item:

```json
{
  "image": "samples/CAM_FRONT/example.jpg",
  "image_with_2dbox": "data/processed/nuscenes/annotated_image/images_with_bbox/example.jpg",
  "objects": [
    {
      "node_id": "instance-token",
      "local_id": 3,
      "attributes": {
        "category_name": "vehicle.car",
        "translation": {"x": 1.2, "y": 0.5, "z_cam": 8.4},
        "size": {"w": 1.8, "l": 4.4, "h": 1.6}
      }
    }
  ],
  "edges": [
    {
      "from": "ego",
      "to": "instance-token",
      "distance": 2.3,
      "relation": {
        "left": 1,
        "right": 0,
        "in front of": 1,
        "behind": 0,
        "up": 0,
        "down": 0
      }
    }
  ]
}
```

Common keys:

- `image`: raw image path, relative to the dataset root
- `image_with_2dbox`: annotated image path used by the QA task after conversion
- `objects[*].node_id`: original object identifier
- `objects[*].local_id`: numeric object id shown in the marked image
- `objects[*].attributes.translation`: camera-relative geometry used to compute distance
- `objects[*].attributes.size`: object box dimensions

Convert this source file to the simplified GT input consumed by `crisp_qa_sg` with:

```bash
cd CRISP_bench
python evaluation/convert_GT_scene_graph.py \
  --input ./data/processed/combined/scene_graph/combined_scene_graph.jsonl \
  --output ./generated_sg/gt_sg.json
```

## 5. Converted QA-SG GT JSON

`generated_sg/gt_sg.json` is the derived JSON file consumed by the optional `crisp_qa_sg` task after running the conversion step above.
It is not assumed to be shipped in the public bundle.

Minimal item:

```json
{
  "image": "data/processed/nuscenes/annotated_image/images_with_bbox/example.jpg",
  "objects": [
    {
      "id": 3,
      "dist_to_cam": 8.52,
      "size": {"w": 1.8, "l": 4.4, "h": 1.6}
    }
  ],
  "edges": [
    {
      "from": 3,
      "to": 5,
      "distance": 2.3,
      "relation": ["left", "in front of"]
    }
  ]
}
```

Common keys:

- `image`: image key matched against the QA document image
- `objects[*].id`: numeric object id used in the marked image
- `objects[*].dist_to_cam`: object distance to camera
- `objects[*].size`: object box dimensions
- `edges`: relation list used by the QA-SG prompt

## 6. SGC Task File

Dataset-level SGC export is a JSON list.
The combined benchmark file under `data/processed/combined/SGC_task/sgc_task.jsonl` is the aggregated JSONL version of the same task.

Minimal item:

```json
{
  "image": "data/processed/nuscenes/annotated_image/images_with_bbox/example.jpg",
  "conversations": [
    {"from": "human", "value": "Generate a 3D scene graph JSON..."},
    {"from": "gpt", "value": "{\n  \"objects\": [],\n  \"edges\": []\n}"}
  ]
}
```

The assistant response is a JSON string with two top-level fields:

- `objects`: id, distance-to-camera, and size
- `edges`: center-object edges with distance and relation labels

## 7. Consistency Outputs

Step 07 writes three kinds of files:

- `generated_sg/*_sg.json`: scene graphs extracted from model logs
- `generated_sg/*_qa.jsonl`: QA answers derived from those scene graphs
- `results/consistency_score/*_consistency_results.json`: final consistency metrics

## 8. Aggregated Results

`collect_results/collect_results.py` writes `results/aggregated.json`.

Minimal structure:

```json
{
  "qwen3vl_vllm/Qwen__Qwen3-VL-8B-Instruct": {
    "suite": "qwen3vl_vllm",
    "model": "Qwen__Qwen3-VL-8B-Instruct",
    "tasks": {
      "crisp_qa": {
        "multimodal": {
          "metrics": {
            "overall": 0.54
          }
        }
      }
    }
  }
}
```

`collect_results/collect_qa_sg.py` writes its outputs under `results/qa_sg/`.
