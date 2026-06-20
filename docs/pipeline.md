# Pipeline

Most users only need stages 06 and 07. If you already have prepared benchmark files under `data/processed/combined`, start with the evaluation path below.

## Common Evaluation Path

Required files:

- `data/processed/combined/QA_pairs/qa_data.jsonl`
- `data/processed/combined/SGC_task/sgc_task.jsonl`
- `data/processed/nuscenes/annotated_image/marks.jsonl`
- `data/processed/scannetpp/annotated_image/marks.jsonl`

Optional file for `crisp_qa_sg` GT conversion:

- `data/processed/combined/scene_graph/combined_scene_graph.jsonl`

Keep the original dataset images reachable from the dataroots configured in [configs/nuscenes.yaml](../configs/nuscenes.yaml) and [configs/scannetpp.yaml](../configs/scannetpp.yaml).

Render the repo-local annotated images before stage 06:

```bash
cd CRISP_bench
python data_preprocessing/render_marks.py \
    --marks ./data/processed/nuscenes/annotated_image/marks.jsonl \
    --dataroot ./data/raw/nuscenes \
    --output_dir ./data/processed/nuscenes/annotated_image/images_with_bbox

python data_preprocessing/render_marks.py \
    --marks ./data/processed/scannetpp/annotated_image/marks.jsonl \
    --dataroot ./data/raw/scannetpp \
    --output_dir ./data/processed/scannetpp/annotated_image/images_with_bbox
```

If the `images_with_bbox/` directories are already populated, you can skip this step or add `--skip_existing`.

Before stage 06, set `NUM_GPUS` in [configs/eval.yaml](../configs/eval.yaml).
The local vLLM wrapper uses it as the default tensor-parallel size, and the accelerate-based wrappers use it as the default process count.

## Optional: Convert QA-SG GT Input

You only need this step if you want to run `crisp_qa_sg` with GT scene graphs.

```bash
cd CRISP_bench
python evaluation/convert_GT_scene_graph.py \
    --input ./data/processed/combined/scene_graph/combined_scene_graph.jsonl \
    --output ./generated_sg/gt_sg.json
```

Then set `CRISP_QA_SG_GT_SG_PATH=./generated_sg/gt_sg.json` and point `CRISP_QA_SG_PRED_SG_PATH` to your predicted scene-graph JSON before starting stage 06.

Run one model:

```bash
cd CRISP_bench
bash run.sh 06 --config configs/eval.yaml --model-script vllm_qwen3vl.sh
```

Run the full default model matrix:

```bash
cd CRISP_bench
bash run.sh 06 --config configs/eval.yaml
```

Run consistency evaluation:

```bash
cd CRISP_bench
bash run.sh 07 --config configs/eval.yaml batch
```

Use the numbered shell entrypoints under `run.sh` below only if you need the full rebuild workflow.

## Full Stage Summary

| Stage | Environment | Command | Main output |
| --- | --- | --- | --- |
| 01 | main repo env | `bash run.sh 01 --config configs/nuscenes.yaml all` | `data/processed/nuscenes/scene_graph/filtered_nodes_cam.json` |
| 02 | ScanNet++ official env | `bash run.sh 02 --config configs/scannetpp.yaml official` | `data/processed/scannetpp/scene_graph/scene_graph.json` |
| 03 | main repo env | `bash run.sh 03 --config configs/nuscenes.yaml nuscenes` | `filtered_scene_graph.json` |
| 04 | main repo env | `bash run.sh 04 --config configs/nuscenes.yaml nuscenes` | `qa_data.jsonl` |
| 05 | main repo env | `bash run.sh 05 --config configs/nuscenes.yaml nuscenes` | `sgc_task.json` |
| 06 | local `lmms-eval` checkout | `bash run.sh 06 --config configs/eval.yaml` | `lmms-eval/logs/*` and `results/aggregated.json` |
| 07 | main repo env | `bash run.sh 07 --config configs/eval.yaml batch` | `results/consistency_score/*.json` |

## Rebuild Order

### NuScenes

```bash
cd CRISP_bench
bash run.sh 01 --config configs/nuscenes.yaml all
bash run.sh 03 --config configs/nuscenes.yaml nuscenes
bash run.sh 04 --config configs/nuscenes.yaml nuscenes
bash run.sh 05 --config configs/nuscenes.yaml nuscenes
```

### ScanNet++

```bash
cd CRISP_bench
bash run.sh 02 --config configs/scannetpp.yaml official
bash run.sh 03 --config configs/scannetpp.yaml scannetpp
bash run.sh 04 --config configs/scannetpp.yaml scannetpp
bash run.sh 05 --config configs/scannetpp.yaml scannetpp
```

### Combined Benchmark Files

```bash
cd CRISP_bench
bash run.sh 04 all
bash run.sh 05 all
```

If you only need evaluation, you can skip stages 01 to 05 and use the common evaluation path above.

## Optional: Export Reusable `marks.jsonl`

You do not need this step for normal benchmark evaluation.
Use it only if you are rebuilding raw data and also want a standalone marks manifest that can be rendered later on another machine.

NuScenes:

```bash
cd CRISP_bench
python data_preprocessing/export_marks.py \
    --dataset nuscenes \
    --input ./data/processed/nuscenes/scene_graph/nodes_filtered.json \
    --output ./data/processed/nuscenes/annotated_image/marks.jsonl \
    --nusc_root ./data/raw/nuscenes \
    --nusc_version v1.0-trainval
```

ScanNet++:

```bash
cd CRISP_bench
python data_preprocessing/export_marks.py \
    --dataset scannetpp \
    --input ./data/processed/scannetpp/scene_graph/nodes_filtered.json \
    --output ./data/processed/scannetpp/annotated_image/marks.jsonl \
    --dataroot ./data/raw/scannetpp \
    --original_jsonl_dir ./data/processed/scannetpp/scene_graph/sg_jsonl
```

## Result Collection

Step 06 already runs the two supported collection scripts automatically.
You can also run them manually:

```bash
cd CRISP_bench
python collect_results/collect_results.py \
    --input_dir ./lmms-eval/logs \
    --output_dir ./results \
    --consistency_dir ./results/consistency_score

python collect_results/collect_qa_sg.py \
    --input_dir ./lmms-eval/logs \
    --output_dir ./results/qa_sg
```

## Confusion Matrices

Relation confusion matrix:

```bash
cd CRISP_bench
python visualization/visualize_relation_confusion_matrix.py \
    --input ./results/aggregated.json \
    --output_dir ./results/relation_confusion_matrices
```

QA confusion matrix:

```bash
cd CRISP_bench
python visualization/visualize_qa_confusion_matrix.py \
    --input_dir ./results \
    --output_dir ./results/qa_confusion_matrices \
    --derived_qa_dir ./generated_sg
```