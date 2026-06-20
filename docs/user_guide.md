# User Guide

Most users only need prepared benchmark files. Start with the evaluation path below unless you need to rebuild the benchmark from raw datasets.

## Path A: Evaluate with Prepared Benchmark Files

### 1. Install the environments

Follow [docs/installation.md](installation.md) first.

### 2. Place the required benchmark files

Keep these files under the standard repo layout:

```text
data/processed/
├── combined/
│   ├── QA_pairs/qa_data.jsonl
│   └── SGC_task/sgc_task.jsonl
│   └── scene_graph/combined_scene_graph.jsonl  # optional, only for crisp_qa_sg GT conversion
├── nuscenes/
│   └── annotated_image/marks.jsonl
└── scannetpp/
│   └── annotated_image/marks.jsonl
```

Keep the original dataset images reachable from the dataroots configured in:

- [configs/nuscenes.yaml](../configs/nuscenes.yaml)
- [configs/scannetpp.yaml](../configs/scannetpp.yaml)

### 3. Verify the setup

```bash
cd CRISP_bench
python scripts/verify_setup.py --env-file .env --skip-data-roots --check-prepared-benchmark
```

If you plan to run `crisp_qa_sg`, add `--check-qa-sg-master` so the shipped master scene graph source file is also validated.

### 4. Render the benchmark images from `marks.jsonl`

The prepared QA and SGC files reference repo-local annotated image paths.
Render those images once before stage 06:

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

If you rerun this step on a partially populated directory, add `--skip_existing`.

Before stage 06, set `NUM_GPUS` in [configs/eval.yaml](../configs/eval.yaml).
That single setting controls the default local GPU process count for the vLLM and accelerate-based wrappers.

### 5. Run one model

Example: Qwen3-VL-8B through the local `lmms-eval` checkout.

```bash
cd CRISP_bench
bash run.sh 06 --config configs/eval.yaml --model-script vllm_qwen3vl.sh
```

### 6. Run the full default matrix

```bash
cd CRISP_bench
bash run.sh 06 --config configs/eval.yaml
```

Preview the matrix without starting model jobs:

```bash
cd CRISP_bench
bash run.sh 06 --config configs/eval.yaml --dry-run
```

### 7. Run consistency evaluation

```bash
cd CRISP_bench
bash run.sh 07 --config configs/eval.yaml batch
```

Dry-run pair discovery first if needed:

```bash
cd CRISP_bench
bash run.sh 07 --config configs/eval.yaml batch --dry_run
```

### 8. Read the outputs

Main result files:

- `results/aggregated.json`
- `results/aggregated_long.csv`
- `results/qa_sg/`
- `results/consistency_score/`

## Path B: Rebuild the Benchmark from Raw Data

Only use this path if you need to reconstruct benchmark files from raw NuScenes or ScanNet++ downloads.

### 1. Edit the dataset configs

- [configs/nuscenes.yaml](../configs/nuscenes.yaml)
- [configs/scannetpp.yaml](../configs/scannetpp.yaml)

### 2. Prepare NuScenes

```bash
cd CRISP_bench
bash run.sh 01 --config configs/nuscenes.yaml all
bash run.sh 03 --config configs/nuscenes.yaml nuscenes
bash run.sh 04 --config configs/nuscenes.yaml nuscenes
bash run.sh 05 --config configs/nuscenes.yaml nuscenes
```

### 3. Prepare ScanNet++

Run step 02 in the ScanNet++ official environment, then return to the main environment:

```bash
cd CRISP_bench
bash run.sh 02 --config configs/scannetpp.yaml official
bash run.sh 03 --config configs/scannetpp.yaml scannetpp
bash run.sh 04 --config configs/scannetpp.yaml scannetpp
bash run.sh 05 --config configs/scannetpp.yaml scannetpp
```

### 4. Build the combined benchmark files

```bash
cd CRISP_bench
bash run.sh 04 all
bash run.sh 05 all
```

For dataset-specific details, use [docs/datasets.md](datasets.md) and [docs/pipeline.md](pipeline.md).

## Optional: QA-SG Evaluation

`crisp_qa_sg` is supported, but it needs two scene-graph JSON files.
The released data should include the master source file at `./data/processed/combined/scene_graph/combined_scene_graph.jsonl`.
Convert it once before you add a `crisp_qa_sg` model entry to [configs/eval.yaml](../configs/eval.yaml):

```bash
cd CRISP_bench
python evaluation/convert_GT_scene_graph.py \
  --input ./data/processed/combined/scene_graph/combined_scene_graph.jsonl \
  --output ./generated_sg/gt_sg.json
```

Then set these paths:

- `CRISP_QA_SG_GT_SG_PATH=./generated_sg/gt_sg.json`
- `CRISP_QA_SG_PRED_SG_PATH`

The config file already includes commented examples for both variables.
You can verify the optional master source file directly with:

```bash
cd CRISP_bench
python scripts/verify_setup.py \
  --env-file .env \
  --skip-data-roots \
  --check-qa-sg-master
```

## Supported Post-processing

Step 06 already runs the two supported collection scripts automatically.
You can rerun them manually if needed:

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