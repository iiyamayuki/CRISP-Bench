# Datasets

Only use this page if you need to rebuild benchmark files from raw datasets.
If you already have prepared benchmark files, go to [docs/user_guide.md](user_guide.md) and start from the evaluation path there.

This repository uses a simple repo-local layout by default.

```text
data/
├── raw/
│   ├── nuscenes/
│   └── scannetpp/
└── processed/
	 ├── nuscenes/
	 ├── scannetpp/
	 └── combined/
```

If your raw data lives somewhere else, edit the matching paths in [configs/nuscenes.yaml](../configs/nuscenes.yaml) and [configs/scannetpp.yaml](../configs/scannetpp.yaml).

## NuScenes

1. Download the official NuScenes data and set `NUSCENES_VERSION` to the split `v1.0-mini`.
2. Edit these keys in [configs/nuscenes.yaml](../configs/nuscenes.yaml):
	- `NUSCENES_DATAROOT`
	- `NUSCENES_VERSION`
3. Run the NuScenes pipeline:

```bash
cd CRISP_bench
bash run.sh 01 --config configs/nuscenes.yaml all
bash run.sh 03 --config configs/nuscenes.yaml nuscenes
bash run.sh 04 --config configs/nuscenes.yaml nuscenes
bash run.sh 05 --config configs/nuscenes.yaml nuscenes
```

Main outputs:

- `data/processed/nuscenes/scene_graph/filtered_scene_graph.json`
- `data/processed/nuscenes/QA_pairs/qa_data.jsonl`
- `data/processed/nuscenes/SGC_task/sgc_task.json`

## ScanNet++

1. Download ScanNet++ through the official access process. Use [data_preprocessing/scannetpp_official/my_config.yml](../data_preprocessing/scannetpp_official/my_config.yml) as a starting configuration file.
2. Edit these keys in [configs/scannetpp.yaml](../configs/scannetpp.yaml):
	- `SCANNETPP_DATAROOT`
	- `SCANNETPP_SCENE_DIR`
3. Run the official preprocessing step in the ScanNet++ official environment:

```bash
cd CRISP_bench
bash run.sh 02 --config configs/scannetpp.yaml official
```

4. Switch back to the main repository environment and continue:

```bash
cd CRISP_bench
bash run.sh 03 --config configs/scannetpp.yaml scannetpp
bash run.sh 04 --config configs/scannetpp.yaml scannetpp
bash run.sh 05 --config configs/scannetpp.yaml scannetpp
```

Main outputs:

- `data/processed/scannetpp/scene_graph/filtered_scene_graph.json`
- `data/processed/scannetpp/QA_pairs/qa_data.jsonl`
- `data/processed/scannetpp/SGC_task/sgc_task.json`

## Build the Combined Benchmark Files

After both datasets are ready under the standard repo layout, build the merged benchmark files:

```bash
cd CRISP_bench
bash run.sh 04 all
bash run.sh 05 all
```

Main outputs:

- `data/processed/combined/QA_pairs/qa_data.jsonl`
- `data/processed/combined/SGC_task/sgc_task.jsonl`

These combined files are the default inputs for the evaluation tasks under `tasks/`.
Once they are ready, return to [docs/user_guide.md](user_guide.md) for the evaluation steps.

If you also want to run `crisp_qa_sg` with GT scene graphs, convert the combined master scene-graph file once:

```bash
cd CRISP_bench
python evaluation/convert_GT_scene_graph.py \
	--input ./data/processed/combined/scene_graph/combined_scene_graph.jsonl \
	--output ./generated_sg/gt_sg.json
```

Then use `./generated_sg/gt_sg.json` as `CRISP_QA_SG_GT_SG_PATH` during stage 06.

## Optional: Export Reusable `marks.jsonl`

You do not need this step for normal benchmark evaluation.
Stage 03 already creates the annotated images required by the downstream pipeline.
Use `export_marks.py` only if you need a standalone marks manifest that can later regenerate those images with `render_marks.py`.

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

## Path Rule

Raw `image` and `marks.jsonl` `source_image` fields are stored relative to the dataset root.
Generated `image_with_2dbox` fields and downstream evaluation-task `image` fields are stored relative to the repository root.
Keep the raw images reachable from the dataroot you configured in the YAML files, and run evaluation from the repository root.
