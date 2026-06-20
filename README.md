# From Hallucination to Grounding: Diagnosing Visual Spatial Intelligence via CRISP [ECCV 2026]

This repository contains the code and data for the CRISP benchmark, which is designed to evaluate the visual spatial intelligence of vision language models (VLMs) in a comprehensive manner. 

## Installation

Use Python 3.11 or newer.

Main repository environment:

```bash
cd CRISP_bench
uv sync --dev
source .venv/bin/activate
```

Local `lmms-eval` checkout:

```bash
cd CRISP_bench
git clone https://github.com/iiyamayuki/lmms-eval-CRISP-bench.git lmms-eval
cd lmms-eval
uv sync
cd ..
```

Detailed setup instructions are in [docs/installation.md](docs/installation.md).

## Config First

Edit these files before you run the pipeline:

- [configs/nuscenes.yaml](configs/nuscenes.yaml)
- [configs/scannetpp.yaml](configs/scannetpp.yaml)
- [configs/eval.yaml](configs/eval.yaml)

If the same variable appears both in a config file and in the shell environment, the config file wins.

## Quick Start

Download dataset from [https://huggingface.co/datasets/AiriMomoi/CRISP_Bench](https://huggingface.co/datasets/AiriMomoi/CRISP_Bench), and place them under the standard repo layout:

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

- [configs/nuscenes.yaml](configs/nuscenes.yaml)
- [configs/scannetpp.yaml](configs/scannetpp.yaml)

Verify the local setup:

```bash
cd CRISP_bench
python scripts/verify_setup.py --env-file .env --skip-data-roots --check-prepared-benchmark
```

Render the benchmark images referenced by the released QA and SGC files:

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

The prepared benchmark files open repo-local annotated image paths directly.
If `images_with_bbox/` already exists and is complete, you can skip this step or add `--skip_existing`.

If you also plan to run `crisp_qa_sg`, validate the shipped master scene graph source file as well:

```bash
cd CRISP_bench
python scripts/verify_setup.py \
	--env-file .env \
	--skip-data-roots \
	--check-prepared-benchmark \
	--check-qa-sg-master
```

Before running the evaluation, set the local GPU count once in [configs/eval.yaml](configs/eval.yaml) through `NUM_GPUS`.

Run one model:

```bash
cd CRISP_bench
bash run.sh 06 --config configs/eval.yaml --model-script vllm_qwen3vl.sh
```

Preview the default evaluation matrix:

```bash
cd CRISP_bench
bash run.sh 06 --config configs/eval.yaml --dry-run
```

Run the default evaluation matrix:

```bash
cd CRISP_bench
bash run.sh 06 --config configs/eval.yaml
```

Run consistency evaluation:

```bash
cd CRISP_bench
bash run.sh 07 --config configs/eval.yaml batch
```

Main result locations:

- `results/aggregated.json`
- `results/aggregated_long.csv`
- `results/qa_sg/`
- `results/consistency_score/`

If you need to rebuild benchmark files from raw NuScenes or ScanNet++, use [docs/datasets.md](docs/datasets.md) and [docs/pipeline.md](docs/pipeline.md).

## Workflow Entrypoints

| Script | Purpose |
| --- | --- |
| `run.sh` | Top-level dispatcher for staged workflow commands |
| `scripts/01_preprocess_nuscenes.sh` | NuScenes preprocessing |
| `scripts/02_preprocess_scannetpp.sh` | ScanNet++ official preprocessing |
| `scripts/03_build_scene_graph.sh` | Shared scene graph construction |
| `scripts/04_generate_qa.sh` | QA generation |
| `scripts/05_prepare_tasks.sh` | SGC task export |
| `scripts/06_run_evaluation.sh` | Model evaluation plus result collection |
| `scripts/07_consistency_eval.sh` | Consistency evaluation |

## Evaluation Tasks

Supported default tasks:

- `crisp_qa`
- `crisp_sgc`
- optional `crisp_qa_sg`

The default `model_matrix` in [configs/eval.yaml](configs/eval.yaml) only includes the first two tasks.
If you want to run `crisp_qa_sg`, the release bundle should include the master scene-graph source file at `./data/processed/combined/scene_graph/combined_scene_graph.jsonl`.
Convert it once with:

```bash
cd CRISP_bench
python evaluation/convert_GT_scene_graph.py \
	--input ./data/processed/combined/scene_graph/combined_scene_graph.jsonl \
	--output ./generated_sg/gt_sg.json
```

Then set `CRISP_QA_SG_GT_SG_PATH=./generated_sg/gt_sg.json`, and point `CRISP_QA_SG_PRED_SG_PATH` to the predicted scene graph JSON you want to compare against.

## Documentation

- [docs/user_guide.md](docs/user_guide.md)
- [docs/installation.md](docs/installation.md)
- [docs/pipeline.md](docs/pipeline.md)
- [docs/datasets.md](docs/datasets.md)
- [docs/data_format.md](docs/data_format.md)

## Citation

Citation metadata is tracked in `CITATION.cff`.
Until the paper link is public, please use:

```bibtex
@misc{li_yu_crisp,
  title = {From Hallucination to Grounding: Diagnosing Visual Spatial Intelligence via CRISP},
  author = {Zhixing Li and Yinan Yu},
  note = {Public paper link to be added}
}
```

## License

CRISP Bench is released under the MIT License. See `LICENSE`.
