# Installation

Use `configs/*.yaml` as the main runtime configuration.
If a setting appears both in a config file and in the shell environment, the config file wins.
Use `.env` for secrets and cache locations only.
The commands below assume your repo root directory is named `CRISP_bench`.

## 1. Main Repository Environment

Recommended setup:

```bash
cd CRISP_bench
uv sync --dev
source .venv/bin/activate
```

If you do not use `uv`:

```bash
cd CRISP_bench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Local `lmms-eval` Checkout

Keep a local `lmms-eval` checkout at `./lmms-eval`.
The evaluation wrappers under `scripts/eval_models/` use this local path by default.
Use the CRISP Bench-compatible checkout from `iiyamayuki/lmms-eval-CRISP-bench`:

```bash
cd CRISP_bench
git clone https://github.com/iiyamayuki/lmms-eval-CRISP-bench.git lmms-eval
cd lmms-eval
uv sync
cd ..
```

If your local checkout lives somewhere else, edit `LMMS_EVAL_ROOT` in [configs/eval.yaml](../configs/eval.yaml).

## 3. Optional ScanNet++ Official Environment

You only need this environment for `bash run.sh 02 ...`.

```bash
cd CRISP_bench/data_preprocessing/scannetpp_official
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ../../
```

## 4. Local Secret and Cache File

Create a local `.env` file:

```bash
cd CRISP_bench
cp .env.example .env
```

Fill in only what you need:

- `OPENAI_API_KEY` for OpenAI-hosted models
- `GOOGLE_API_KEY` for Gemini-family models
- optional cache locations such as `HF_HOME`, `LMMS_EVAL_HOME`, and `VLLM_CACHE_ROOT`

## 5. Config Files You Should Edit

Edit these files before running the pipeline:

- [configs/nuscenes.yaml](../configs/nuscenes.yaml)
- [configs/scannetpp.yaml](../configs/scannetpp.yaml)
- [configs/eval.yaml](../configs/eval.yaml)

## 6. Verify the Setup

Quick check:

```bash
cd CRISP_bench
python scripts/verify_setup.py --env-file .env --skip-data-roots
```

Prepared benchmark check:

```bash
cd CRISP_bench
python scripts/verify_setup.py --env-file .env --skip-data-roots --check-prepared-benchmark
```

If your raw datasets are already in place:

```bash
cd CRISP_bench
python scripts/verify_setup.py \
	--env-file .env \
	--nuscenes-root ./data/raw/nuscenes \
	--scannetpp-root ./data/raw/scannetpp
```

Next:

- if you are using prepared benchmark files, run `verify_setup.py` with `--check-prepared-benchmark` so it also checks `qa_data.jsonl`, `sgc_task.jsonl`, and both `marks.jsonl` manifests
- if you also plan to run `crisp_qa_sg`, add `--check-qa-sg-master`; the default target is `./data/processed/combined/scene_graph/combined_scene_graph.jsonl`
- if you plan to use local GPUs for stage 06, set `NUM_GPUS` in [configs/eval.yaml](../configs/eval.yaml); the local wrappers derive tensor-parallel size or accelerate process count from that value by default
- if you are using prepared benchmark files, render `images_with_bbox/` from the provided `marks.jsonl` manifests before stage 06; the commands are documented in [docs/user_guide.md](user_guide.md)
- use [docs/user_guide.md](user_guide.md) for the evaluation flow
- use [docs/datasets.md](datasets.md) only if you need to rebuild data from raw datasets
