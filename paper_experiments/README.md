# Paper Experiments

> 中文版：[README_CN.md](README_CN.md)

`paper_experiments/` is now the single entry point for experiment-related code in this repo.

## Directory Map

- `latest_release/`
  - The latest Section 4.x experiment code that should be read first.
  - Includes the latest runnable section scripts, shared helpers, runtime support modules, manifest template, and the latest generated results.
- `preprocessing/`
  - Data preprocessing and score-building utilities.
  - Includes both the current preprocessing pipeline and older raw-trace / score-builder scripts kept for reproducibility.
- `archive/`
  - Historical iterations, old experiment scripts, and intermediate experiment artifacts that are no longer the primary code path.

## What New Readers Should Open First

1. `latest_release/README.md`
2. `latest_release/sections/`
3. `latest_release/results/`
4. `docs/experiment_pipeline.md` — full stage-by-stage technical reference (EN)
5. `../PROJECT_GUIDE_CN.md` — newcomer guide (中文)

## Quick Start

### 1. Build dataset files

```bash
python paper_experiments/preprocessing/current_pipeline/preprocess_traces.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --skip-missing
```

### 2. Train the BRNN model

```bash
python train.py --dataset DATASET_NAME
```

### 3. Export BRNN-driven scores

```bash
python paper_experiments/preprocessing/current_pipeline/infer_bnn_window_scores.py \
  --dataset-dir dataset/DATASET_NAME \
  --model-path save/DATASET_NAME/.../brnn-best
```

### 4. Run the latest paper sections

```bash
python paper_experiments/latest_release/sections/run_section41_setup.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/41 \
  --skip-missing
```

```bash
python paper_experiments/latest_release/sections/run_section42_detection.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/42 \
  --auto-tau \
  --skip-missing
```

```bash
python paper_experiments/latest_release/sections/run_section43_filter.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/43 \
  --auto-tau \
  --skip-missing
```

```bash
python paper_experiments/latest_release/sections/run_section44_sensitivity.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/44 \
  --detection-json paper_experiments/latest_release/results/42/42_detection_results.json \
  --skip-missing
```

```bash
python paper_experiments/latest_release/sections/run_section46_ablation.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/46 \
  --auto-tau \
  --skip-missing
```

## Notes

- Section 4.5 is still intentionally excluded here.
- `latest_release/results/` already contains the latest local outputs in this workspace.
- Historical directories were moved into `archive/` so the repo stays readable for GitHub visitors.
