# Preprocessing

> 中文版：[README_CN.md](README_CN.md)

This folder groups all preprocessing and data-transformation code used by the experiment pipeline.

## Structure

- `current_pipeline/`
  - Current preprocessing scripts used by the latest paper experiments.
  - `preprocess_traces.py`: build dataset files from raw traces.
  - `infer_bnn_window_scores.py`: inject BRNN scores into window-level stream records.
- `score_builders/`
  - Historical and current score-export helpers used during preprocessing experiments.
- `raw_trace_tools/`
  - Older raw-trace conversion utilities kept for reproducibility.

## What Most Users Need

Open `current_pipeline/` first.

The other two subfolders are mainly for:

- reproducing older preprocessing behavior
- understanding how the score-export path evolved
- reusing raw-trace conversion snippets when adding a new dataset
