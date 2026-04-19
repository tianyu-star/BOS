# Latest Release

> 中文版：[README_CN.md](README_CN.md)

This folder is the clean, reader-facing version of the current paper experiment pipeline.

## Structure

- `common.py`
  - Shared helpers for manifest loading, evaluation, plotting, and runtime module loading.
- `manifest_template.json`
  - Template manifest for multi-dataset experiments.
- `sections/`
  - Latest runnable scripts for Sections 4.1, 4.2, 4.3, 4.4, and 4.6.
- `runtime_support/`
  - Runtime implementations reused by the latest section runners.
  - These are copied out of older experiment folders so the latest pipeline does not depend on archived directories.
- `results/`
  - Latest generated JSON summaries and figures, grouped by section id.

## Recommended Reading Order

1. `sections/run_section41_setup.py`
2. `sections/run_section42_detection.py`
3. `sections/run_section43_filter.py`
4. `sections/run_section44_sensitivity.py`
5. `sections/run_section46_ablation.py`
6. `results/`

## Current Results

- `results/41/`
- `results/42/`
- `results/42_smoke/`
- `results/43/`
- `results/44/`
- `results/46/`

## Why This Folder Exists

The repo contains many experiment iterations. For GitHub readers, the files in this folder are the primary code path to read, run, and compare against the latest figures/results.
