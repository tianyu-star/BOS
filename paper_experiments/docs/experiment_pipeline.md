# Experiment Pipeline — Technical Reference

> 中文版：[experiment_pipeline_cn.md](experiment_pipeline_cn.md)

This document explains the full experiment pipeline end-to-end, from raw pcap files to the figures that appear in the paper. It is written for GitHub readers who want to understand or reproduce the results.

---

## Overview

The pipeline has four stages:

```
raw pcap  →  preprocess  →  BRNN inference  →  section scripts  →  results/figures
```

Each stage produces files consumed by the next. All experiment-related code lives under `paper_experiments/`. The model code at the project root (`train.py`, `model.py`, etc.) is a separate concern.

---

## Stage 1 — Raw Trace Preprocessing

**Script:** `paper_experiments/preprocessing/current_pipeline/preprocess_traces.py`

**What it does:**

Reads raw pcap files listed in the manifest, slices them into fixed-duration windows, extracts per-flow window records, assigns trending labels (growing / damping), and writes structured JSON/JSONL files under `dataset/<dataset_id>/json/`:

| Output file | Contents |
|---|---|
| `window_stream_all.jsonl` | All window-level records (item_id, window_id, freq, label) |
| `window_stream_train.jsonl` | Training split |
| `window_stream_test.jsonl` | Test split |
| `train.json` | Flow-level training samples |
| `test.json` | Flow-level test samples |
| `preprocess_overview.json` | Preprocessing metadata |

**Key parameters (in `manifest_template.json`):**

| Parameter | Meaning |
|---|---|
| `window_seconds` | Duration of each sliding window (e.g. 0.1 s) |
| `max_duration_seconds` | How many seconds of the pcap to use |
| `max_packets` | Packet cap per dataset |
| `flow_key` | Granularity of a "flow" (`5tuple`, `srcip`, `dstip`) |
| `G` / `D` | Growth / decay thresholds for trending label assignment |
| `rho2` | Minimum consecutive growth/decay windows to assign a label |
| `min_flow_packets` | Minimum packets for a flow to be included |
| `train_ratio` | Train/test split ratio |

**Run:**
```bash
python paper_experiments/preprocessing/current_pipeline/preprocess_traces.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --skip-missing
```

---

## Stage 2 — BRNN Score Injection

**Script:** `paper_experiments/preprocessing/current_pipeline/infer_bnn_window_scores.py`

**What it does:**

Loads a trained BRNN model checkpoint and runs inference on the window-level test (and optionally train) stream records. Writes `stream_records_bnn_test.jsonl` and `stream_records_bnn_train.jsonl` with a `score` field added to each record.

This step requires a trained model. Train one first:

```bash
python train.py --dataset <dataset_id>
```

Then infer scores:

```bash
python paper_experiments/preprocessing/current_pipeline/infer_bnn_window_scores.py \
  --dataset-dir dataset/<dataset_id> \
  --model-path save/<dataset_id>/<checkpoint>/brnn-best
```

The section scripts look for `stream_records_bnn_test.jsonl` first. If it is missing, they fall back to `stream_records_proxy_test.jsonl` or `stream_records_test.jsonl` (which lack BRNN scores).

---

## Stage 3 — Section Scripts

All section scripts live in `paper_experiments/latest_release/sections/`. Each corresponds to a paper section and accepts `--manifest`, `--out-dir`, and `--skip-missing` as common arguments.

### Section 4.1 — Dataset Overview

**Script:** `sections/run_section41_setup.py`

Generates dataset profile tables and plots: size, class balance, window activity, frequency distribution, score separation.

```bash
python paper_experiments/latest_release/sections/run_section41_setup.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/41 \
  --skip-missing
```

Outputs: `41_dataset_table.png`, `41_dataset_size.png`, `41_class_balance.png`, `41_window_activity.png`, `41_frequency_distribution.png`, `41_score_separation.png`, `41_dataset_profiles.json`

---

### Section 4.2 — Detection Accuracy

**Script:** `sections/run_section42_detection.py`

Evaluates detection methods (Scout Manual, Scout Grid, NeuTrend-Filter, NeuTrend-Adaptive, NeuTrend Full) across memory budgets (1–40 KB).

```bash
python paper_experiments/latest_release/sections/run_section42_detection.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/42 \
  --auto-tau \
  --skip-missing
```

Key flags:
- `--auto-tau` — automatically select τ threshold by F1 optimization on training records
- `--tau <float>` — manually set τ (overrides auto-tau)

Outputs: `42_detection_results.json`, `42_f1_vs_memory.png`, `42_10kb_bar_f1.png`, `42_10kb_precision_recall.png`, `42_table1_10kb_current.png`

---

### Section 4.3 — Filter Analysis (FPR vs Memory)

**Script:** `sections/run_section43_filter.py`

Compares the false positive rate of standard filter vs. learned filter variants across memory budgets.

```bash
python paper_experiments/latest_release/sections/run_section43_filter.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/43 \
  --auto-tau \
  --skip-missing
```

Outputs: `43_filter_analysis.json`, `43_fpr_vs_memory.png`, `43_detector_occupancy.png`, `43_score_histograms.png`

---

### Section 4.4 — Parameter Sensitivity

**Script:** `sections/run_section44_sensitivity.py`

Sweeps key hyperparameters (G, D, rho2, tau, backup_ratio, backup_margin) and shows F1 sensitivity.

```bash
python paper_experiments/latest_release/sections/run_section44_sensitivity.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/44 \
  --detection-json paper_experiments/latest_release/results/42/42_detection_results.json \
  --skip-missing
```

Outputs: `44_sensitivity_results.json`, `44_parameter_histograms.png`, `44_parameter_summary.png`

---

### Section 4.6 — Ablation Study

**Script:** `sections/run_section46_ablation.py`

Ablates key components of NeuTrend (confidence replacement, backup filter) to show their individual contributions.

```bash
python paper_experiments/latest_release/sections/run_section46_ablation.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/46 \
  --auto-tau \
  --skip-missing
```

Outputs: `46_ablation_results.json`, `46_ablation_f1.png`, `46_ablation_precision_recall.png`

---

## Shared Library: `common.py`

`paper_experiments/latest_release/common.py` contains all shared helpers used by the section scripts:

| Function / class | Purpose |
|---|---|
| `load_manifest()` | Parse `manifest_template.json`, resolve paths, validate datasets |
| `load_json()` / `load_jsonl()` | File I/O helpers |
| `infer_score_polarity_and_tau_f1_opt()` | Auto-select τ and score direction from training records |
| `evaluate_learned_full_detection()` | Run NeuTrend Full sketch and return P/R/F1 metrics |
| `tune_learned_full_config()` | Grid-search backup_ratio / backup_margin / backup_min_hits |
| `compute_prf()` | Precision / Recall / F1 from predicted and ground-truth sets |
| `use_paper_plot_style()` | Apply consistent matplotlib style for all figures |
| `get_shared_ablation_module()` | Load `runtime_support/run_ablation_46_v1.py` once per session |
| `get_shared_filter_module()` | Load `runtime_support/run_filter_analysis_43_v9.py` once |

---

## Runtime Support Modules

`paper_experiments/latest_release/runtime_support/` contains the underlying sketch implementations:

| File | Provides |
|---|---|
| `run_ablation_46_v1.py` | `DetectionSketch` — the core BOS sketch (primary + backup filter, adaptive mode, confidence replacement) |
| `run_filter_analysis_43_v9.py` | `compute_filter_fpr()` — FPR analysis for filter variants |
| `run_parameter_sensitivity_44_v1.py` | Parameter sweep helpers |

These are **not** standalone scripts. They are loaded dynamically by `common.py` using `importlib` to avoid circular imports and keep the section scripts self-contained.

---

## Manifest Format

`manifest_template.json` defines one entry per dataset. A minimal entry:

```json
{
  "dataset_id": "my_dataset",
  "display_name": "My Dataset Label",
  "source": "MAWI",
  "raw_files": ["../path/to/capture.pcap"],
  "output_dir": "../dataset/my_dataset",
  "flow_key": "5tuple",
  "window_seconds": 0.1,
  "max_duration_seconds": 3.0,
  "max_packets": 300000,
  "min_flow_packets": 9,
  "train_ratio": 0.8,
  "seed": 7,
  "G": 1.2,
  "D": 0.8,
  "rho2": 3
}
```

Optional `full_config_override` allows per-dataset tuning of backup parameters used by NeuTrend Full:

```json
"full_config_override": {
  "backup_ratio": 0.8,
  "backup_margin": 4,
  "backup_min_hits": 2
}
```

If omitted, the pipeline auto-tunes these values on the training split.

---

## Adding a New Dataset

1. Obtain a pcap file (or `.pcap.gz`).
2. Add an entry to `manifest_template.json` pointing at the pcap and choosing an `output_dir`.
3. Run Stage 1 preprocessing to build the dataset directory.
4. Train BRNN on the new dataset and run Stage 2 to inject scores.
5. Run the section scripts with `--skip-missing` — they will include the new dataset automatically.

---

## Where the Results Are

Pre-generated results from the current workspace are in `paper_experiments/latest_release/results/`:

```
results/
  41/    ← dataset overview figures and profile JSON
  42/    ← detection accuracy figures and JSON
  43/    ← filter analysis figures and JSON
  44/    ← sensitivity figures and JSON
  46/    ← ablation figures and JSON
```

These are the figures referenced in the paper. To regenerate them, run the section scripts in order (4.1 → 4.2 → 4.3 → 4.4 → 4.6), making sure Stage 1 and Stage 2 have already completed for all datasets.
