# BOS

> 中文版：[README_CN.md](README_CN.md)

Network trend detection using Binary RNN inference on programmable switches.

## Quick Navigation

| What you want | Where to look |
|---|---|
| Latest paper experiment code | `paper_experiments/latest_release/sections/` |
| Latest experiment results / figures | `paper_experiments/latest_release/results/` |
| Data preprocessing pipeline | `paper_experiments/preprocessing/current_pipeline/` |
| Full pipeline technical reference (EN) | `paper_experiments/docs/experiment_pipeline.md` |
| Newcomer guide (中文) | `PROJECT_GUIDE_CN.md` |
| Historical experiment iterations | `paper_experiments/archive/` |

## Model Code

The BRNN model code stays at the project root:

| File | Purpose |
|---|---|
| `model.py` | Binary RNN architecture |
| `opts.py` | Hyperparameter configuration |
| `train.py` / `train2.py` | Training entry points |
| `trainer.py` | Training loop |
| `aggregator.py` | Confidence aggregation and CDF export |
| `model_convertion.ipynb` | Convert model to P4 feed-forward tables |

### Train a model

```bash
python train.py --dataset DATASET_NAME
```

Checkpoints and training logs are saved to `save/DATASET_NAME/`.

### Determine confidence thresholds

```bash
python aggregator.py --dataset DATASET_NAME
```

The CDF of packet confidence is saved to `save/DATASET_NAME/` and used to set analysis-escalation thresholds.

### Convert to P4 feed-forward tables

Run `model_convertion.ipynb`. Tables are saved to `../p4/parameter/DATASET_NAME/`.

## Running the Paper Experiments

See `paper_experiments/README.md` for the complete quick-start commands, or read `paper_experiments/docs/experiment_pipeline.md` for a full stage-by-stage technical walkthrough.
