# 论文实验目录说明

> 英文版：[README.md](README.md)

`paper_experiments/` 是本项目所有实验相关代码的统一入口。

## 目录结构

- **`latest_release/`**
  — 当前最新的 Section 4.x 实验代码，应优先阅读。
  包含最新可运行的 section 脚本、共享辅助库、runtime 支持模块、manifest 模板，以及最新生成的实验结果。

- **`preprocessing/`**
  — 数据预处理与分数构建工具。
  包含当前预处理流程和用于可复现性保留的旧版 raw-trace / score-builder 脚本。

- **`archive/`**
  — 历史实验迭代、旧版实验脚本和中间产物，不再是主要代码路径。

- **`docs/`**
  — 技术参考文档。

## 新读者推荐阅读顺序

1. `latest_release/README_CN.md`
2. `latest_release/sections/`
3. `latest_release/results/`
4. `docs/experiment_pipeline_cn.md` — 完整分阶段技术说明（中文）
5. `../PROJECT_GUIDE_CN.md` — 新人上手指南（中文）

## 快速启动

### 第 1 步：构建数据集文件

```bash
python paper_experiments/preprocessing/current_pipeline/preprocess_traces.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --skip-missing
```

### 第 2 步：训练 BRNN 模型

```bash
python train.py --dataset DATASET_NAME
```

### 第 3 步：导出 BRNN 驱动的分数

```bash
python paper_experiments/preprocessing/current_pipeline/infer_bnn_window_scores.py \
  --dataset-dir dataset/DATASET_NAME \
  --model-path save/DATASET_NAME/.../brnn-best
```

### 第 4 步：运行各 section 实验脚本

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

## 备注

- Section 4.5 目前不包含在本目录中。
- `latest_release/results/` 已包含当前工作区最新的本地导出结果。
- 历史目录已移入 `archive/`，以保持主目录对 GitHub 读者的可读性。
  目前目录中不包含源数据文件，因为源数据文件太大，需要自行去MAWI网站上下载，然后给数据预处理的代码文件处理。
