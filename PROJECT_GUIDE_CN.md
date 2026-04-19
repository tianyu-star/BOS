# BOS 项目上手说明


## 1. 这个项目里最重要的几块内容

### 主模型代码

这一部分保持原目录不动，主要负责训练和推理二值 RNN / BRNN 模型。

重点文件：

- `train.py`
- `train2.py`
- `trainer.py`
- `model.py`
- `opts.py`
- `aggregator.py`
- `utils/`
- `dataset/`
- `save/`

如果你的目标是理解模型本身，先看这部分。

### 最新论文实验代码

这一部分已经整理到：

- `paper_experiments/latest_release/`

这里是现在最值得看的实验代码，包含：

- Section 4.1 数据集概览
- Section 4.2 detection accuracy
- Section 4.3 filter analysis
- Section 4.4 parameter sensitivity
- Section 4.6 ablation

同时这里也放着最新结果：

- `paper_experiments/latest_release/results/`

如果你的目标是复现实验、看最新图、理解论文里每个实验怎么跑，先看这里。

### 数据预处理代码

这一部分已经整理到：

- `paper_experiments/preprocessing/`

其中：

- `current_pipeline/` 是当前仍在使用的预处理流程
- `score_builders/` 是分数构造/导出相关脚本
- `raw_trace_tools/` 是更早期的原始 trace 处理脚本

如果你后面要接入新数据源，主要会看这里。

### 历史迭代代码

为了保留实验演化痕迹，但又不影响新读者理解，历史代码被统一归档到：

- `paper_experiments/archive/`

平时不建议先看这里，除非你要：

- 对比老版本和新版本实验逻辑
- 找历史图/历史结果
- 追溯某个实验设计的演化过程

## 2. 推荐阅读顺序

### 如果你想先理解整个项目

1. 先看根目录 `README.md`
2. 再看 `paper_experiments/README.md`
3. 然后看 `paper_experiments/latest_release/README.md`

### 如果你想先看最新实验

建议顺序：

1. `paper_experiments/latest_release/sections/run_section41_setup.py`
2. `paper_experiments/latest_release/sections/run_section42_detection.py`
3. `paper_experiments/latest_release/sections/run_section43_filter.py`
4. `paper_experiments/latest_release/sections/run_section44_sensitivity.py`
5. `paper_experiments/latest_release/sections/run_section46_ablation.py`
6. `paper_experiments/latest_release/results/`

### 如果你要接入新数据源

建议顺序：

1. `paper_experiments/preprocessing/current_pipeline/preprocess_traces.py`
2. `paper_experiments/latest_release/manifest_template.json`
3. `paper_experiments/preprocessing/current_pipeline/infer_bnn_window_scores.py`
4. `paper_experiments/latest_release/sections/` 中对应实验脚本

## 3. 当前最新实验目录怎么理解

`paper_experiments/latest_release/` 里有 4 类东西：

### `sections/`

每个 `run_sectionXX_*.py` 都对应论文里的一个实验小节。

### `runtime_support/`

这是最新实验脚本依赖的底层 runtime 实现。它们从旧目录里抽出来单独放在这里，目的是让最新实验目录本身自洽，不再依赖归档目录。

### `results/`

这里放的是当前工作区里最新导出的实验结果和图。

### `manifest_template.json`

这个文件定义了实验所需的数据集配置，是整个多数据集实验流水线的入口配置之一。

## 4. 最常见的工作流

### 场景 A：只看结果，不跑实验

直接看：

- `paper_experiments/latest_release/results/`

### 场景 B：想复现实验

1. 先准备数据：`paper_experiments/preprocessing/current_pipeline/preprocess_traces.py`
2. 如果需要模型分数，先训练主模型，再跑 `infer_bnn_window_scores.py`
3. 再运行 `latest_release/sections/` 中对应 section 的脚本
4. 到 `latest_release/results/` 看结果

### 场景 C：想接新的数据源

重点改这几类地方：

- `paper_experiments/latest_release/manifest_template.json`
- `paper_experiments/preprocessing/current_pipeline/`
- 必要时补充 `paper_experiments/preprocessing/raw_trace_tools/`

一般不要先改归档目录。

## 5. 这次整理后的目录原则

这次整理遵循 3 个原则：

1. 主模型代码不动，避免影响训练/推理主线。
2. 最新实验代码和最新实验结果放在一起，方便 GitHub 阅读。
3. 历史迭代代码单独归档，减少目录噪音，但保留可追溯性。

## 6. 你现在最应该看哪里

如果你是第一次接手这个项目，最推荐从这里开始：

- `paper_experiments/latest_release/README.md`
- `paper_experiments/latest_release/sections/`
- `paper_experiments/latest_release/results/`
- `paper_experiments/preprocessing/current_pipeline/`

读完这几块，基本就能掌握项目的主实验流程。
