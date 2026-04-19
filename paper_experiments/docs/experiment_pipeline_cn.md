# 实验 Pipeline 技术参考文档

> 英文版：[experiment_pipeline.md](experiment_pipeline.md)

本文档从头到尾说明完整的实验 pipeline：从原始 pcap 文件到论文中出现的图。面向希望理解或复现实验结果的 GitHub 读者。

---

## 总览

Pipeline 分为四个阶段：

```
原始 pcap  →  预处理  →  BRNN 推理  →  section 脚本  →  结果/图
```

每个阶段产生的文件由下一个阶段消费。所有实验相关代码均位于 `paper_experiments/` 下。项目根目录的模型代码（`train.py`、`model.py` 等）是独立部分，不属于此 pipeline 的管理范围。

---

## 阶段 1 — 原始流量预处理

**脚本：** `paper_experiments/preprocessing/current_pipeline/preprocess_traces.py`

**功能：**

读取 manifest 中列出的原始 pcap 文件，将其切分为固定时长的窗口，提取逐流的窗口记录，分配趋势标签（growing / damping），并将结构化 JSON/JSONL 文件写入 `dataset/<dataset_id>/json/`：

| 输出文件 | 内容 |
|---|---|
| `window_stream_all.jsonl` | 所有窗口级记录（item_id、window_id、freq、label） |
| `window_stream_train.jsonl` | 训练划分 |
| `window_stream_test.jsonl` | 测试划分 |
| `train.json` | 流级训练样本 |
| `test.json` | 流级测试样本 |
| `preprocess_overview.json` | 预处理元数据 |

**`manifest_template.json` 中的关键参数：**

| 参数 | 含义 |
|---|---|
| `window_seconds` | 每个滑动窗口的时长（如 0.1 秒） |
| `max_duration_seconds` | 使用 pcap 的时长上限（秒） |
| `max_packets` | 每个数据集的数据包上限 |
| `flow_key` | "flow" 的粒度（`5tuple`、`srcip`、`dstip`） |
| `G` / `D` | 趋势标签分配的增长/衰减阈值 |
| `rho2` | 分配标签所需的最少连续增长/衰减窗口数 |
| `min_flow_packets` | 纳入统计的最少数据包数 |
| `train_ratio` | 训练/测试划分比例 |

**运行：**
```bash
python paper_experiments/preprocessing/current_pipeline/preprocess_traces.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --skip-missing
```

---

## 阶段 2 — BRNN 分数注入

**脚本：** `paper_experiments/preprocessing/current_pipeline/infer_bnn_window_scores.py`

**功能：**

加载已训练的 BRNN 模型 checkpoint，对窗口级测试（以及可选的训练）流记录进行推理。将带有 `score` 字段的 `stream_records_bnn_test.jsonl` 和 `stream_records_bnn_train.jsonl` 写入数据集目录。

本步骤需要先完成模型训练：

```bash
python train.py --dataset <dataset_id>
```

然后注入分数：

```bash
python paper_experiments/preprocessing/current_pipeline/infer_bnn_window_scores.py \
  --dataset-dir dataset/<dataset_id> \
  --model-path save/<dataset_id>/<checkpoint>/brnn-best
```

section 脚本优先查找 `stream_records_bnn_test.jsonl`；若缺失，会回退到 `stream_records_proxy_test.jsonl` 或 `stream_records_test.jsonl`（后两者不含 BRNN 分数）。

---

## 阶段 3 — Section 脚本

所有 section 脚本位于 `paper_experiments/latest_release/sections/`。每个脚本对应论文中的一个小节，共同接受 `--manifest`、`--out-dir`、`--skip-missing` 参数。

### Section 4.1 — 数据集概览

**脚本：** `sections/run_section41_setup.py`

生成数据集 profile 表格和图：大小、类别均衡情况、窗口活跃度、频率分布、分数分离度。

```bash
python paper_experiments/latest_release/sections/run_section41_setup.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/41 \
  --skip-missing
```

输出：`41_dataset_table.png`、`41_dataset_size.png`、`41_class_balance.png`、`41_window_activity.png`、`41_frequency_distribution.png`、`41_score_separation.png`、`41_dataset_profiles.json`

---

### Section 4.2 — 检测精度

**脚本：** `sections/run_section42_detection.py`

在 1–40 KB 内存预算下评估各检测方法（Scout Manual、Scout Grid、NeuTrend-Filter、NeuTrend-Adaptive、NeuTrend Full）。

```bash
python paper_experiments/latest_release/sections/run_section42_detection.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/42 \
  --auto-tau \
  --skip-missing
```

关键参数：
- `--auto-tau` — 通过在训练记录上优化 F1 自动选择阈值 τ
- `--tau <float>` — 手动指定 τ（覆盖 auto-tau）

输出：`42_detection_results.json`、`42_f1_vs_memory.png`、`42_10kb_bar_f1.png`、`42_10kb_precision_recall.png`、`42_table1_10kb_current.png`

---

### Section 4.3 — Filter 分析（FPR vs 内存）

**脚本：** `sections/run_section43_filter.py`

在不同内存预算下比较标准 filter 与 learned filter 变体的假阳性率。

```bash
python paper_experiments/latest_release/sections/run_section43_filter.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/43 \
  --auto-tau \
  --skip-missing
```

输出：`43_filter_analysis.json`、`43_fpr_vs_memory.png`、`43_detector_occupancy.png`、`43_score_histograms.png`

---

### Section 4.4 — 参数敏感性

**脚本：** `sections/run_section44_sensitivity.py`

扫描关键超参数（G、D、rho2、tau、backup_ratio、backup_margin），展示 F1 对各参数的敏感程度。

```bash
python paper_experiments/latest_release/sections/run_section44_sensitivity.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/44 \
  --detection-json paper_experiments/latest_release/results/42/42_detection_results.json \
  --skip-missing
```

输出：`44_sensitivity_results.json`、`44_parameter_histograms.png`、`44_parameter_summary.png`

---

### Section 4.6 — 消融实验

**脚本：** `sections/run_section46_ablation.py`

对 NeuTrend 的关键组件（置信度替换、backup filter）逐一做消融，展示各组件的独立贡献。

```bash
python paper_experiments/latest_release/sections/run_section46_ablation.py \
  --manifest paper_experiments/latest_release/manifest_template.json \
  --out-dir paper_experiments/latest_release/results/46 \
  --auto-tau \
  --skip-missing
```

输出：`46_ablation_results.json`、`46_ablation_f1.png`、`46_ablation_precision_recall.png`

---

## 共享库：`common.py`

`paper_experiments/latest_release/common.py` 包含所有 section 脚本共用的辅助函数：

| 函数 / 类 | 用途 |
|---|---|
| `load_manifest()` | 解析 `manifest_template.json`，解析路径，验证数据集 |
| `load_json()` / `load_jsonl()` | 文件读写辅助 |
| `infer_score_polarity_and_tau_f1_opt()` | 从训练记录自动选择 τ 和分数方向 |
| `evaluate_learned_full_detection()` | 运行 NeuTrend Full sketch，返回 P/R/F1 指标 |
| `tune_learned_full_config()` | 网格搜索 backup_ratio / backup_margin / backup_min_hits |
| `compute_prf()` | 从预测集和真值集计算 Precision / Recall / F1 |
| `use_paper_plot_style()` | 为所有图统一应用 matplotlib 样式 |
| `get_shared_ablation_module()` | 每次会话加载一次 `runtime_support/run_ablation_46_v1.py` |
| `get_shared_filter_module()` | 加载一次 `runtime_support/run_filter_analysis_43_v9.py` |

---

## Runtime 支持模块

`paper_experiments/latest_release/runtime_support/` 包含底层 sketch 实现：

| 文件 | 提供内容 |
|---|---|
| `run_ablation_46_v1.py` | `DetectionSketch` — BOS sketch 核心（primary + backup filter、adaptive 模式、置信度替换） |
| `run_filter_analysis_43_v9.py` | `compute_filter_fpr()` — filter 变体的 FPR 分析 |
| `run_parameter_sensitivity_44_v1.py` | 参数扫描辅助函数 |

这些文件**不是**独立脚本，由 `common.py` 通过 `importlib` 动态加载，以避免循环导入，并让各 section 脚本保持自洽。

---

## Manifest 格式

`manifest_template.json` 中每个数据集对应一个条目。最小化示例：

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

可选的 `full_config_override` 允许对 NeuTrend Full 使用的 backup 参数进行按数据集调优：

```json
"full_config_override": {
  "backup_ratio": 0.8,
  "backup_margin": 4,
  "backup_min_hits": 2
}
```

若省略，pipeline 会在训练划分上自动调优这些参数。

---

## 接入新数据集

1. 获取一个 pcap 文件（或 `.pcap.gz`）。
2. 在 `manifest_template.json` 中添加一个条目，指定 pcap 路径和 `output_dir`。
3. 运行阶段 1 预处理，构建数据集目录。
4. 在新数据集上训练 BRNN，运行阶段 2 注入分数。
5. 携带 `--skip-missing` 运行各 section 脚本 — 新数据集会被自动纳入。

---

## 结果文件位置

当前工作区预生成的结果位于 `paper_experiments/latest_release/results/`：

```
results/
  41/    ← 数据集概览图和 profile JSON
  42/    ← 检测精度图和 JSON
  43/    ← filter 分析图和 JSON
  44/    ← 参数敏感性图和 JSON
  46/    ← 消融实验图和 JSON
```

这些是论文中引用的图。如需重新生成，请按顺序运行 section 脚本（4.1 → 4.2 → 4.3 → 4.4 → 4.6），并确保所有数据集的阶段 1 和阶段 2 均已完成。
