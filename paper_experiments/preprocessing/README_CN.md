# preprocessing — 数据预处理

> 英文版：[README.md](README.md)

本目录汇总了实验 pipeline 使用的所有预处理和数据变换代码。

## 目录结构

- **`current_pipeline/`**
  — 当前最新论文实验使用的预处理脚本，大多数情况下只需要看这里。
  - `preprocess_traces.py` — 从原始 pcap 构建数据集文件（窗口切片、趋势标签、train/test 划分）
  - `infer_bnn_window_scores.py` — 将 BRNN 模型输出的分数注入到 window-level stream records 中

- **`score_builders/`**
  — 预处理实验过程中使用过的历史和当前分数导出辅助脚本，按版本保留用于可复现性。

- **`raw_trace_tools/`**
  — 更早期的 raw trace 转换工具，接入新数据源时可参考。

## 大多数用户需要看的部分

直接打开 `current_pipeline/`。

其余两个子目录主要用于：

- 复现更早期的预处理行为
- 了解分数导出路径的演化过程
- 在接入新数据集时复用 raw trace 转换片段
