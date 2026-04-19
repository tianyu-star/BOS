# latest_release — 当前最新实验版本

> 英文版：[README.md](README.md)

本目录是当前论文实验 pipeline 面向读者的整洁版本，是 GitHub 访客应该首先打开的代码路径。

## 目录结构

- **`common.py`**
  — 共享辅助库：manifest 加载、指标计算、绘图工具、runtime 模块动态加载。

- **`manifest_template.json`**
  — 多数据集实验的配置入口，定义每个数据集的路径、分窗参数、趋势标签参数等。

- **`sections/`**
  — Section 4.1、4.2、4.3、4.4、4.6 的最新可运行脚本，每个文件对应论文中的一个实验小节。

- **`runtime_support/`**
  — 最新 section 脚本复用的底层 runtime 实现（从旧目录抽取出来，让当前 pipeline 自洽，不依赖归档目录）。

- **`results/`**
  — 当前工作区最新导出的实验结果和图，按 section id 分组存放。

## 推荐阅读顺序

1. `sections/run_section41_setup.py`
2. `sections/run_section42_detection.py`
3. `sections/run_section43_filter.py`
4. `sections/run_section44_sensitivity.py`
5. `sections/run_section46_ablation.py`
6. `results/`

## 当前结果目录

| 目录 | 内容 |
|---|---|
| `results/41/` | 数据集概览图和 profile JSON |
| `results/42/` | 检测精度图和 JSON |
| `results/42_smoke/` | 快速验证用的 42 节结果 |
| `results/43/` | Filter 分析图和 JSON |
| `results/44/` | 参数敏感性图和 JSON |
| `results/46/` | 消融实验图和 JSON |

## 为什么单独存在这个目录

项目中保存了大量实验迭代痕迹。对于 GitHub 读者，本目录中的文件是唯一需要阅读、运行和对比最新图的代码路径。`archive/` 中保留了历史迭代供追溯，但正常阅读不需要看那里。
