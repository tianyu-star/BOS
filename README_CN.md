# BOS — Brain-on-Switch

基于可编程交换机上二值 RNN 推理的网络趋势检测系统。

> 英文版：[README.md](README.md)

## 快速导航

| 你想看的内容 | 对应位置 |
|---|---|
| 最新论文实验代码 | `paper_experiments/latest_release/sections/` |
| 最新实验结果 / 图 | `paper_experiments/latest_release/results/` |
| 数据预处理流程 | `paper_experiments/preprocessing/current_pipeline/` |
| 完整 pipeline 技术说明（英文） | `paper_experiments/docs/experiment_pipeline.md` |
| 完整 pipeline 技术说明（中文） | `paper_experiments/docs/experiment_pipeline_cn.md` |
| 上手指南（中文） | `PROJECT_GUIDE_CN.md` |
| 历史实验迭代代码 | `paper_experiments/archive/` |

## 主模型代码

模型代码保持在项目根目录不动：

| 文件 | 用途 |
|---|---|
| `model.py` | 二值 RNN 模型结构 |
| `opts.py` | 超参数配置（每个数据集的默认值见文件内注释） |
| `train.py` / `train2.py` | 训练入口 |
| `trainer.py` | 训练主循环 |
| `aggregator.py` | 置信度聚合与 CDF 导出 |
| `model_convertion.ipynb` | 将模型转换为 P4 前向表 |

### 训练模型

```bash
python train.py --dataset DATASET_NAME
```

模型 checkpoint 和训练日志保存到 `save/DATASET_NAME/`。

### 确定置信度阈值

```bash
python aggregator.py --dataset DATASET_NAME
```

数据包置信度的 CDF 保存到 `save/DATASET_NAME/`，用于确定分析升级的置信度阈值。

### 转换为 P4 前向表

运行 `model_convertion.ipynb`，转换结果保存到 `../p4/parameter/DATASET_NAME/`。

## 运行论文实验

完整的快速启动命令见 `paper_experiments/README.md`。

完整的分阶段技术说明见 `paper_experiments/docs/experiment_pipeline_cn.md`。

目前实验结果的数据源文件为caida_1000w.pcap 202410231400.pcap 202511261400.pcap 202603061400.pcap, 后面三个都是MAWI网站上https://mawi.wide.ad.jp/mawi/的数据包，分别为2024年10月23日，2025年11月26日和2026年3月6日的。
