# Jittor Cora Node Classification Warmup

本仓库为计图挑战赛赛道一热身赛开源代码。任务是在基于 Cora 的引文网络上进行节点分类，并为测试集节点生成 `result.json`。

当前 `result.json` 对应已提交打榜版本，线上准确率为 0.839。

## 环境安装

推荐环境：

- Python 3.12
- Jittor 1.3.10.0
- jittor-geometric 2.0.0

安装依赖：

```bash
pip install -r requirements.txt
```

如果在比赛远程环境 `/workspace/JDJT` 中运行，可直接使用已经配置好的环境：

```bash
cd /workspace/JDJT/warmup-01
USE_CUDA=1 bash run_gcn.sh --config configs/default.json
```

## 数据准备

本仓库不提交原始数据文件。请将比赛发布包中的数据放到：

```text
data/cora.pkl
```

数据字段包括 `x`、`y`、`edge_index`、`train_mask`、`val_mask`、
`test_mask`、`num_classes` 和 `num_features`。测试集标签在 `y` 中为 `-1`。

## 训练

默认训练命令：

```bash
USE_CUDA=1 bash run_gcn.sh --config configs/default.json
```

脚本默认读取 `data/cora.pkl`，训练 GCNII 模型，并将预测结果写入
`result.json`。主要参数在 `configs/default.json` 中定义：随机种子为
`42`，隐藏层维度为 `64`，层数为 `64`，训练最多 `2000` 轮，并使用验证集早停。

也可以用命令行覆盖配置，例如：

```bash
bash run_gcn.sh --config configs/default.json --epochs 10 --output outputs/debug_result.json
```

如果当前环境没有可用 GPU，可使用 CPU smoke test 检查数据读取、
训练和结果写出流程：

```bash
USE_CUDA=0 bash run_gcn.sh \
  --config configs/default.json \
  --model appnp_cpu \
  --use-cuda 0 \
  --epochs 1 \
  --log-interval 1 \
  --output outputs/smoke_result.json
```

该 CPU 后端仅用于无 GPU 环境验证流程，线上 0.839 结果对应默认 GCNII 配置。

## 评测与推理

训练结束后脚本会生成测试集预测文件：

```text
result.json
```

本热身赛不需要单独加载 checkpoint；推理在训练结束后直接对测试集节点生成预测。若只需要生成提交压缩包，可运行：

```bash
python zip_results.py
```

生成的 `result.zip` 包含 `gcn.py` 和 `result.json`，符合热身赛提交格式。

## 结果说明

线上评测指标为测试集分类准确率 Accuracy。由于测试集真实标签隐藏，本地日志仅输出训练集和验证集准确率；最终成绩以线上评测为准。

仓库中的 `result.json` 是当前提交预测结果，包含 1000 个测试节点预测。该文件来自比赛打榜提交包中的结果文件。

## 开源说明

- 不提交 `data/cora.pkl`、模型权重、日志和 `result.zip`。
- 第三方依赖包括 Jittor、JittorGeometric 和 NumPy。
- 代码基于比赛提供的热身赛框架完成，主要模型为 GCNII。
