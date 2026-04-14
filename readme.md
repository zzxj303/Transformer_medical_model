# 项目架构与复现说明

## 1. 项目简介

本项目实现了一个基于 **Transformer** 的多任务二分类框架，用于对结构化医学特征进行建模，并同时预测多个标签任务。当前仓库中保留的是 **非影像（Non-Img）版本** 的训练流程，即仅使用 CSV 中的表型、临床和人口学等结构化特征，不启用 MRI 图像分支。

从代码实现来看，本项目的核心特点包括：

- 以 `CSV` 文件作为输入数据源
- 通过 `TOML` 配置文件定义特征和标签的模态信息
- 使用 `Transformer Encoder` 对多模态结构化输入进行统一建模
- 支持多任务二分类输出
- 训练时使用 `Focal Loss`
- 支持可选的 `ranking loss`
- 支持训练集中的随机特征缺失模拟与缺失值处理
- 支持保存最佳 checkpoint 与中间 checkpoint

---

## 2. 项目目录结构

```text
project/
├─ adrd/
│  ├─ model/
│  │  ├─ __init__.py
│  │  └─ adrd_model.py
│  ├─ nn/
│  │  ├─ __init__.py
│  │  ├─ transformer.py
│  │  ├─ focal_loss.py
│  │  └─ net_resnet3d.py
│  └─ utils/
│     ├─ __init__.py
│     ├─ formatter.py
│     ├─ imputer.py
│     ├─ masker.py
│     ├─ transformer_dataset.py
│     └─ misc.py
│
├─ dev/
│  ├─ train.py
│  ├─ train.sh
│  └─ data/
│     ├─ dataset_csv.py
│     └─ toml_files/
│        └─ default_conf_new.toml
│
├─ testdata/
│  └─ nonimg_smoke/
│     ├─ smoke_all.csv
│     ├─ smoke_train.csv
│     ├─ smoke_vld.csv
│     └─ smoke_test.csv
│
└─ .vscode/
```

---

## 3. 整体架构设计

### 3.1 总体流程

```text
CSV 数据
  ↓
CSVDataset 读取与清洗
  ↓
根据 TOML 提取 feature / label 定义
  ↓
构造 features / labels 字典列表
  ↓
TransformerTrainingDataset / ValidationDataset
  ↓
缺失值填补 + mask 生成
  ↓
Transformer 编码
  ↓
多任务二分类输出
  ↓
Focal Loss / Ranking Loss
  ↓
验证集评估并保存最佳模型
```

---

## 4. 核心模块说明

### 4.1 `dev/train.py`：训练主入口

该脚本负责：

- 解析命令行参数
- 加载训练集、验证集、测试集
- 读取配置文件
- 构建 `ADRDModel`
- 启动训练流程

当前代码中已经明确限制：

- `--img_mode` 只能为 `-1`
- `--img_net` 只能为 `NonImg`

这说明当前仓库实际可运行的是 **非影像结构化特征版本**。

训练入口中会分别实例化：

- `dat_trn = CSVDataset(...)`
- `dat_vld = CSVDataset(...)`
- `dat_tst = CSVDataset(...)`

然后将训练集和验证集传入：

```python
mdl.fit(
    dat_trn.features,
    dat_vld.features,
    dat_trn.labels,
    dat_vld.labels,
    img_train_trans=None,
    img_vld_trans=None,
    img_mode=args.img_mode,
)
```

需要注意的是，当前 `train.py` 中虽然读取了测试集，但主流程里并没有在训练结束后自动执行测试评估。

---

### 4.2 `dev/data/dataset_csv.py`：数据读取与预处理

`CSVDataset` 是项目的数据入口，主要负责：

- 读取 CSV 文件
- 读取 TOML 配置文件
- 检查特征列和标签列是否存在
- 删除配置里存在但数据里缺失的字段
- 删除所有特征全缺失的样本
- 删除所有标签全缺失的样本
- 对若干类别字段做字符串到整数的映射
- 将每个样本构造成字典形式的 `features` 和 `labels`

最终形成：

- `self.features`：每个样本的输入特征字典
- `self.labels`：每个样本的标签字典
- `self.label_fractions`：每个标签中正样本占比

这一层是模型训练前最重要的数据清洗步骤。

---

### 4.3 `adrd/utils/transformer_dataset.py`：训练/验证/测试数据集

该模块在 `CSVDataset` 输出的基础上进一步处理训练输入，核心职责包括：

- 将原始样本格式化为模型可接收的形式
- 对缺失值进行填补
- 生成输入特征 mask
- 生成标签 mask
- 在训练阶段对输入做随机 dropout 模拟，提高鲁棒性

其中几个主要数据集类如下：

- `TransformerTrainingDataset`
  - 用于训练
  - 使用 `FrequencyImputer` 对输入缺失值进行填补
  - 使用 `DropoutMasker` 做训练时随机缺失模拟

- `TransformerValidationDataset`
  - 用于验证
  - 使用 `ConstantImputer`
  - 使用 `MissingMasker` 按真实缺失情况生成 mask

- `TransformerTestingDataset`
  - 用于推理测试
  - 不依赖标签

此外还实现了：

- `TransformerBalancedTrainingDataset`
- `Transformer2ndOrderBalancedTrainingDataset`

用于平衡采样训练。

---

### 4.4 `adrd/nn/transformer.py`：模型主体

模型主体是一个面向结构化输入的 Transformer 编码器，其设计流程如下。

#### 输入嵌入层

不同类型特征采用不同嵌入方式：

- `categorical` 特征使用 `Embedding`
- `numerical` 特征使用 `BatchNorm1d + Linear`
- `imaging` 特征当前非影像模式下不会启用

#### 位置编码

对所有输入特征 embedding 加入位置编码，使 Transformer 能区分不同特征槽位。

#### 目标任务辅助 token

模型为每个目标标签任务维护一个可学习的辅助 embedding，作为任务查询 token，与输入特征 token 一起送入 Transformer。

#### Transformer 编码

使用 `torch.nn.TransformerEncoder` 对特征 token 和任务 token 进行联合建模。

#### 分类头

每个标签任务对应一个独立的线性层：

```python
self.modules_cls[k] = torch.nn.Linear(d_model, 1)
```

因此这是一个 **多任务二分类** 框架，每个标签输出一个 logit。

---

### 4.5 `adrd/model/adrd_model.py`：训练与验证主逻辑

`ADRDModel` 是项目的训练封装核心，主要负责：

- 初始化网络
- 构造 dataloader
- 配置优化器与学习率调度器
- 执行训练循环
- 执行验证循环
- 根据验证集指标保存最佳模型
- 提供 `predict_logits`、`predict_proba`、`predict` 接口

#### 损失函数

默认使用 `SigmoidFocalLoss`，并根据标签正样本比例动态设置 `alpha`。

若启用 `ranking_loss`，则在训练前若干 epoch 后额外加入排序损失。

#### 优化器与调度器

- 优化器：`AdamW`
- 学习率调度器：`CosineAnnealingWarmRestarts`

#### 模型选择指标

当前训练脚本中使用：

```python
criterion="AUC (ROC)"
```

即默认按照验证集平均 `AUC (ROC)` 保存最佳 checkpoint。

如果启用了 `--save_intermediate_ckpts`，还会额外保存按 `AUC (PR)` 最优的模型。

---

### 4.6 `adrd/utils/misc.py`：评估指标计算

该模块实现了多任务分类指标计算，包括：

- Accuracy
- Balanced Accuracy
- Precision
- Sensitivity / Recall
- Specificity
- F1 score
- MCC
- AUC (ROC)
- AUC (PR)

其中预测标签是通过下面的规则得到的：

```python
y_pred_vld = (scores_vld > 0).to(torch.int)
```

也就是对 logit 以 `0` 为阈值分类，等价于对 sigmoid 概率以 `0.5` 为阈值分类。

需要注意，若模型把所有样本都预测成同一类，则会出现：

- `Precision` 分母为 0
- `F1` 无法定义
- `MCC` 分母为 0

这时打印时会将 `nan` 显示为 `------`。

---

## 5. 配置文件说明

项目通过 `dev/data/toml_files/default_conf_new.toml` 定义输入特征和输出标签。

当前仓库中的示例配置为：

- 特征：
  - `his_NACCREAS`：类别型，3 类
  - `bat_MMSECOMP`：类别型，2 类
  - `his_NACCAGE`：数值型
  - `bat_NACCMMSE`：数值型

- 标签：
  - `NC`：二分类
  - `DE`：二分类

这说明当前 smoke 示例是一个 **4 个输入特征、2 个输出任务** 的最小可复现实验。

---

## 6. 运行环境建议

根据代码依赖，建议的 Python 环境至少包含以下包：

```bash
python
torch
pandas
numpy
scikit-learn
scipy
toml
tqdm
icecream
wandb
```

如果使用 `dev/train.sh`，还需要准备好对应的 `conda` 环境，例如脚本中使用的是：

```bash
conda activate adrd
```

如果不使用 `wandb`，训练也可以运行，因为代码中在未开启时会使用 disabled 模式。

---

## 7. 数据准备说明

当前仓库自带了一个最小复现数据集，位于：

```text
testdata/nonimg_smoke/
```

包含：

- `smoke_all.csv`
- `smoke_train.csv`
- `smoke_vld.csv`
- `smoke_test.csv`

这些文件可直接用于验证训练流程是否正常跑通。

如果替换成自己的数据，需要保证：

1. CSV 中字段名与 TOML 配置中的特征名和标签名一致
2. 标签为二分类形式
3. 至少保留部分非缺失特征
4. 至少保留部分非缺失标签

---

## 8. 复现步骤



### 8.1 配置 Python 环境

如果使用 Conda，可参考：

```bash
conda create -n adrd python=3.10 -y
conda activate adrd
pip install -r requirements.txt
(这里可能有几个包官方给漏了，自己pip一下)
```

### 8.2 使用仓库自带 smoke 数据复现训练

可以直接参考 `dev/train.sh` 中的命令。等价命令如下：

```bash
python dev/train.py \
    --data_path "./testdata/nonimg_smoke/smoke_all.csv" \
    --train_path "./testdata/nonimg_smoke/smoke_train.csv" \
    --vld_path "./testdata/nonimg_smoke/smoke_vld.csv" \
    --test_path "./testdata/nonimg_smoke/smoke_test.csv" \
    --cnf_file "./dev/data/toml_files/default_conf_new.toml" \
    --d_model 256 \
    --nhead 1 \
    --num_epochs 256 \
    --batch_size 32 \
    --lr 0.001 \
    --gamma 0 \
    --img_mode -1 \
    --img_net NonImg \
    --weight_decay 0.0005 \
    --ranking_loss \
    --save_intermediate_ckpts \
    --ckpt_path "./dev/ckpt/debug/model.pt"
```

### 8.3 直接运行脚本复现

Linux 或 Git Bash 环境下可直接运行：

```bash
bash dev/train.sh
```

该脚本已经完成了：

- 切换到项目根目录
- 配置 `PYTHONPATH`
- 指定 smoke 数据路径
- 设置训练超参数
- 指定 checkpoint 保存位置

---

## 9. 训练输出说明

训练过程中主要会输出以下信息：

- 数据集加载情况
- 配置文件中可用和不可用特征、标签
- 样本删除情况
- 标签分布
- 模型训练与验证指标
- 最优 checkpoint 保存情况

若指定：

```bash
--save_intermediate_ckpts
```

则会保存：

- 按主指标最优的模型
- 按 `AUC (PR)` 最优的模型

checkpoint 保存路径由 `--ckpt_path` 指定，例如：

```text
./dev/ckpt/debug/model.pt
```

---

## 10. 结果指标解释

验证阶段会输出多任务指标，每一列对应一个标签任务。

常见指标包括：

- `Accuracy`
- `Balanced Accuracy`
- `Precision`
- `Sensitivity/Recall`
- `Specificity`
- `F1 score`
- `MCC`
- `AUC (ROC)`
- `AUC (PR)`
- `Loss`

如果出现 `------`，通常表示该指标在当前预测结果下无定义，例如：

- 模型把所有样本都预测成同一类
- 某一类没有被预测到
- 分母为 0 导致 `nan`

这并不一定表示代码错误，更常见的原因是：

- 数据量过小
- 类别极不平衡
- 阈值固定为 `0.5` 后导致某一类完全未被预测

---

## 11. 当前版本的特点与限制

结合当前仓库代码，复现时需要注意以下几点：

- 当前仓库实际支持的是 **NonImg 非影像模式**
- `train.py` 中会读取测试集，但默认训练流程未自动输出测试集最终结果
- 训练所用最佳模型选择依据是验证集平均 `AUC (ROC)`
- 预测阈值固定为 `0.5`
- 若样本很少，部分指标容易出现 `nan`
- 仓库中保留了一些影像相关分支代码，但当前构建下并不是主路径

---

## 12. 项目总结

本项目可以视为一个面向结构化医学数据的多任务 Transformer 分类框架，其代码结构清晰地分为四层：

1. 数据读取层：`dev/data/dataset_csv.py`
2. 数据加工层：`adrd/utils/transformer_dataset.py`
3. 网络结构层：`adrd/nn/transformer.py`
4. 训练封装层：`adrd/model/adrd_model.py`

对于当前仓库而言，最直接的复现方式是使用 `testdata/nonimg_smoke/` 下的示例数据，通过 `dev/train.py` 或 `dev/train.sh` 跑通一个最小训练流程。若后续需要迁移到自己的数据集，只需保证 CSV 字段与 TOML 配置一致，并正确设置训练、验证和测试路径即可。
