# T1+D4：MSPE–SERS 扩散增强与 Transformer 多农药识别/定量项目

## 1. 项目简介

本分支整合了 MSPE–SERS 农药残留智能分析流程中的两个核心阶段：

1. **基于扩散模型的 SERS 小样本光谱增强**
2. **基于 Transformer 的多农药识别与浓度预测**

研究目标是结合磁性固相萃取—表面增强拉曼光谱（MSPE–SERS）与深度学习，实现水体和土壤中农药残留的快速检测、定性识别和定量分析。

当前使用的磁性 SERS 基底为：

**Fe₃O₄@Au@SiO₂@Au@SH-β-CD**

当前主要研究对象包括：

- **DEL**：溴氰菊酯（Deltamethrin）
- **CHL**：百菌清（Chlorothalonil）
- **TEB**：戊唑醇（Tebuconazole）

当前数据体系覆盖：

- 单组分农药
- 双组分农药
- 三组分农药
- water 基质
- soil 基质

> 实际农药类别、浓度等级、混合组合、基质标签、Raman 轴和样本定义，以项目中的真实数据目录、标签表和配置文件为准。

---

## 2. 仓库总体结构

扩散模型与 Transformer 模型在仓库根目录下作为**两个同级项目目录**保存：

```text
T1+D4/
│
├── README.md
├── .gitignore
│
├── diffusion/
│   ├── config/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   └── ...
│
└── transformer/
    ├── Dataset.py
    ├── Model_v2.py
    ├── SERSFormer_Training.py
    ├── Inference.py
    ├── Metric.py
    ├── EvalutionPlots.py
    ├── tests/
    └── ...
```

当前整合版本名称 **T1+D4** 表示：

- `diffusion/`：当前扩散模型 D4 系列代码
- `transformer/`：当前 Transformer T1 Query-only / Peak-Guided 相关代码

两个项目在代码层面相对独立，但共同构成完整的研究流程：

```text
diffusion/
    SERS 光谱增强
        │
        ▼
经过验证的生成光谱
        │
        ▼
transformer/
    多农药识别 + 浓度预测
```

---

## 3. 总体算法流程

```text
真实 SERS 光谱
      │
      ├──────────────────────────────────────────────┐
      │                                              │
      ▼                                              │
扩散模型增强                                         │
D0 → D1 → D2 → D3 → D4                              │
      │                                              │
      ▼                                              │
经过验证的生成光谱                                   │
      │                                              │
      └──────────────┬───────────────────────────────┘
                     ▼
              真实光谱 + 生成光谱
                     │
                     ▼
                Transformer / T1
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
多标签农药识别                 浓度预测
DEL / CHL / TEB             DEL / CHL / TEB
```

扩散模型生成的光谱不能只要求“整体曲线看起来相似”，还需要同时满足：

- 特征峰位置合理
- 峰高、峰宽和峰形合理
- 峰间相对强度合理
- 保留真实 SERS 噪声和基线变化
- 保留样本间差异
- 不复制训练样本
- 不发生模式坍塌
- 不同农药、浓度、混合组成和基质之间保持合理差异
- 最终能够提升下游识别和定量任务性能

---

# 第一部分：扩散模型

## 4. 扩散模型定位

扩散模型代码位于：

```text
diffusion/
```

当前扩散模型总体路线为：

- **D0**：普通无条件一维 DDPM 基准模型
- **D1**：比较 `pred_noise`、`pred_x0` 等预测目标
- **D2**：加入先验残差扩散
- **D3**：加入 SERS 光谱/物理约束
- **D4**：加入农药类别、浓度、混合组成和基质等条件信息
- **下游阶段**：将真实光谱和生成光谱用于 Transformer T1 及后续模型进行识别与定量

其中 D0 必须始终理解为：

> **普通无条件一维 DDPM 基准模型**

不能将 D0 作为最终模型或核心创新模型。

---

## 5. 扩散模型目标

扩散模型生成的 SERS 光谱需要尽可能保留：

- 主要特征峰位置
- 峰高分布
- 峰宽
- 峰形
- 峰间相对强度
- 合理的 SERS 噪声
- 基线变化
- 样本间多样性
- 农药类别差异
- 浓度差异
- 混合组成差异
- water / soil 基质差异

生成结果不能：

- 简单复制训练样本
- 只生成平均光谱
- 只做简单插值
- 出现明显模式坍塌
- 产生严重不合理负峰或伪峰

---

## 6. 扩散模型目录结构

```text
diffusion/
├── config/
│   └── ddpm_training.yaml
│
├── scripts/
│   ├── inspect_spectrum_data.py
│   ├── start_ddpm_training.py
│   ├── generate_spectra.py
│   └── ...
│
├── src/
│   ├── one_dimensional_ddpm.py
│   ├── model_builder.py
│   ├── ddpm_trainer.py
│   ├── checkpoint_manager.py
│   ├── spectrum_file_reader.py
│   ├── spectrum_dataset.py
│   ├── dataset_splitter.py
│   ├── spectrum_length_adapter.py
│   ├── intensity_normalizer.py
│   ├── spectrum_generator.py
│   ├── spectrum_exporter.py
│   └── ...
│
├── tests/
│   └── ...
│
└── ...
```

### `config/`

负责保存：

- 数据配置
- 模型配置
- 扩散配置
- 训练参数
- 输出目录
- 生成参数

当前优先使用：

```text
config/ddpm_training.yaml
```

避免无必要地创建大量重复配置文件。

### `scripts/`

负责命令行入口，例如：

- 检查数据
- 启动训练
- 恢复训练
- 条件生成
- 光谱评价
- 模型诊断

### `src/`

负责扩散模型核心功能，例如：

- 一维 U-Net
- Gaussian Diffusion
- 训练循环
- EMA
- checkpoint 管理
- 数据读取
- 数据划分
- Raman 轴适配
- 归一化
- 条件编码
- 先验残差
- SERS 物理/光谱约束
- 光谱生成
- 光谱导出

### `tests/`

用于验证：

- 数据读取
- 数据划分
- Raman 轴处理
- 归一化
- 前向传播
- 迷你训练
- checkpoint 恢复
- 光谱生成
- 条件生成
- 评价流程

---

## 7. 扩散模型数据原则

真实 SERS 数据通常不上传 GitHub，服务器中可位于类似：

```text
diffusion/data/input/
```

输入文件一般为 Excel 或 CSV。

典型格式：

```text
第一列：
Raman shift / cm⁻¹

第二列及以后：
不同 SERS 光谱强度
```

一个源文件通常包含约 20 条 mapping 光谱。

数据划分时必须避免：

> 同一独立实验样品的高度相关 mapping 光谱同时出现在 train、validation 和 test 中。

正式论文实验应优先按：

```text
source_file
```

或：

```text
sample_id
```

作为划分单位。

---

## 8. Raman 轴与归一化原则

扩散项目需要支持不同 Raman 范围和不同数据点数。

基本原则：

- 自动读取真实 Raman 轴
- 按 Raman 位移插值
- 不按数组下标简单截断
- checkpoint 中保存模型轴和轴元数据
- 归一化参数只在训练集拟合
- validation / test 使用训练集归一化状态
- 生成时使用同一状态反归一化
- 不能按每条生成光谱自身的最小值/最大值恢复强度

这些内容属于：

> **数据适配和工程功能**

不应作为扩散模型核心物理创新。

---

## 9. 扩散生成光谱评价

生成光谱至少需要从以下几个方面评价。

### 9.1 特征峰评价

- 峰命中率
- 峰位误差
- 峰高误差
- 峰宽误差
- 峰形差异
- 相对峰强误差

### 9.2 整体相似性

- Cosine Similarity
- Pearson Correlation
- RMSE
- MAE
- Spectral Angle
- 导数相似性

### 9.3 分布一致性

- PCA
- UMAP / t-SNE
- MMD
- Wasserstein Distance
- 强度分布
- 频域分布

### 9.4 多样性与过拟合

- 生成样本之间的差异
- 最近训练样本距离
- 重复样本检测
- 训练样本复制检查
- 模式坍塌检查

### 9.5 下游任务评价

最终需要比较：

```text
仅真实光谱
```

与：

```text
真实光谱 + 生成光谱
```

在 Transformer 下游任务中的表现。

---

# 第二部分：Transformer / T1

## 10. Transformer T1 模型定位

Transformer 项目位于：

```text
transformer/
```

当前 T1 模型主要承担：

1. DEL / CHL / TEB 多标签识别
2. DEL / CHL / TEB 浓度等级预测
3. 单组分、双组分、三组分分析
4. water / soil 基质分析
5. 扩散生成光谱的下游有效性验证

---

## 11. 当前 T1 模型结构

当前结构为：

```text
SERS 光谱
      │
      ├── 原始光谱 1D CNN ─────────────┐
      │                                │
      ├── 平滑光谱 1D CNN ─────────────┼── 特征融合
      │                                │
      └── 分位数特征 1D CNN ───────────┘
                                       │
                                       ▼
                                Transformer Encoder
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
             DEL Query              CHL Query              TEB Query
                │                      │                      │
             DEL 特征               CHL 特征               TEB 特征
              ┌──┴──┐                ┌──┴──┐                ┌──┴──┐
              ▼     ▼                ▼     ▼                ▼     ▼
            存在   浓度等级         存在   浓度等级         存在   浓度等级
```

DEL、CHL、TEB 各自拥有一个独立的可学习 Query。

每个 Query 都可以查看整条有效 Raman 光谱，因此不会只依赖单个重叠峰进行判断。

---

## 12. Query-only 模式

当前 T1 使用：

```text
query_only
```

该模式下：

- DEL 有独立 Query
- CHL 有独立 Query
- TEB 有独立 Query
- 三个 Query 均查看完整有效光谱
- 不加入显式 peak prior
- 不提前规定某个峰属于某一种农药

当前阶段主要用于回答：

> pesticide-specific learnable query 是否能够从真实混合 SERS 光谱中自动学习 DEL、CHL、TEB 的有效证据。

后续可以再加入经过严格验证的 soft peak guidance 进行消融比较。

---

## 13. Transformer 公共 Raman 轴

对真实数据检查发现，原始 Raman 轴长度与 CHL 标签存在明显混杂：

```text
CHL 不存在 → 1401 点
CHL 存在   → 1901 点
```

如果直接把不同有效长度送入 Transformer，模型可能学习：

> Raman 轴长度

而不是：

> CHL 的真实化学光谱特征

因此当前 Transformer T1 模型统一使用公共 Raman 区间：

```text
600–2000 cm⁻¹
1401 points
```

这样所有样本都采用统一输入长度。

原有 `valid_mask` 机制仍然保留，用于未来真正存在缺失 Raman 区间的情况。

---

## 14. 当前数据规模

当前真实数据：

```text
126 个条件源文件
每个源文件约 20 条 mapping 光谱
总计 2520 条光谱
```

当前数据量：

```text
train      : 1512
validation : 504
test       : 504
```

当前实现中，一个源文件的 20 条 mapping 光谱按：

```text
12 / 4 / 4
```

进入 train / validation / test。

### 重要说明

如果一个源文件中的 20 条 mapping 光谱属于同一个独立实验样品，那么正式论文实验应改为按：

```text
source_file
```

或：

```text
sample_id
```

进行划分。

同一独立样品的 mapping 光谱不应同时进入 train、validation 和 test。

---

## 15. Transformer 项目结构

```text
transformer/
├── Dataset.py
├── Model_v2.py
├── SERSFormer_Training.py
├── Inference.py
├── Metric.py
├── EvalutionPlots.py
├── tests/
└── ...
```

### `Dataset.py`

主要负责：

- 读取 SERS 文件
- 解析 DEL / CHL / TEB 标签
- 构建多标签分类目标
- 构建回归目标
- 构建 train / validation / test
- 公共 Raman 轴处理
- Raman 轴与标签关系审计
- 后续 peak prior 构建接口

### `Model_v2.py`

主要负责：

- 原始光谱 CNN 分支
- 平滑光谱 CNN 分支
- 分位数特征 CNN 分支
- 特征融合
- Transformer Encoder
- DEL / CHL / TEB 可学习 Query
- 农药专属分类头
- 农药专属回归头
- query-only / peak-guided 模式支持

### `SERSFormer_Training.py`

主要负责：

- 模型训练入口
- Optimizer
- 多任务 Loss
- validation
- 动态任务权重
- checkpoint 保存
- training history 保存
- Query 模式选择

### `Inference.py`

主要负责：

- checkpoint 加载
- validation / test 推理
- 综合性能评价
- 预测结果导出
- Query Attention 诊断

### `Metric.py`

主要负责：

- 多标签分类指标
- 单农药分类指标
- 回归指标
- 多任务指标汇总

### `EvalutionPlots.py`

主要负责：

- 混淆矩阵
- 归一化混淆矩阵
- ROC 曲线
- 小提琴图
- Predicted-vs-True 图
- 训练曲线
- 其他评价图

### `tests/`

当前主要测试：

- 公共 Raman 轴处理
- Query-only Attention
- Peak-Guided Attention
- 前向/反向传播
- 评价管线
- checkpoint 兼容性

---

## 16. 多标签分类评价

当前 Transformer T1 评价系统包括：

### 整体多标签指标

- Exact Match Accuracy
- Hamming Accuracy
- Sample-wise Jaccard Accuracy
- Micro Precision
- Macro Precision
- Micro Recall
- Macro Recall
- Micro F1
- Macro F1
- Micro AUROC
- Macro AUROC

### 单农药指标

分别对：

```text
DEL
CHL
TEB
```

计算：

- Accuracy
- Precision
- Recall
- F1
- Specificity
- AUROC

---

## 17. 混淆矩阵

分别为：

```text
DEL
CHL
TEB
```

生成独立的 2×2 混淆矩阵：

```text
TN  FP
FN  TP
```

同时保存：

- 原始数量矩阵
- 按行归一化矩阵

可以直接分析：

- False Positive
- False Negative
- Recall
- Specificity

---

## 18. 农药组合共现评价

评价系统还会分析：

```text
DEL + CHL
DEL + TEB
CHL + TEB
DEL + CHL + TEB
```

用于判断：

- 特征峰重叠
- 光谱干扰
- 某种农药信号占优
- 混合复杂度增加

对模型识别能力的影响。

---

## 19. ROC / AUROC

自动生成：

```text
DEL
CHL
TEB
Micro Average
Macro Average
```

对应的 ROC 曲线和 AUROC。

AUROC 与固定 0.5 阈值下的 Precision / Recall / F1 配合使用，可以更全面评价模型的分类能力。

---

## 20. 回归评价

当前回归指标包括：

- MAE
- MSE
- RMSE
- R²

目前浓度标签仍是等级码：

```text
0 = 不存在
1 = 低浓度
2 = 中浓度
3 = 高浓度
```

因此当前回归任务本质上是：

> **浓度等级预测**

而不是最终的真实物理浓度定量。

后续需要接入实际浓度值，才能将 R²、RMSE、MAE 解释为正式浓度定量指标。

---

## 21. 回归评价口径

当前评价系统包括：

- 所有回归输出
- 真实存在农药的样本
- SERSFormer-2.0 风格：正确识别后再评价浓度
- 分类门控后的端到端回归
- 不存在农药时的浓度误差

这样可以避免只使用单一回归指标造成过度乐观。

---

## 22. 回归可视化

自动生成：

- Predicted vs True
- DEL 小提琴图
- CHL 小提琴图
- TEB 小提琴图
- 各农药回归指标表

用于观察：

- S / M / H 是否能分开
- 是否存在预测偏移
- 是否存在严重离群点
- 不存在农药时是否错误预测较高浓度

---

## 23. 分层评价

当前模型还会按以下方式分层评价。

### 混合复杂度

```text
single
binary
ternary
```

即：

- 单组分
- 双组分
- 三组分

### 基质

```text
water
soil
```

### 农药组合

```text
DEL
CHL
TEB
DEL+CHL
DEL+TEB
CHL+TEB
DEL+CHL+TEB
```

可以直接判断：

> 随着混合农药数量增加，模型性能是否下降。

---

## 24. Query Attention 诊断

T1 会导出 DEL / CHL / TEB 三个 Query 的 Attention 分布。

典型输出：

```text
query_attention/
├── query_attention_profiles.csv
├── query_attention_top_regions.csv
├── DEL_query_attention.png
├── CHL_query_attention.png
└── TEB_query_attention.png
```

用途是检查：

- DEL Query 主要关注哪些 Raman 区域
- CHL Query 主要关注哪些 Raman 区域
- TEB Query 主要关注哪些 Raman 区域
- 三个 Query 是否真正学习出不同的关注模式

注意：

> Attention 只能作为模型解释和诊断依据，不能单独作为某个 Raman 峰属于某农药的化学证据。

---

## 25. 当前 T1 状态

目前已通过：

```text
Python 语法检查              PASS
Peak-Guided 单元测试          PASS
公共 Raman 轴测试             PASS
Query-only 单元测试           PASS
Query-only smoke 训练         PASS
checkpoint 保存              PASS
完整 validation 评价管线      PASS
```

当前 smoke 实验只用于确认：

> 整个模型和评价流程可以正常运行。

不能作为最终性能结果。

---

## 26. Transformer 后续消融路线

建议保持：

```text
T0
原始 SERSFormer
        │
        ▼
T1
Pesticide-Specific Query Attention
Query-only
        │
        ▼
T2
Pesticide-Specific Query Attention
+ 经过验证的 Soft Peak Guidance
```

这样可以分别回答：

1. Pesticide-specific Query 本身是否有效？
2. 在 Query 基础上加入 SERS 峰引导是否进一步提高性能？

---

# 第三部分：Diffusion + Transformer 联合研究

## 27. 最终下游对比实验

最终需要比较：

```text
仅真实光谱
```

与：

```text
真实光谱 + 经过验证的扩散生成光谱
```

还可以进一步比较不同：

```text
真实 : 生成
```

比例。

最终判断扩散模型是否有价值，不只看生成曲线本身，还要看是否提升：

- DEL / CHL / TEB 多标签识别
- 双组分识别
- 三组分识别
- water / soil 基质泛化
- 浓度回归
- 低浓度样本识别
- 混合峰干扰下的鲁棒性

---

## 28. 服务器环境

当前典型服务器环境：

```text
Ubuntu 24.04
Python 3.11
PyTorch 2.11.0 + cu128
2 × NVIDIA RTX 3090 24 GB
Conda 环境：sers_ddpm
```

激活环境：

```bash
conda activate sers_ddpm
```

---

## 29. 常用命令

### 扩散模型

进入：

```bash
cd diffusion
```

检查数据：

```bash
python scripts/inspect_spectrum_data.py \
  --config config/ddpm_training.yaml
```

训练：

```bash
python -m scripts.start_ddpm_training \
  --config config/ddpm_training.yaml
```

生成光谱：

```bash
python -m scripts.generate_spectra \
  --config config/ddpm_training.yaml \
  --checkpoint outputs/checkpoints/latest.pt \
  --number 100
```

---

### Transformer

进入：

```bash
cd transformer
```

Query-only 训练：

```bash
CUDA_VISIBLE_DEVICES=0 \
python SERSFormer_Training.py \
  --query-attention-mode query_only \
  --output-directory outputs/t1_query_only \
  --epochs 20 \
  --batch-size 32
```

运行测试：

```bash
python -m pytest -q
```

---

## 30. Git 中不上传的内容

以下内容原则上不上传 GitHub：

```text
diffusion/data/
diffusion/outputs/

transformer/data/
transformer/outputs/

*.pt
*.pth
*.ckpt

*.xlsx
*.xls

__pycache__/
.vscode/

*.patch
.backup*/
```

如果某些小型 CSV 是：

- 标签定义
- 手工整理元数据
- 配置文件

可以根据实际需要单独纳入 Git。

但自动生成的评价 CSV、训练日志和大规模输出通常不建议上传。

---

## 31. 论文创新定位

项目中需要明确区分以下几类内容。

### 1. 基础模型

- 普通一维 DDPM
- CNN
- Transformer
- 原始 SERSFormer

### 2. 参数调整

- 学习率
- Batch Size
- 网络宽度
- Epoch
- Diffusion timestep

### 3. 工程功能

- 数据读取
- Raman 轴适配
- metadata
- checkpoint
- 文件导出
- 绘图

### 4. 模型变体

- `pred_noise`
- `pred_x0`
- 先验残差扩散
- 条件扩散
- Pesticide-Specific Query Attention

### 5. 核心研究创新方向

包括：

- 与 SERS 光谱先验结合的残差扩散
- 特征峰与峰形约束
- 农药类别/浓度/混合组成/基质条件控制
- 经过严格验证的 pesticide-specific peak guidance
- 面向下游任务的生成光谱筛选
- 通过多标签识别和回归验证扩散增强价值

---

## 32. Git 分支信息

本次整合分支名称：

```text
T1+D4
```

本次建议 commit message：

```text
T1+D4
```
