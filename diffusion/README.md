# 当前生成端：D4.31 第20轮二维曲线图 DiT

当前推荐来自 D4.35 的三组同预算长训练对照：**D4.31、epoch 20、step 1900、EMA**。它在9组验证结果中具有最高一阶导数相关性和最低绝对 patch 边界跳变，并非所有指标均最优。网络有83,083,776个参数。旧一维 D4.27 代码和配置仍供历史对照。

## 文件与入口

| 文件 | 用途 |
|---|---|
| `run_generation.py` | 当前模型的权重检查、生成及常规评价统一入口 |
| `current_diffusion.py` | 选定权重SHA256核验、固定坐标先验整谱重抽、EMA生成和独立评价 |
| `src/dit_curve2d.py` | 真正二维输入输出的全高度条带DiT，768×16 patch，120 token |
| `src/curve_image_codec.py` | 保存坐标、渲染、解码、掩码和残差桥接 |
| `src/curve_image_diffusion.py` | 二维像素加噪、DiT去噪和光谱约束 |
| `extensions/D4_31_Budget2/` | 实际训练所用的2%损失预算扩展、整谱先验重抽及回归测试 |
| `scripts/evaluate_d4_2_generation.py` | MSE、Pearson、QQ、MMD、JSD、SWD、多样性及峰分布评价 |
| `../current_configs/D4.31/` | 选定检查点、模型参数、训练协议和验证结果摘要 |
| `../current_configs/D4.35/` | 9组同口径比较与逐条件改善数量 |

## 在已有服务器权重上生成

以下命令从仓库根目录运行，要求现有PyTorch环境。仓库不包含权重和原始数据，也不会自动下载它们。

```bash
GEN_CP=/home/wqzheng/d4_35_long20_v1/d4_31/boundary/checkpoints/step_00001900.pt
REAL_DATA=/home/wqzheng/project/data/input

python diffusion/run_generation.py inspect --checkpoint "$GEN_CP"

python -u diffusion/run_generation.py generate \
  --checkpoint "$GEN_CP" --real-data "$REAL_DATA" \
  --condition DEL-H_water --device cuda --save-stage-trace \
  --output "$HOME/d431_current_generation_v1"
```

使用`--condition all`生成126个保存条件。每条件固定50条，使用原200步扩散／100步DDIM设置。输出目录必须是新目录。推理仍需要原始训练谱，以沿用已有校准和交付保护；改变`--real-data`路径不会重新划分训练、验证、测试。

输入格式如`DEL-H_TEB-S_water`。规则将其转换为14维结构化条件，不使用自然语言编码器。二维图尺寸1920×768、归一化纵轴[-1.5,1.5]保持检查点原值。越界先验按同条件整条重抽，记录`prior_bounds_audit.json`，不逐点截断。现有最终交付保护保持启用。

## 独立计算常规评价

```bash
python -u diffusion/run_generation.py evaluate \
  --checkpoint "$GEN_CP" --real-data "$REAL_DATA" \
  --generated "$HOME/d431_current_generation_v1" \
  --condition DEL-H_water --reference-split validation \
  --output "$HOME/d431_current_metrics_v1"
```

本入口直接计算原有常规指标，**FID不是依赖，不连接网络下载特征权重**。只有生成了全部条件时才使用默认`--condition all`。输出`condition_metrics.csv`、`overall_metric_summary.csv`、`peak_morphology_details.csv`和独立的完成／结果哈希记录。一阶导数明细仍由原评价器输出；已选实验的阶段导数和接缝结果见保存的比较摘要。验证集已用于模型选择；测试集不应参与反复选模型。

## 重现训练协议

若保留原D4.28 suite、原始初始化权重和数据，可使用实际验证过的扩展重现D4.31额外20轮训练：

```bash
python diffusion/extensions/D4_31_Budget2/d4_30.py prepare \
  --project "$PWD/diffusion" \
  --suite /home/wqzheng/project/outputs/experiments/d4_28_curve2d_20261015_204734_914086 \
  --output "$HOME/d431_reproduction_v1" --epochs 20 --learning-rate 1e-5

python -u diffusion/extensions/D4_31_Budget2/d4_30.py train \
  --experiment "$HOME/d431_reproduction_v1" --arm boundary
```

该入口从同一D4.28原始EMA初始化，不从已选第20轮权重再训练。重新建立AdamW和EMA，固定学习率1e-5，训练20轮；不是恢复原优化器状态。原suite若使用相对数据路径，需要先使`diffusion/data/input`指向同一原始数据目录。源码、扩展、数据和初始化权重都有哈希保护，校验不一致时停止。

扩展文件名保留`d4_30.py`以维持实际实验指纹，其`VERSION`是`d4_31_boundary_budget2_v1`。`maximum_ratio=0.02`限制辅助损失值，不限制梯度范数。该驱动每轮保存一次；实际D4.35连续训练每5轮保存一次，这不改变训练目标和优化步数。历史对照目录不代表新的推荐模型。

## 已知限制

第20轮D4.31导数Pearson宏平均0.807751，边界跳变量6.076895。QQ尾部不足仍为122/126，峰宽IQR比宏平均约1.361。固定坐标往返最大误差0.047805像素；不能据此认定采样内部保护无影响。只有一个训练种子，未证明独立测试或下游浓度识别增益。

[根目录说明](../README.md) · [当前模型登记](../CURRENT_MODEL.json) · [原始参考与许可证](THIRD_PARTY_NOTICES.md)
