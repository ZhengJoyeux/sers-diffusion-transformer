# T3.33：冻结第16轮二维基线，修正三元DEL/TEB浓度

独立增量补丁，不改动T3.32代码、权重、数据或一维DiT，不上传GitHub。需要已安装的 `/home/wqzheng/T3_32_Image_Recognition`。默认加载T3.32的 `train/checkpoints/best_ordinal.pt`，必须为第16轮，且保存的验证计数为1026个等级正确、406条完整正确、131条三元完整正确、504条类别正确。

## 方法

1. 原图像CNN、Transformer、Query、多峰证据、分类与浓度头全部冻结，始终使用eval模式，包括Dropout及所有归一化状态。
2. 保留T3.32权重中保存的2048×512绘图尺度、原始强度归一化、峰先验和median解码，不重新拟合原模块。
3. 从冻结模型提取三药浓度特征、预测logits、类别概率，以及已有17峰窗口的数值统计/可用掩码。无类别标签、文件名、基质标签、条件ID进入修正网络。
4. 371维输入经过两个独立的小MLP，分别输出DEL、TEB的两个CORN logit修正量。默认隐藏32维，共23940个可训练参数；修正上限为±0.5，末层零初始化。
5. 推理只根据冻结类别预测 `(cp >= 0.5).all()` 决定是否启用。CHL、类别概率及未被预测为三元的样本直接保留基线输出。真实一元/二元样本如果被基线误判为三元，仍可能进入修正，因此保护定义是“预测非三元”，不是用真实标签路由。
6. 修正后保持CORN概率单调。采用概率差补偿避免缓存批大小带来的微小数值偏差，使零修正时与基线概率完全一致。

新MLP的输入均来自冻结特征；新的特征标准化仅拟合真实训练集中“预测为三元”的样本，不用生成、验证或测试数据。优化器仅包含新MLP，训练损失仅对真实标签和预测均为三元的训练样本计算。

损失：DEL与TEB条件CORN的平均值＋小幅修正惩罚＋已判对训练样本的概率稳定惩罚。默认对基线判错的训练等级赋2倍权重；所有判错判断只发生在训练损失中。保留原训练集和生成谱选取，默认沿用第16轮记录的每条件48条生成谱，不改变原始数值或12/4/4划分。

## 文件

| 文件 | 功能 |
|---|---|
| baseline_bridge.py | 核对T3.32源码、epoch16权重、数据身份与冻结状态 |
| correction_model.py | 冻结基线、特征提取、有界修正、路由保护和训练损失 |
| feature_cache.py | 缓存固定图像特征、构建训练视图及输出评价表 |
| run_correction.py | check/smoke/train/evaluate/pack入口 |
| test_correction.py | 验证零初始化、路由、输出保护、梯度、归一化与权重往返 |
| t332_sources.json | 所需T3.32 Python源码的SHA256 |
| verify_package.py | 补丁完整性检查 |
| checksums.json | 本补丁文件校验值 |
| validation_report.json | 制作时验证范围，不是你的真实训练成绩 |

## 解压与检查

将zip放到 `/home/wqzheng/`：

```bash
conda activate sers_ddpm
unzip -n /home/wqzheng/SERS_T3_33_Ternary_Correction_patch.zip -d /home/wqzheng
python /home/wqzheng/T3_33_Ternary_Correction/verify_package.py
python /home/wqzheng/T3_33_Ternary_Correction/test_correction.py
```

```bash
CUDA_VISIBLE_DEVICES=0 python -u /home/wqzheng/T3_33_Ternary_Correction/run_correction.py \
  --phase check --device cuda
```

检查会：

- 检查旧包关键源码未变、默认权重是报告中的第16轮；不要使用latest.pt。
- 读取旧Dataset，核对12/4/4及生成训练子集身份与第16轮完全一致。
- 用冻结基线处理训练和验证图片，并缓存特征。第一次需要一段时间，每25批打印进度；后续smoke/train复用，不再重复跑全部图片。
- 完整复核504条真实验证谱的1026/406/131计数；不匹配时停止，不自动放宽。
- 检查零修正输出与基线相同，以及反向传播不会更新基线。
- 不对测试样本做特征提取、预测或调参。

默认路径：

```text
基线包：/home/wqzheng/T3_32_Image_Recognition
基线权重：/home/wqzheng/project_transformer/outputs/t3_32_line_image_seed2026/train/checkpoints/best_ordinal.pt
数据项目：/home/wqzheng/project_transformer
输出：/home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026
原PNG缓存：/home/wqzheng/project_transformer/outputs/t3_32_line_image_seed2026/image_cache
```

路径不同时使用 `--image-package`、`--baseline`、`--project`、`--real-root`、`--generated-root`、`--image-cache` 显式指定。不支持从其他权重静默继续。缓存身份包含数据文件内容、样本选择、基线权重及特征提取源码哈希，输入变化不会误用旧缓存。

## 冒烟

```bash
CUDA_VISIBLE_DEVICES=0 python -u /home/wqzheng/T3_33_Ternary_Correction/run_correction.py \
  --phase smoke --device cuda
```

训练2个小网络批次，完整评价验证缓存，并检查保存读取及基线保护。smoke与train输出分开；重复运行同一阶段需要更换`--output-directory`，已有权重不会覆盖。

## 正式训练

```bash
CUDA_VISIBLE_DEVICES=0 python -u /home/wqzheng/T3_33_Ternary_Correction/run_correction.py \
  --phase train --epochs 20 --batch-size 32 --warmup-epochs 3 --device cuda
```

默认AdamW、学习率1e-4、最低1e-6、3epoch预热＋余弦退火、weight decay0.01、patience5、seed2026。早停计数从预热结束后开始。batch32只用于缓存特征上的小网络训练；冻结图像前向仍默认batch16。特征缓存完成后，训练较轻。

每轮检查基线所有参数与缓冲区的内容哈希不变，并核对类别、CHL和未路由样本的输出完全相同。

## 权重选择

同时保留：

- `best_accepted.pt`：推荐候选，必须三元完整正确数严格超过131，且总体等级正确不低于1026、完整正确不低于406、类别正确不低于504。通过后优先按三元、完整、总体等级及损失排序。
- `best_ternary.pt`：三元优先的探索候选，可能不满足总体保护条件，不能直接当作推荐结果。
- `best_overall.pt`：总体等级优先的探索候选。
- `latest.pt`：最后一轮。

开始时前三份均保存epoch0零修正基线。如果没有满足条件的新结果，`best_accepted.pt`保留epoch0，`comparison.json`明确标记`baseline retained`，不会声称有提升。epoch0是新分支训练轮次，嵌入的原基线仍是T3.32第16轮。

最佳候选：

```text
/home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026/train/checkpoints/best_accepted.pt
```

报告：`train/comparison.json`、`train/training_history.csv`、`train/accepted_validation/ordinal_metrics.json`和predictions.csv。check_report.json中记录基线复核、训练归一化范围与可训练参数数目。

新checkpoint内嵌冻结基线权重、绘图尺度和修正参数，可整体部署；加载代码依赖已核对的T3.32包。无需另外读取原best_ordinal.pt才能运行新checkpoint。

## 评价与打包

```bash
CUDA_VISIBLE_DEVICES=0 python -u /home/wqzheng/T3_33_Ternary_Correction/run_correction.py \
  --phase evaluate --split validation --device cuda \
  --checkpoint /home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026/train/checkpoints/best_accepted.pt
```

方案固定后才将`--split validation`换为`--split test`进行最终测试，测试不参与训练和选择。验证集已经多次使用，验证提升不能直接称为独立测试提升。

训练后打包报告（不包含大权重和缓存）：

```bash
python /home/wqzheng/T3_33_Ternary_Correction/run_correction.py --phase pack --device cpu
```

得到：`/home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026_reports.zip`。
