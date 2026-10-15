# T3.36：三元DEL/TEB专用浓度融合分支

T3.35仅微调等级头没有超过当前T3.33。这次只把微调位置前移到浓度融合和投影，不解冻二维图像CNN、Transformer、峰查询或多峰数值证据。

## 修改范围

- 从T3.33当前第4轮权重出发，复制原混合物Query融合、128→64浓度投影、DEL/TEB两个64→64→2 CORN头。默认共148,807个可训练参数。
- 先提取一次融合前的三组128维Query。原CNN、Transformer、峰注意力、17峰数值辅助及其归一化全部冻结；不重新拟合强度尺度或统计量。
- 原T3.33输出作为固定参考。只把新分支与原分支的logits差用于DEL/TEB；原T3.33小修正器保持冻结。新分支没有±0.5硬限制。
- 初始化时新旧分支相同，概率/logits逐元素复现当前模型。仅当原分类输出预测三药均存在时启用新分支；推理不读取真实标签、条件名或组合ID。
- 分类概率、CHL概率/logits、未预测为三元的输出保持逐元素不变。旧分支及原模型继续保存，可随时回退。

这是一项待验证实验，不能保证提高准确率。当前三元真实训练与验证差距仍需要关注。没有合格提升就保留T3.33，不修改数据、划分或验收标准来制造提升。

## 数据和缓存

默认保持126条件、真实12/4/4和每条件48条生成谱。直接读取T3.33的371维特征缓存，完全不覆盖旧缓存。

第一次check会按原图像尺度读取真实和生成光谱，重新运行一次冻结图像主干，缓存 **train/validation** 的融合前Query。原始文件、固定划分、T3.32代码、原缓存与权重指纹必须一致。重提取预测与原缓存有明显差异时停止。

新增缓存保存在本次输出的query_cache目录。后续train会核对来源并复用，不再运行CNN/Transformer提取。构造原Dataset时会审计固定测试划分的身份，但不为测试集构建DataLoader、提取特征、计算预测或用于选择模型。

## 训练配置

| 项目 | 默认值 |
|---|---|
| 新融合模块学习率 | 3e-6 |
| 新浓度投影学习率 | 1e-5 |
| 新等级头学习率 | 3e-5 |
| 最低学习率 | 各组最高值的1% |
| 调度 | 3epoch预热＋余弦退火 |
| 优化器 | AdamW，weight decay 0.01 |
| epoch上限 / patience | 20 / 5，预热后计早停 |
| 训练batch | 32，真实16＋生成16 |
| 来源损失 | 真实75%＋生成25% |
| 约束 | 条件CORN＋原本正确预测的概率稳定项0.2＋原分支参数锚定0.01 |

每轮遍历648条真实三元训练样本，从2592条生成三元池中打乱抽取等量样本；不修改生成文件或真实数据。三元选择仅在训练监督时使用真实标签，推理路由始终仅用分类预测。

验收与T3.35相同，当前参考为整体等级1027、完整谱408、三元133、分类504。三元完整正确数必须增加，整体等级/完整谱/分类以及DEL、TEB各自正确数均不得下降。合格权重按三元、完整谱、整体等级、同口径全局验证loss排序；无增益时best_accepted.pt标记epoch=0，为当前模型回退。

## 上传和运行

将ZIP下载后上传服务器 `/home/wqzheng/`。默认需保留T3.32、T3.33源码和T3.33权重/特征缓存；不依赖T3.35目录，不上传GitHub。

```bash
conda activate sers_ddpm
ls -lh /home/wqzheng/SERS_T3_36_Ternary_Fusion_Finetune.zip && \
unzip -n /home/wqzheng/SERS_T3_36_Ternary_Fusion_Finetune.zip -d /home/wqzheng/ && \
python /home/wqzheng/T3_36_Ternary_Fusion_Finetune/verify_package.py
```

首次提取建议GPU0，batch16；这里只做冻结前向，随后才训练小分支。

```bash
CUDA_VISIBLE_DEVICES=0 python -u /home/wqzheng/T3_36_Ternary_Fusion_Finetune/run_fusion.py --phase check --device cuda && \
CUDA_VISIBLE_DEVICES=0 python -u /home/wqzheng/T3_36_Ternary_Fusion_Finetune/run_fusion.py --phase train --device cuda && \
python /home/wqzheng/T3_36_Ternary_Fusion_Finetune/run_fusion.py --phase pack
```

数据和代码非默认位置时，所有阶段使用同一组 `--project`、`--real-root`、`--generated-root`、`--image-package`、`--correction-package`、`--teacher-run`、`--teacher-checkpoint`、`--image-cache`、`--output-directory`。不要混用不同来源的生成数据。CPU也能运行，但首次大图提取通常较慢。

默认输入参考：
`/home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026/train/checkpoints/best_accepted.pt`

默认权重：
`/home/wqzheng/project_transformer/outputs/t3_36_ternary_fusion_seed2026/train/checkpoints/best_accepted.pt`

发回报告包：
`/home/wqzheng/project_transformer/outputs/t3_36_ternary_fusion_seed2026_reports.zip`

已有train或smoke目录时拒绝覆盖，重复实验统一指定新的output-directory。check会复用同一输出下的query_cache。--phase smoke仅2步优化，使用独立smoke目录。

## 报告与诊断

- check_report.json：当前参考复现、初始化逐元素一致、可训练参数和各模块梯度检查。
- query_extraction_audit.json：新增Query形状、来源和冻结主干检查。
- train/training_history.csv：各组学习率、真实训练三元准确率、验证三元准确率及验收状态。
- reference_validation：当前T3.33参考预测，统一全局loss口径。
- train/accepted_validation及accepted_training_*：最终保留模型的预测与分来源成绩。
- train/latest_validation及latest_training_*：最后训练候选的预测与成绩，即使回退也保留，避免误把回退成绩当成新候选训练成绩。
- train/comparison.json和run_summary.json：增益、实际保留轮次、输入哈希/冻结状态、零测试预测。

报告ZIP排除权重和特征缓存。训练loss与验证loss含有不同来源加权和正则，不直接比较大小判断过拟合。验证集已多次用于选择，这些成绩不能替代最终独立测试。

## 部署

```python
from run_fusion import load_for_inference
model, scale, state = load_for_inference(
    '/实际路径/best_accepted.pt',
    '/home/wqzheng/T3_32_Image_Recognition',
    '/home/wqzheng/T3_33_Ternary_Correction',
    device='cuda',
)
output = model(image_inputs)
```

输入保持T3.32四个张量：image、image_valid_columns、raw_intensity、measured_mask。checkpoint嵌入完整旧模型及新分支，部署不需要旧权重文件或训练缓存；仍需原T3.32/T3.33源码。--phase evaluate只支持train/validation，无自动测试入口。

## 文件功能

- run_fusion.py：来源核对、缓存准备、分层学习率、微调/验收/保存重载、报告导出和部署加载。
- fusion_model.py：复制融合分支、冻结参考、预测路由、输出保护、条件CORN和稳定约束。
- query_cache.py：钩取融合前Query、复现原缓存、缓存来源和行身份核对。
- cached_evaluation.py、metrics_core.py、ordinal_core.py、cache_audit.py：沿用已有评价/解码/审计口径，新增Query输入。
- test_fusion.py：合成数据梯度、保护、路由和往返检查。
- verify_package.py、checksums.json：文件完整性检查。
- validation_report.json、SOURCE_PROVENANCE.json：实际本地验证范围及代码来源。
