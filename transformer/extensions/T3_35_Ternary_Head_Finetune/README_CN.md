# T3.35：冻结二维识别主干，仅微调三元 DEL/TEB 浓度等级头

本补丁根据 T3.34 诊断实施一次小范围实验。原二维图像 CNN、Transformer、峰查询、多峰强度证据、混合融合、共享投影、分类和 CHL 均保持冻结。仅复制原 DEL/TEB 的 `64 -> 64 -> 2` CORN 等级头并训练，共 **8,580个参数**。只在原分类预测为三元时启用；路由不读真实标签。

## 为什么这样改

当前真实训练三元完整准确率78.86%，验证61.57%；在原±0.5修正范围和既有保护条件下，验证集逐样本理想修正的乐观上限仅66.20%。因此，这次检验冻结特征能否通过调整等级边界得到更好结果，不继续训练原来的有限幅度小修正头。

这不是已证明有效的优化。训练—验证差距可能涉及泛化或同条件谱的差异；解除输出限制不保证提升。

## 与现有模型的关系

- 读取 T3.33 当前 `train/checkpoints/best_accepted.pt`，默认第4轮。比较对象为当前1027/408/133/504，而不是更早的T3.32。
- 固定保留已训练T3.33修正器。新头更新原CORN logits，旧修正器不更新。
- 使用原头与复制头输出之差替换原等级映射，因此初始化与当前T3.33概率及logits逐元素一致；新头本身没有±0.5硬限制。
- 保持分类概率、CHL概率/logits、未预测为三元的输出逐元素不变。
- 复用T3.33缓存的371维特征。前192维是三种农药各64维的共享投影输出，新头使用各自64维；旧修正器仍使用原371维。
- 不重新绘图或提取CNN特征，不重拟合归一化，不修改原始光谱、固定12/4/4划分、原权重和原报告。
- 训练只读train和validation缓存；无自动测试入口，不上传GitHub。

## 默认训练

| 项目 | 默认值 |
|---|---|
| 优化器 | AdamW，weight decay=0.01 |
| 学习率 | 3e-5，最低3e-7 |
| 预热及调度 | 3 epoch预热，随后余弦退火 |
| 上限及早停 | 最多20 epoch，预热后连续5轮无合格改善早停 |
| batch size | 32：16条真实、16条生成；末批按真实剩余量调整 |
| 来源损失 | 0.75 ×真实损失 + 0.25 ×生成损失 |
| 监督 | DEL/TEB条件CORN，无错误额外加权 |
| 稳定约束 | 原本正确等级的累计概率MSE，权重0.2 |
| 参数约束 | 相对原头权重的归一化平方距离，权重0.01 |
| 训练样本 | 真实且预测均为三元，预计真实648条、生成池2592条 |

每完整epoch真实三元样本随机遍历一次，同时从生成池打乱抽取等量样本，预计648条。75%/25%指两个来源的损失权重，不是修改数据文件或标签。每轮重排，生成池不是永久裁剪。

验收要求：相对于当前T3.33，三元完整正确数必须增加；整体等级正确数、完整谱正确数、分类正确数，以及DEL/TEB各自等级正确数均不得下降。合格模型按三元、完整谱、整体等级、同口径验证loss依次排序。没有合格提升时，`best_accepted.pt`为明确标记epoch=0的当前T3.33回退。

验证和训练中的监督损失不同，不直接比较二者大小判断过拟合。参考与候选验证loss均按整个验证集统一汇总条件CORN，修复旧报告loss汇总口径不一致的问题。验证集已多次用于选择，因此不能将小幅验证改善当作独立测试增益。

## 下载、解压、校验

先把ZIP上传服务器 `/home/wqzheng/`，不是仅下载到Windows电脑。

```bash
conda activate sers_ddpm
ls -lh /home/wqzheng/SERS_T3_35_Ternary_Head_Finetune.zip && \
unzip -n /home/wqzheng/SERS_T3_35_Ternary_Head_Finetune.zip -d /home/wqzheng/ && \
python /home/wqzheng/T3_35_Ternary_Head_Finetune/verify_package.py
```

依赖已安装的T3.33目录和原输出目录中的特征缓存；不会覆盖旧代码。CPU可完成本次小头训练，原GPU任务可继续运行。若使用GPU，添加`CUDA_VISIBLE_DEVICES=0`并把`--device cpu`换为`--device cuda`。

## 检查、正式训练、打包

```bash
python -u /home/wqzheng/T3_35_Ternary_Head_Finetune/run_heads.py --phase check --device cpu && \
python -u /home/wqzheng/T3_35_Ternary_Head_Finetune/run_heads.py --phase train --device cpu && \
python /home/wqzheng/T3_35_Ternary_Head_Finetune/run_heads.py --phase pack
```

默认输入：

- `/home/wqzheng/T3_33_Ternary_Correction/`
- `/home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026/train/checkpoints/best_accepted.pt`
- 上述T3.33输出目录的`feature_cache/<fingerprint>/train.pt`和`validation.pt`。

输入位置变化时，设置 `--correction-package`、`--teacher-run`、`--teacher-checkpoint`。不会猜测其它权重或自动切回旧版本。

默认输出：

- 权重：`/home/wqzheng/project_transformer/outputs/t3_35_ternary_heads_seed2026/train/checkpoints/best_accepted.pt`
- 报告包：`/home/wqzheng/project_transformer/outputs/t3_35_ternary_heads_seed2026_reports.zip`

发回报告包即可。包含训练历史、当前参考/新候选验证预测及指标、最终真实/生成训练预测与分组指标，排除权重和缓存。正式训练目录已存在时拒绝覆盖；重复实验需为check/train/pack统一指定新的`--output-directory`。

如需先冒烟：`run_heads.py --phase smoke --device cpu`。仅2步优化，使用独立smoke目录，不替代正式训练。

## 部署接口

保存权重内嵌原T3.33及原二维主干参数，后续部署无需原T3.33权重文件和训练特征缓存，但仍需原T3.32/T3.33源码。

```python
from run_heads import load_for_inference
model, plot_scale, checkpoint = load_for_inference(
    '/实际路径/best_accepted.pt',
    '/home/wqzheng/T3_32_Image_Recognition',
    '/home/wqzheng/T3_33_Ternary_Correction',
    device='cuda',
)
# 与T3.32相同的输入：image/image_valid_columns/raw_intensity/measured_mask。
# plot_scale保留原训练集拟合结果，不重新拟合，不用标签或条件ID。
output = model(image_inputs)
```

`--phase evaluate --checkpoint ... --split validation`可复评缓存中的验证集。默认没有测试选项，最终模型确定后再单独进行测试。

## 文件

- run_heads.py：依赖审计、缓存读取、检查、真实样本主导训练、验收、保存/重载、部署加载、报告打包。
- head_model.py：复制的DEL/TEB等级头、冻结路由、输出保护、稳定损失、图片部署封装。
- cached_evaluation.py：沿用T3.33的预测表构建和同口径全局loss评价。
- metrics_core.py：沿用T3.32的准确率、混淆矩阵、分组评价、报告输出及调度公式。
- ordinal_core.py：原有CORN条件损失和S/M/H解码。
- cache_audit.py：沿用T3.34的来源、固定划分、形状、有限值和唯一性审计。
- test_heads.py：合成数据单元测试；不读取用户实验数据。
- verify_package.py/checksums.json：补丁文件完整性检查。
- SOURCE_PROVENANCE.json/validation_report.json：代码来源及本地实际验证范围。
