# T3.32：二维折线谱图识别补丁

本补丁只新增一个独立识别入口：既有一维残差 DiT 继续输出数值光谱；真实谱和生成谱绘制为二维折线图片，经二维 CNN、Transformer、峰引导 Query、混合物融合，输出 DEL/CHL/TEB 类别及 S/M/H 等级。解压后即可使用，不覆盖原项目源码、权重或原始数据，也不上传 GitHub。

## 默认结构

- 图像：2048×512、单通道灰度、白底黑线，无标题、坐标文字、图例或类别信息。图像在 Dataset 中确定性生成，并缓存 PNG；训练实际读取 image 张量。
- 横轴：固定600–2500 cm⁻¹；所有谱统一1901点。原来无实测数据的区间留白，传递 measured_mask，不将既有补尾噪声绘入图片。
- 纵轴：仅用1512条真实训练谱的实测点拟合一个共享 asinh 强度映射及上下界。保留负值，真实/生成/验证/测试共用尺度，不按单条光谱归一化。超出画布的比例写入报告，不删除样本；数值分支仍保留原始强度。
- 二维 CNN：三个卷积阶段；加入横纵坐标通道，保留8个纵向位置带和曲线纵向矩，输出210个光谱位置 Token。纵向位置表示峰高，不做整图全局平均后直接分类。
- Transformer：6层、8头、128维注意力、256维前馈网络。沿用当前结构规格；输出按横轴光谱位置排列，并使用位置编码。
- 峰引导 Query：DEL/CHL/TEB独立查询，化学先验及训练集辅助先验映射到实际图像 Token。未知区域只能被掩码排除，不把标签放进图像或 forward。
- 多峰数值辅助：保留已有17峰窗口 MultiPeakEvidence，包括局部曲线、峰高、背景及其他强度统计，归一化只用真实训练谱拟合。默认“图像＋数值辅助”；`--image-only` 可做关闭数值分支的对照。
- 混合物融合、Sigmoid类别头、CORN等级头、Residual Adapter保留；跨组合关系监督及等级边界加权保留为可选训练项，默认关闭。
- 从T3.30的student子网络迁移兼容的Transformer、Adapter、查询及任务头权重，二维 CNN 新建。数值归一化重新在本次真实训练集拟合。

旧模型的冻结一维teacher、DEL联合头以及TEB_only路由依赖旧特征分布，不能原样当作已训练的二维模型使用。本补丁采用三种农药直接预测的新二维识别路径，不声称复现旧模型的503/504或其他成绩。旧模型仍可独立作为基线。

## 文件作用

| 文件 | 功能 |
|---|---|
| run_image.py | 检查、冒烟训练、正式训练、评价、导图、结果对比 |
| image_data.py | 训练集共享尺度、灰度折线绘制、实测掩码、PNG缓存 |
| image_model.py | 二维CNN、坐标与纵向特征、Transformer、图像峰先验、T3.30迁移 |
| spectral_core.py | 从已核对源码提取的CORN损失、等级解码、Query与混合物融合 |
| intensity_evidence.py | 既有多峰强度证据模块，保留原实现 |
| adapter.py | 既有Residual Adapter及其层钩子 |
| relations.py | 既有训练集跨组合配对与关系损失 |
| bridge.py | 读取当前Dataset.py，检查12/4/4和生成数据只进训练集 |
| test_image.py | 强度、掩码、梯度、权重往返及数据划分测试 |
| verify_package.py | 核对补丁文件SHA256，防止解压混入旧文件 |
| source_provenance.json | 基础模块来源与源文件SHA256 |
| checksums.json | 补丁文件校验值 |
| requirements.txt | 所需Python包；已有sers_ddpm通常无需重装 |
| validation_report.json | 制作补丁时实际完成的验证及其范围 |

## 1．解压与检查

将zip放在 `/home/wqzheng/`：

```bash
conda activate sers_ddpm
unzip -n /home/wqzheng/SERS_T3_32_2D_Recognition_patch.zip -d /home/wqzheng
python /home/wqzheng/T3_32_Image_Recognition/verify_package.py
python /home/wqzheng/T3_32_Image_Recognition/test_image.py
```

新入口读取 `/home/wqzheng/project_transformer/Dataset.py`（也支持仓库根目录下的transformer/Dataset.py）。默认真实数据是该目录的data/real，生成数据是data/generated。

```bash
CUDA_VISIBLE_DEVICES=0 python -u /home/wqzheng/T3_32_Image_Recognition/run_image.py \
  --phase check --include-generated --maximum-generated-per-condition 48 \
  --batch-size 2 --device cuda
```

默认迁移的权重：

`/home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026/train/concentration/checkpoints/TEB_only/best_accepted.pt`

若存储位置不同，用 `--anchor-checkpoint /实际路径/best_accepted.pt`；若确实要随机初始化，用 `--from-scratch`。缺失权重时不会偷偷切换成随机初始化。

检查输出：`outputs/t3_32_line_image_seed2026/check/check_report.json`、plot_scale.json、split_audit.json、example_input.png。

## 2．冒烟训练

```bash
CUDA_VISIBLE_DEVICES=0 python -u /home/wqzheng/T3_32_Image_Recognition/run_image.py \
  --phase smoke --include-generated --maximum-generated-per-condition 48 \
  --batch-size 16 --device cuda
```

冒烟只进行2个训练批次、2个验证批次，检查正反向、保存及评价，不读取测试样本。使用独立smoke子目录；重复运行需换新的`--output-directory`，不会覆盖之前权重。

## 3．正式训练

```bash
CUDA_VISIBLE_DEVICES=0 python -u /home/wqzheng/T3_32_Image_Recognition/run_image.py \
  --phase train --include-generated --maximum-generated-per-condition 48 \
  --epochs 50 --batch-size 16 --num-workers 2 --device cuda
```

默认seed2026、AdamW、3epoch预热＋余弦退火。新图像模块/Adapter为1e-4，迁移主干1e-5、任务头2e-5、数值证据1e-5；最低比例0.01。迁移主干/Query前3epoch冻结，之后解冻；Adapter与新图像模块持续训练。随机初始化时不冻结主干，主干和任务头使用新模块学习率。

验证集有序准确率优先，其次完整谱正确数、三元正确数和损失选择权重，patience12。正式训练不会自动预测测试集，也不会覆盖旧权重。

默认每条件最多48条生成谱，以便先与当前T3.30的训练数量对照。若想改用D4.27数据，显式指定`--generated-root`和`--maximum-generated-per-condition 100`，并确认126条件均齐全。不混用不同生成来源；若生成集缺少条件，检查直接停止，不自动补造。

保存：

`/home/wqzheng/project_transformer/outputs/t3_32_line_image_seed2026/train/checkpoints/best_ordinal.pt`

## 4．评价与对比

默认最佳权重的验证结果已经写入train/validation，包括predictions.csv、ordinal_metrics.json、各药等级混淆矩阵、按基质/混合复杂度统计。

重新评价验证集：

```bash
CUDA_VISIBLE_DEVICES=0 python -u /home/wqzheng/T3_32_Image_Recognition/run_image.py \
  --phase evaluate --split validation --device cuda \
  --checkpoint /home/wqzheng/project_transformer/outputs/t3_32_line_image_seed2026/train/checkpoints/best_ordinal.pt
```

固定最终方案后，用同一条指令将`--split validation`改成`--split test`进行504条真实测试谱评价。选择权重不能使用测试结果。阶段evaluate在独立evaluation目录输出；等级解码使用权重中保存的策略。

对比旧模型与二维模型：

```bash
python /home/wqzheng/T3_32_Image_Recognition/run_image.py \
  --phase compare --device cpu \
  --baseline-predictions /旧模型同一划分的/predictions.csv \
  --image-predictions /home/wqzheng/project_transformer/outputs/t3_32_line_image_seed2026/train/validation/predictions.csv
```

两份CSV必须具有相同的真实谱标识、划分、标签及matrix；按标识匹配，不依赖行顺序。特别查看三元完整正确数与各药浓度等级，而不是只看类别准确率。

## 可选：导出所有训练谱图片

```bash
python /home/wqzheng/T3_32_Image_Recognition/run_image.py \
  --phase export --split train --include-generated --maximum-generated-per-condition 48 --device cpu
```

先运行check以取得训练集尺度。导出PNG与labels.csv到images/train；图像文件内没有标签文字。训练入口本身使用原始光谱派生图像和数值辅助，不要求预先导出。默认辅助方案不是仅凭PNG推理，需要同时提供对应数值光谱；image_only可做纯图片特征对照。

## 可选对照

- 纯图像：在check/smoke/train中同时添加`--image-only`和新的`--output-directory`。
- 跨组合关系：添加`--relation-weight 0.05`，只从真实训练样本建立三元组。
- 边界约束：添加`--boundary-weight 0.1 --ternary-multiplier 1.5`。

这些选项需独立输出目录，以避免一次同时改变多个因素。

本补丁改变的是识别端的输入和特征提取，不保证效果提高。二维输入会产生像素化误差；数值辅助保留原始强度。当前README中列出的旧模型成绩不能当作本补丁成绩。
