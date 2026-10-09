# T3.24：浓度融合及等级头更新，与残差修正进行对照

本轮要检验：在保留T3.22/T3.23原始谱窗残差修正的基础上，允许浓度专用的跨药物融合和等级输出头小幅更新，是否能改善三元混合的浓度判断。结果需要真实验证谱检验，补丁本身不代表已经提高准确率。

本包不替换项目源码、数据或旧权重。使用原T3.18的8头、6层、attention_dim=128最佳权重作为固定起点，创建独立实验目录。全部模块不从头重训；新增残差网络以零输出起步，浓度融合与等级头副本从T3.18已训练权重初始化。

## 两组分别做什么

| 项目 | concentration_update：实验组 | residual_control：对照组 |
|---|---|---|
| 三分支CNN、Encoder、峰引导Query | 冻结且保持eval | 冻结且保持eval |
| 分类投影和分类头 | 冻结 | 冻结 |
| 128→64浓度投影 | 冻结 | 冻结 |
| 原谱窗残差修正 | 更新 | 更新 |
| 浓度跨药物融合、三个等级头 | 更新独立副本 | 保持原权重 |
| 本轮损失 | 0.7×CORN + 0.02×总logit变化平方 | 相同 |
| 可训练参数 | 185,583 | 40,742 |
| 固定原模型参数 | 1,267,823 | 1,267,823 |
| 包含固定起点的全部参数 | 1,453,406 | 1,308,565 |

实验组新增可训练参数来自混合融合131,971个及等级头12,870个。两个模型共享缓存、原权重、数据、批次顺序、优化器、学习率和损失，差别是实验组额外更新这两个模块。两组参数量不同，不能把结果解释成同等参数量的架构对比。

T3.15/T3.16已经尝试过浓度模块微调，因此本轮不是首次解冻浓度分支。本轮改为当前8头/6层模型、每条件48条固定生成谱，并保留原谱窗修正；检验的是两条路径联合学习是否比同学习率的残差单独学习更有效。之前微调收益有限，所以本轮同样不预先承诺效果。

上一轮T3.23的中等级分离和正确边界保护，在本轮两组中都不使用。参考谱匹配的72个特征在两组中继续屏蔽。现有参考和尺度缓冲仍按兼容流程由真实训练谱拟合，推理时从检查点恢复，不重新拟合。

## 缓存位置和预测方式

共享主干仅运行一次并缓存融合前的Query，形状为[B,3,128]。这384维保留了三个药物各自的表示，让训练中的浓度融合可以重新组合它们。原残差修正仍接收原来的1,266维输入，包括原谱窗和原模型上下文，两组的这部分输入完全相同。缓存共1,650维；对照不使用新增的384维。

实验组预测为：

```
原T3.18等级logit
+ 原谱窗残差修正
+ [更新后浓度融合/等级头输出 − 固定原浓度融合/等级头输出]
```

固定和更新路径使用相同的融合前Query和预测类别概率。起步时更新副本与原权重完全一致，因此第三项严格为零；原谱窗修正也为零，初始化时类别、等级概率和预测均复现原模型。

浓度副本保持eval模式关闭原有dropout，但它的参数保留梯度并由优化器更新；eval不等于no_grad。只有固定原路径使用no_grad。残差网络的dropout策略与对照一致。分类概率直接来自固定原模型，不接收真实类别标签。

实验组的总logit变化包含浓度分支变化和残差修正，因此平方惩罚作用于两者的合计。原残差分量仍受原先的2×tanh界限约束，合计变化没有另设硬界限；学习率、变化惩罚、梯度裁剪和验证采用规则共同约束更新幅度。

## 数据和训练设置

| 设置 | 数值 |
|---|---|
| 真实谱 | 每条件固定12训练、4验证、4测试 |
| 生成谱 | 从原200条中按seed=2026固定随机抽48条/条件 |
| 正式训练集 | 1,512真实 + 6,048生成 = 7,560条 |
| 验证集 | 504条真实谱 |
| 测试集 | 保留504条；本轮不做测试预测或测试特征缓存 |
| Epoch / batch size | 50 / 16，两组各50轮，依次运行 |
| 优化器 | Adam，weight_decay=0 |
| 最高学习率 / 最低学习率 | 1e-5 / 1e-7，两组相同 |
| Warmup | 3轮；初始学习率1e-6 |
| 调度 | Warmup + 余弦退火；每次优化器更新后推进 |
| 预计优化器更新 | 每组473×50=23,650次；warmup为1,419次 |
| 梯度裁剪 | 全部可训练参数的L2范数上限1 |
| 随机种子 | 2026 |
| GPU | CUDA_VISIBLE_DEVICES=0 |
| 输入轴 | 保持600–2500 cm⁻¹，1901点；短轴沿用原内存补尾策略 |

本轮两组统一降低最高学习率，目的是小幅更新已训练浓度模块。T3.22/T3.23采用1e-4的候选结果只作为历史比较，不作为本轮相同学习率的对照。

本轮记录的train_loss为0.7×CORN加变化惩罚，冻结分类不贡献训练损失。它不能直接与从零训练主干时的分类和浓度联合总损失比较。日志分别输出CORN、变化惩罚和浓度分支logit变化。共享主干被缓存，因此训练比全主干重训快是预期行为；参数更新记录会检查额外模块确实发生了更新。

## 下载后一次检查、试跑并自动训练

把ZIP放在`/home/wqzheng/T3_24_concentration_update.zip`，执行：

```bash
conda activate sers_ddpm
cd /home/wqzheng/project_transformer

unzip -n /home/wqzheng/T3_24_concentration_update.zip -d /home/wqzheng && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_24_concentration_update/run_t324.sh check && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_24_concentration_update/run_t324.sh smoke && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_24_concentration_update/run_t324.sh train
```

`&&`确保检查或试跑失败时不会开始正式训练。默认依次运行实验组和对照组，不需要额外输入确认。check只用人工数据测试模型、梯度和检查点。smoke运行两组各2轮，仅用96条训练和48条验证做流程验证；尺度及归一化仍用全部真实训练谱拟合。smoke候选不能被采用。正式训练先复现原模型验证统计，复现失败就停止。

默认需要下面三个已有实验目录：

- T3.18：`/home/wqzheng/project_transformer/outputs/t3_18_random48_train_h8_l6_d128_real_generated_20261008_145007_1824752`
- T3.22：`/home/wqzheng/project_transformer/outputs/t3_22_reference_joint_train_20261009_125205_2457417`
- T3.23：`/home/wqzheng/project_transformer/outputs/t3_23_protected_correction_train_20261009_142431_2503118`

目录来自实际上传报告。若移动过目录，给每个check/smoke/train命令都加同一组路径参数：`--baseline-run /实际T318目录 --previous-run /实际T322目录 --protected-run /实际T323目录`。已经准备过的包不能换源路径；换路径需要重新解压到新包目录。

可以单独运行实验组，例如：

```bash
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_24_concentration_update/run_t324.sh train --variant concentration_update
```

默认仍推荐`both`，让同学习率对照完整运行。若实际服务器提示数据、代码或旧权重与已验证版本不一致，保留错误输出检查原因；本包不会自动修改原文件来绕过检查。

## 查看结果及提交报告

正式结果保存在：

```
/home/wqzheng/project_transformer/outputs/t3_24_concentration_update_train_时间戳/
```

运行后执行：

```bash
T324_RUN="$(cat /home/wqzheng/T3_24_concentration_update/last_train_path.txt)"
cat "$T324_RUN/comparison.md"
ls -lh "$T324_RUN/T3_24_concentration_update_reports.zip"
```

提交最后显示的`T3_24_concentration_update_reports.zip`即可。报告不包含大型模型检查点及特征缓存，包含以下内容：

- `comparison.md`：原模型、旧候选和本轮两组候选/推荐的比较。
- 各组`training_history.csv`：总损失、CORN、变化惩罚、分类/浓度指标、采用原因、优化器更新数及分支变化幅度。
- 各组`soil_fixed_DEL_M_nine_conditions.csv`：九组合逐组DEL/CHL/TEB预测分布。
- 各组`validation_error_transitions.csv`、`new_validation_errors.csv`：修正了哪些错误、新增了哪些错误。
- 各组`candidate_metrics_by_source.csv`：真实训练、生成训练和真实验证分开统计。
- 各组`latest_parameter_changes.csv`：最后一轮各可训练参数的变化；这不是最佳候选的参数变化报告。
- 候选和推荐逐条预测包含总变化、残差变化、浓度分支变化，可检查两条路径是否相互抵消。
- `cache_audit.json`、`selection_audit.json`：缓存位置、拟合范围、样本身份和生成谱抽取审计。

## 推荐及恢复规则

沿用T3.23采用规则，均相对原T3.18判断：

1. 三元完整等级组合正确数必须增加；土壤三元DEL-M正确数必须增加。
2. 总体、单元、二元、三元及水/土壤三元的浓度正确数和完整组合正确数不得下降。
3. 总体及三元各药物正确数不得下降；DEL-M按TEB=S/M/H分层均不得下降。
4. 分类概率保持原模型输出；训练和推理均不根据验证或测试真实标签选择分支。

每组单独保存`best_candidate.pt`；满足规则的最佳轮次保存为`recommended.pt`。没有满足规则的轮次时，`recommended.pt`回到零变化的原模型。新模型与对照的相对高低仍需查看报告；本包不根据两组推荐自动覆盖项目里的默认模型。验证集已用于多轮探索，最终效果需要后续保留的独立测试集评价。

无需覆盖式恢复。检查原文件并确认继续使用原模型：

```bash
conda activate sers_ddpm
bash /home/wqzheng/T3_24_concentration_update/run_t324.sh restore
```

原T3.18检查点依然是：

```
/home/wqzheng/project_transformer/outputs/t3_18_random48_train_h8_l6_d128_real_generated_20261008_145007_1824752/checkpoints/best_ordinal.pt
```

本包自己的检查点保存固定原模型、更新副本、残差网络和拟合缓冲，使用本包推理程序恢复。不要把T3.24检查点直接传给旧Inference.py。

验证本轮推荐检查点示例：

```bash
T324_RUN="$(cat /home/wqzheng/T3_24_concentration_update/last_train_path.txt)"
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_24_concentration_update/run_t324.sh evaluate \
  --checkpoint "$T324_RUN/concentration_update/checkpoints/recommended.pt"
```

evaluate只评价完整真实验证集，不重新拟合尺度、参考谱或归一化参数，不做测试预测。check、smoke、train、evaluate及restore都校验原源码、原数据、T3.18和T3.22/T3.23已有检查点；存在已诊断的T3.20权重时也检查其未变化。
