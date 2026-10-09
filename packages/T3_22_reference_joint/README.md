# T3.22：单药参考辅助的联合浓度修正

## 这次解决什么问题

混合谱中，DEL保持M而TEB升高时，1000/1600窗口增强可以是正常的混合响应。本补丁不压平这种变化，不把它判为坏谱，也不把某个窗口直接当成DEL的专属浓度读数。目标是：结合多个窗口、单药参考和三个药物的已有特征，学习区分这些变化对应的浓度等级。

当前T3.18模型已经有MixtureAwareQueryFusion。本次增加训练集单药参考信息，检验它能否提供额外证据；不再仅靠加深encoder、扩大头数或单个峰高纠错。改善只是待验证的假设，完成训练不等于证明问题解决。

默认冻结已经诊断过的T3.18 random48权重。T3.20加权权重如果存在会核对并保护，但不作为本次起点。

## 修改设计

| 部分 | 实现 |
|---|---|
| 原模型 | 原版Model_T3_17.py；8头、6层encoder、attention_dim=128；1,267,823个参数全部冻结，保持eval |
| 参考库 | 水/土壤 × DEL/CHL/TEB × S/M/H，共18个参考；每个仅用该条件12条真实训练谱，合计216条 |
| 新输入 | 17个局部原始强度窗口、局部形状统计、窗口可用标记、7组软对数面积对比、72维参考匹配信息，以及原模型三药query、条件logits和分类概率 |
| 修正网络 | 1266→32→6，GELU，dropout=0.1；40,742个可训练参数 |
| 输出 | 三药分别修正两个CORN条件logits；分类输出保持原值 |
| 初始化 | 最后一层为零，初始修正为零；零修正必须复现原模型 |
| 对照 | reference与no_reference结构、参数量、初始化、样本、顺序种子、损失和训练轮数相同；后者仅在标准化后将72维参考匹配特征置零 |

完整模型共有1,308,565个参数，其中1,267,823个冻结、40,742个训练；参考库和标准化量属于固定缓冲区，不计为可训练参数。

### 1. 参考、强度与缺失范围

窗口中心为813、827、981、1000、1013、1090、1125、1216、1298、1369、1463、1487、1534、1578、1597、1600、2230 cm^-1，沿用现有核心/辅助先验。每个窗口±24 cm^-1，共49个原始采样点。局部统计包括高度、正面积、重心、宽度、不对称度、残差RMS、背景和噪声。

不对每条谱单独做min-max以形成参考输入，而采用真实训练谱拟合的固定窗口尺度和signed asinh压缩，保留正负值及相对强度。原模型本身的旧预处理保持原状。局部背景仅用于统计，不引入broad/local双残差模型，也不要求光谱整体基线发生某种变化。

每个单药参考取12条真实训练谱在压缩空间中的逐点中位数。窗口缺少任意原始测量点时，整个窗口不参与新分支；运行时补到2500 cm^-1的噪声尾不被当作参考测量。原主干仍沿用原来的补尾和预处理，因此其特征可能保留补尾的间接影响。

每条待识别谱都同时与水、土壤两套参考匹配；不向修正网络提供真实基质名称、文件名、条件编号、样本编号、来源或真实浓度。拟合参考时需要训练标签，损失计算与分组报告也使用标签，这些均不属于推理输入。

参考特征包括余弦相似度、压缩后的拟合增益、相对拟合误差、共同测量范围比例。它们是在压缩空间上的辅助匹配证据，不能解释成物理浓度或纯组分含量。没有要求三药谱严格线性相加，也没有谱分解重建损失。相邻窗口会重叠，特征有相关性，实际有用性由对照实验判断。

### 2. 联合修正

修正网络一次读取三个药物的上下文，输出6个独立值。对药物j、条件边界k：

`corrected_logit[j,k] = frozen_logit[j,k] + 2*tanh(network(features)[j,k])`

修正幅度限制在[-2,2]；CORN仍按第一条件概率与两条件概率乘积计算累计概率，保证P(等级≥H)不大于P(等级≥M)。两个条件分别学习修正，不强制用同一个偏移。不只在概率0.5附近启动，因为现有错误可能有较高置信度。

所谓“联合”指修正网络共享三个药物的观测与上下文，不是强制DEL输出不随TEB变化。训练标签指导修正；不存在“TEB升高就减去DEL峰高”的手写规则。

训练目标为`0.7*CORN + 0.02*存在药物的logit修正均方值`。CORN继续只训练存在药物；S样本不计算第二条件边界损失。分类头冻结，因此不重复优化分类损失。修正惩罚用于减少无必要的大幅改动，是本次新增项。

### 3. 数据与训练设置

| 设置 | 值 |
|---|---|
| 每条件真实谱 | 训练12 / 验证4 / 测试4；共126条件 |
| 每条件生成谱 | 从现有200条池中按原条件种子策略seed=2026固定随机48条；不重新生成 |
| 总训练/验证/测试引用数 | 7560 / 504 / 504 |
| 训练来源 | 1512真实 + 6048生成；源文件不改写 |
| 参考尺度、面积平滑下限、特征标准化 | 只拟合1512条真实训练谱；18个参考模板只用其中216条单药谱 |
| 每组正式训练 | 50 epochs，batch=16，Adam，weight_decay=0 |
| 学习率 | 从1e-5预热，3 epochs到1e-4，cosine下降到1e-6；每次optimizer更新推进 |
| 每组更新数 | 473步/epoch，23650步总计；1419步预热 |
| 生成/真实损失权重 | 均为1，维持原设置；生成样本总量仍是真实的4倍，不代表来源已被平衡 |
| 随机种子 | 2026 |

先用冻结主干提取训练/验证特征缓存，再只训练小网络。两组共用同一缓存，不是各自从头训练两遍主干。缓存保留固定主干的实际预处理，修正网络自身的dropout在训练时开启。

smoke每组2轮，训练取48真实+48生成、验证48条；但参考、尺度与标准化仍严格按完整真实训练集拟合。smoke只能验证流程，不能采用其训练候选。

测试集只核对样本引用，不提取测试特征、不做测试推理。仓库构建需要读取含全部谱列的原文件，这不等于把测试谱用来拟合或选择模型。验证集用于选择epoch，仍然需要之后一次独立测试才能判断最终泛化。

## 服务器操作

把T3_22_reference_joint.zip下载并放到`/home/wqzheng/`。保持目前的数据、原项目和T3_17_encoder_upgrade文件夹。

### 一条命令：检查、短跑、自动正式训练

```bash
conda activate sers_ddpm
cd /home/wqzheng/project_transformer
unzip -n /home/wqzheng/T3_22_reference_joint.zip -d /home/wqzheng && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_22_reference_joint/run_t322.sh check && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_22_reference_joint/run_t322.sh smoke && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_22_reference_joint/run_t322.sh train
```

使用`&&`，任何检查或短跑失败均停止后续步骤。默认正式训练依次完成reference和no_reference两个50轮实验。

需要逐步运行时，分别执行上面的check、smoke、train三条bash命令即可。已解压成功以后，不必重复unzip。此包不自动安装依赖；沿用sers_ddpm已有torch/numpy/pandas/scipy等环境。

正式训练在开始修正优化之前必须输出：

```text
SOURCE/SUBSET/SPLIT AUDIT: PASS
ORIGINAL VALIDATION REPRODUCTION: PASS (973/1152, 366/504; ternary 118/216)
Variant=reference; trainable=40742; frozen=1267823; planned_updates=23650
```

同时核对诊断过的原checkpoint及模型state指纹、17个窗口中心。无法复现时直接停止，避免在不同起点上解释结果。不要绕过这些检查。

正式起点是：

```text
/home/wqzheng/project_transformer/outputs/t3_18_random48_train_h8_l6_d128_real_generated_20261008_145007_1824752/checkpoints/best_ordinal.pt
```

显存紧张时，给smoke和train命令加`--cache-batch-size 16`。这只调节冻结主干特征提取的batch，修正网络训练batch仍为16。可用`--num-workers 0`排查数据加载问题。若只想运行一组，使用`train --variant reference`或`train --variant no_reference`；正式比较建议默认两组。

## 结果、推荐规则与后续推理

输出创建在新的`outputs/t3_22_reference_joint_train_<时间>_<pid>/`，不会覆盖旧实验。终端最后显示报告ZIP的完整路径。

| 文件 | 含义 |
|---|---|
| comparison.md / suite_summary.json | 原模型、两组最佳训练候选、两组推荐结果的比较 |
| selection_audit.json / reference_fit_audit.json / cache_audit.json | 样本、拟合范围和冻结核对 |
| reference/training_history.csv | 每轮损失、总体/三元性能、学习率、修正幅度、是否通过推荐规则 |
| reference/candidate_validation_predictions.csv | 最佳训练候选逐谱预测及6个修正量 |
| reference/validation_metrics_by_condition.csv | 每条件原模型/候选/推荐的等级计数与组合正确数 |
| reference/soil_fixed_DEL_M_nine_conditions.csv | 重点九组合：DEL固定M，CHL与TEB各S/M/H，比较DEL被判S/M/H的变化 |
| reference/candidate_metrics_by_source.csv | 真实训练、生成训练、真实验证，按单/二/三元和基质分组 |
| reference/checkpoints/best_candidate.pt | 验证集三元完整组合优先排序的训练候选，可能退步 |
| reference/checkpoints/recommended.pt | 通过保守规则的修正模型；否则是零修正的原模型 |
| no_reference/ | 同样的文件布局，去掉参考匹配特征的对照 |
| T3_22_reference_joint_reports.zip | 分享分析用报告；不含权重和特征缓存 |

推荐规则按正确样本数比较：三元完整组合必须比原模型118/216增加；总体浓度正确数、总体完整组合、单元/二元完整组合、三元浓度正确数、土壤三元完整组合均不得降低；分类概率保持不变。规则不只看总准确率，也不允许用单元/二元明显损失交换三元收益。未通过时，训练候选仍保存用于分析，但recommended.pt自动选原模型零修正版本。smoke始终选择零修正。

`accepted=True`仅表示通过上述验证规则，不代表已经通过独立测试，也不代表每一个组合都改善。小验证集上的少数样本差异不足以证明稳定优势；先比较两组并检查重点九组合，再决定是否做重复种子实验。

每个T3.22 checkpoint包含原主干、固定参考、固定尺度、特征标准化和修正网络，可独立加载，不需要重新拟合参考。

正式训练完成后，重新加载推荐权重验证：

```bash
conda activate sers_ddpm
T322_RUN="$(cat /home/wqzheng/T3_22_reference_joint/last_train_path.txt)"
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_22_reference_joint/run_t322.sh evaluate \
  --checkpoint "$T322_RUN/reference/checkpoints/recommended.pt"
```

这条命令只评价真实验证集。要检查训练候选，改为`best_candidate.pt`；要检查对照，改为`no_reference/checkpoints/recommended.pt`。当前CLI不提供测试集评价，避免选择方案期间重复使用测试集。模型类的forward可供后续业务推理集成。

## 快速恢复和文件保护

```bash
conda activate sers_ddpm
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_22_reference_joint/run_t322.sh restore
```

补丁不会覆盖原项目代码、Excel/CSV、T3.18 checkpoint或已有T3.20 checkpoint。restore是核对原状的只读操作，旧一维模型始终可用。回到原训练入口：

```bash
cd /home/wqzheng/project_transformer
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_17_encoder_upgrade/run_t318_random48.sh train
```

这最后一条会启动新的原版训练；如果只是继续使用旧模型，直接使用上面列出的旧best_ordinal.pt即可，无需重新训练。

所有正式命令前后核对原代码、数据、checkpoint与隔离快照。不允许用T3.22 checkpoint交给旧版Inference_T3_17.py加载，二者结构不同；使用本包Inference_T3_22.py。

## 本地验证范围

VALIDATION.json记录人工数据检查、冻结与检查点往返、两组短训练、CLI保护/恢复、Python3.11语法及shell检查。人工测试不包含服务器真实光谱或真实权重训练。模型是否提升，需要运行正式训练并查看comparison.md和九组合表。

本包不重新生成扩散谱，不改变现有48条抽样，不转换二维图片，不声称参考匹配就能唯一解混。若两组都无改善，应根据候选的分组错误及修正量继续诊断；这次对照能先判断单药参考是否提供了实际增益。
