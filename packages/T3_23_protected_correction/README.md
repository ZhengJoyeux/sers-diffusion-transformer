# T3.23：中等级分离与正确边界保护

## 优化依据

T3.22有参考组总体浓度973→982/1152、三元完整组合118→122/216，但DEL310→308/384，水三元完整组合65→62/108。无参考组总体略好，但DEL及水三元仍有退步。土壤三元DEL固定M的36条谱仅18→19条正确，TEB=H时新增一次DEL-M→H错误。

这些结果说明小型修正可以提供一些收益，同时存在把原本正确判断改错的情况；不能把正常混合峰增强判为坏谱，也没有证据说明继续加深encoder或加大参考模块能解决此问题。

本轮两组都屏蔽参考匹配特征，比较普通修正plain与带保护的修正protected。继续冻结T3.18主干，从零修正开始重新训练小支路，不继承T3.22的修正参数。T3.22两组权重保留并做文件指纹保护。

## 具体修改

### 1. 混合谱中等级M的双边界分离

原模型有两个CORN条件输出a、b：

`P(等级>=M)=sigmoid(a)`

`P(等级>=H)=sigmoid(a)*sigmoid(b)`

最终M等级需要前者>=0.5且后者<0.5。这里不能直接把第二个条件概率sigmoid(b)当成累计高等级概率。

新增约束只对真实标签为M的二元/三元训练样本生效，三种药物使用同一规则。令u、v为上述两个累计概率的log odds，新增损失为两边平方hinge的平均：

`middle_loss = mean(relu(0.35-u)^2 + relu(0.35+v)^2) / 2`

它希望M样本处于两个决策边界之间，并留出一定距离，约对应P(>=M)至少0.587、P(>=H)至多0.413。这是软训练目标，不是强制推理规则，模型输出仍由实际光谱决定。

累计高等级的log odds用稳定公式`a+b-logsumexp(0,a,b)`计算，避免直接对接近0/1的概率取logit造成数值问题。它与CORN的概率乘积一致。

### 2. 保护训练集中原本正确的浓度判断

用冻结T3.18在训练样本上的浓度预测与训练标签比较，选出原本正确的存在药物。对它们的两个累计边界：

- 先求原输出在正确一侧的距离。
- 保留的目标距离最多为0.5 log odds；不要求复制原模型极高的置信度。
- 新输出接近或越过错误边界时，加入平方hinge惩罚。
- 原本错误的药物判断不受此保持项约束，允许修正。

保护项初始为0，因为修正初始为0。保护使用训练标签，包括真实和生成训练样本；验证/测试标签不决定优化梯度。推理不使用真实浓度、真实基质或来源信息。

该约束可以减少无必要的错误迁移，但不能保证每条验证/新谱都不会被改错。新增推荐规则用于检查实际验证表现。

### 3. 匹配对照与损失

| 实验 | 训练目标 |
|---|---|
| plain | `0.7*CORN + 0.02*修正均方`，沿用T3.22普通修正目标 |
| protected | plain目标 + `0.1*中等级分离 + 0.2*正确边界保护` |

两组主干、修正网络、输入、归一化、初始化、样本、随机种子、训练轮数和学习率一致，差异只有两项新增损失的权重。这验证的是组合保护策略，不单独证明其中某一项的作用。

输入沿用T3.22的17个原始窗口、形状统计、可用性标记、面积对比和三药上下文。为保持对照与缓存格式一致，18个训练参考缓冲区和72维匹配占位仍构建、保存，但两组都在标准化后屏蔽匹配维度；它们不参与网络预测。固定尺度和标准化只使用真实训练谱拟合。

原模型1,267,823个参数冻结；修正网络40,742个参数训练。修正保持±2的条件logit幅度限制，分类概率不变。主干特征先缓存，50轮中只训练小网络，因此速度仍明显快于从头训练CNN+Transformer。

### 4. 新的推荐检查

推荐相对冻结T3.18起点判断，要求：

1. 三元完整组合正确数必须增加，原值118/216。
2. 土壤三元DEL-M正确数必须增加，原值18/36。
3. 总体、单元、二元、三元，以及水/土壤三元的浓度正确数与完整组合正确数均不得下降。
4. 总体、三元总体、水三元、土壤三元中的DEL/CHL/TEB各自正确数均不得下降。
5. 土壤DEL-M分别按TEB=S/M/H分层，DEL正确数均不得下降，原值5/12、6/12、7/12。
6. 分类概率逐条保持不变。

T3.22两组候选会被这套新规则拒绝，原因包含水三元/DEL退步及TEB=H分层下降。规则更严格，可能整轮没有候选通过；此时recommended.pt回到T3.18零修正，而best_candidate.pt仍保留。这样不能被解释为“模型已经改善”。

此规则不覆盖每一个具体条件、每一条谱，仍须查看new_validation_errors.csv。T3.22历史结果仅作比较，不自动选为本轮fallback，不覆盖旧推荐权重。

## 数据与训练参数

维持每条件真实12/4/4划分，原有200条生成池中固定seed=2026随机48条；总训练1512真实+6048生成=7560，真实验证504。生成/真实样本比例仍为4:1，本轮未同时引入来源重加权。

每组50轮，batch=16，Adam，weight_decay=0，seed=2026。学习率从1e-5开始，预热3轮到1e-4，cosine到1e-6；每轮473步，总23650步。smoke每组2轮，取48真实+48生成训练、48验证，但拟合尺度/标准化仍使用完整真实训练集。

测试集仅检查引用身份，不提取测试特征、不做测试推理、不拟合。仓库构建会读取包含所有原始列的文件，不等于用测试谱训练。验证集用于epoch/方案选择，本轮仍是探索，不能代替最后的独立测试。

## 下载后自动检查并训练

将T3_23_protected_correction.zip放到`/home/wqzheng/`，执行：

```bash
conda activate sers_ddpm
cd /home/wqzheng/project_transformer
unzip -n /home/wqzheng/T3_23_protected_correction.zip -d /home/wqzheng && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_23_protected_correction/run_t323.sh check && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_23_protected_correction/run_t323.sh smoke && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_23_protected_correction/run_t323.sh train
```

check失败或smoke失败会停止后续步骤。默认依次完成protected和plain，各50轮。不会自动安装依赖，使用现有sers_ddpm环境。

正式训练前必须复现：

```text
SOURCE/SUBSET/SPLIT AUDIT: PASS
ORIGINAL VALIDATION REPRODUCTION: PASS (973/1152, 366/504; ternary 118/216)
Variant=protected; trainable=40742; frozen=1267823; planned_updates=23650
```

显存不足可给smoke/train加`--cache-batch-size 16`；修正网络训练batch仍为16。排查加载器可加`--num-workers 0`。只跑一组可用`train --variant protected`或`train --variant plain`，完整比较建议默认两组。

默认核对并保护刚刚完成的T3.22目录：

```text
/home/wqzheng/project_transformer/outputs/t3_22_reference_joint_train_20261009_125205_2457417
```

如果你移动了这个目录，对每个命令加`--previous-run /实际路径`，内容必须仍是上传报告对应的两组结果。原T3.18 baseline、T3_17_encoder_upgrade及原项目数据目录也需要保留。

## 输出和判断

输出位于新的`outputs/t3_23_protected_correction_train_<时间>_<pid>/`。每组在protected/或plain/子目录下。

| 文件 | 查看内容 |
|---|---|
| comparison.md | T3.22历史结果、T3.23两组候选和推荐结果；包括水/土壤三元及DEL-M |
| protected/training_history.csv | 总loss、CORN、M分离、保持项、修正惩罚、验证表现、分组拒绝理由 |
| protected/soil_fixed_DEL_M_nine_conditions.csv | 九组合原模型/候选/推荐的DEL S/M/H数量 |
| protected/new_validation_errors.csv | 被候选或推荐模型新改错的具体谱、药物、真实等级、旧/新等级、修正量 |
| protected/validation_transition_counts.csv | 各药物/基质修正了多少、改错多少、净增益 |
| protected/validation_error_transitions.csv | 所有存在药物的逐谱错误迁移状态 |
| protected/candidate_metrics_by_source.csv | 真实训练、生成训练、真实验证的分组表现 |
| protected/checkpoints/best_candidate.pt | 优先按三元完整组合、其次DEL-M等指标排序的训练候选，可能未通过规则 |
| protected/checkpoints/recommended.pt | 通过规则的修正权重，或原模型零修正版本 |
| plain/ | 同样的输出布局，用于匹配对照 |
| T3_23_protected_correction_reports.zip | 汇总报告，不含checkpoint或缓存 |

训练日志train是加权总目标，val是未加权验证CORN；两者不能按大小直接比较。新增损失会改变总loss尺度，优先比较组件、验证准确率和错误迁移，不能只看loss低不低。

推荐通过规则仍不代表优于T3.22所有指标；报告将两者列出。无候选通过时也不要把best_candidate.pt当成已推荐模型。新的全部权重含冻结主干、尺度/标准化缓冲区和修正支路，可独立加载。

重新加载推荐权重，验证不重新拟合：

```bash
conda activate sers_ddpm
T323_RUN="$(cat /home/wqzheng/T3_23_protected_correction/last_train_path.txt)"
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_23_protected_correction/run_t323.sh evaluate \
  --checkpoint "$T323_RUN/protected/checkpoints/recommended.pt"
```

要检查候选改为best_candidate.pt；要检查对照改为plain/checkpoints/recommended.pt。当前CLI只评价真实验证集，暂不使用测试集选方案。请上传终端给出的T3_23_protected_correction_reports.zip完整报告。

## 恢复

```bash
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_23_protected_correction/run_t323.sh restore
```

restore只核对原项目、数据、T3.18及T3.22权重保持原状。原模型和T3.22从未被覆盖，直接回到旧入口或旧checkpoint即可使用；无需重新训练。T3.23权重使用本包Inference_T3_23.py，不能交给旧推理脚本加载。

## 验证边界

本地检查只使用人工数据与上传报告，不能运行你服务器上的真实训练。VALIDATION.json记录人工模型、损失梯度、完整两组短训练、CLI、旧文件保护和独立推理验证。是否改善以服务器正式报告为准，不保证这一轮解决全部三元误判。
