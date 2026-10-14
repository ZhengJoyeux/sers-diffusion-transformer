# T3.30：现有浓度特征微调与末端Encoder对照

从已验证的 T3.28 DEL-only 模型继续：504条验证光谱，类别正确503、浓度等级正确984、完整识别379、三元等级全部正确124。不会从T3.29失败的纠正头继续训练。

本轮依据：T3.29峰特征头在真实三元训练谱上530→547、生成三元训练谱上2066→2094，但真实三元验证谱124→119。它表现为训练拟合改善、验证退步；没有证明生成谱是唯一原因，也没有证明原始光谱不存在浓度信息。

## 本轮具体修改

两个独立实验都先复制同一个已验证模型，再微调已经存在的模块，不安装新LoRA或Adapter，不叠加T3.29纠正头。

| 实验 | 参与更新的原有模块 | 更新参数量 |
|---|---|---:|
| `concentration` | 浓度投影、跨组分浓度融合、DEL/TEB有序等级头 | 148,807 |
| `last_encoder` | 上述模块，以及第6层Encoder（包含该层已有Adapter） | 285,783 |

实际主干仍是6层、8头、注意力维度128、1901点输入、原来的210 token路径。CNN、峰引导Query、多峰窗口分支及其归一化保持冻结，前5层Encoder保持冻结。第二组仅额外更新第6层，不增加Encoder层数。

为了提供稳定的分类、CHL和回退预测，程序保留冻结基线，并建立可训练的现有识别网络副本。**实验完整模型总参数量为3,100,203，两组相同。** 不是所有参数都训练；冻结基线的1,557,644个参数从不更新。保留两条网络路径会增加训练和推理计算量。

T3.29的“旧等级70%、其余各15%”参考分布和旧等级one-hot输入在本轮完全不使用。原T3.28联合头仍保留并冻结，它接受连续特征和连续概率；真实浓度标签只进入损失，不进入网络输入或路由。

两个实验的损失、数据、学习率设置完全相同，以隔离“最后一层Encoder是否参与更新”的影响。

### 损失函数

仅对真实标签为三元的训练样本计算：

```text
总损失 = 三元27状态交叉熵 + 0.5 × TEB的CORN损失 + 0.01 × L2-SP
```

- 三元状态由DEL、CHL、TEB三个真实S/M/H等级编码成27种组合，直接监督可训练特征。它也向DEL相关模块提供梯度。
- 交叉熵使用0.05标签平滑：95%真实组合目标与5%均匀目标混合。平滑只在损失计算中进行，不修改标签文件、不改等级定义、不改评估标签。它旨在缓和过度确定的拟合，不能保证改善。
- TEB使用其原有两个条件logit：S/M/H都监督“是否≥M”；仅M/H监督“是否H”。损失按有效条件任务数量归一化。额外TEB项对应部署时仍采用的TEB有序解码。
- L2-SP是训练参数相对当前已接受基线的平方偏移之和，限制参数漂移。它约束参数，不要求模型继续相信基线错判的等级。
- AdamW的weight decay为0，避免同时施加“向0衰减”和“向基线权重回拉”；AdamW负责更新，L2-SP负责参数锚定。
- 冻结网络和训练副本都保持eval模式的dropout/归一化行为。eval不会关闭参数梯度；只有冻结基线在no_grad下运行。实际梯度已在测试中验证。

### 预测和保护

- 分类概率来自冻结基线。
- 路由只看冻结分类头是否预测三药均存在，不看真实类别、浓度、基质或文件名。
- 在该路由内，DEL取可训练特征形成的联合MAP坐标，TEB取可训练原有等级头的median解码；CHL仍取冻结基线。
- 非该路由内使用冻结基线。
- 同一组训练每轮自动评估`DEL_only`、`TEB_only`、`DEL_TEB`三种部署模式，不重复训练三次。这延续此前只取DEL可以避免其他药物退步的做法。
- DEL的最终等级可能不等于其边缘概率的median。独立推理必须使用`model.predict()`返回的等级，不能二次解码。

采用新候选要求三元正确数严格超过124，同时总体、浓度总正确数、每种药物、一/二/三元分组以及soil/water三元保护规则都通过。失败候选会报告但不推荐；无候选通过时回退当前T3.28基线。

## 1. 下载、解压、测试

将压缩包放到`/home/wqzheng`：

```bash
conda activate sers_ddpm

unzip -n /home/wqzheng/SERS_T330_Feature_Finetune_20261012.zip \
  -d /home/wqzheng

python /home/wqzheng/SERS_T330_Feature_Finetune/test_feature.py
```

最后应显示8项检查通过、`OK`。本地验证使用你上传的实际基线权重、既有验证预测表和合成光谱；验证代码与保护逻辑，不代表新模型在真实光谱上已改善。服务器正式check才会重新运行504条真实验证谱。

## 2. 正式数据与基线检查

```bash
conda activate sers_ddpm
cd /home/wqzheng/project_transformer
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python -u \
  /home/wqzheng/SERS_T330_Feature_Finetune/run_feature.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026 \
  --phase check --device cuda \
  --batch-size 16 --num-workers 0 --threads 2 \
  2>&1 | tee /home/wqzheng/t3_30_feature_check.log
```

必须出现：

```text
CHECK PASS: class=503 grades=984 full=379 ternary=124; train=7560, real=1512, generated=6048
```

check还会核对两组初始预测与基线逐条一致，并对全部训练谱做基线预测，保存分来源对照。它不更新参数。

原数据仍固定12/4/4；真实训练1512、验证504、测试504；生成训练仍固定126条件各48条，共6048条。全部7560条每轮均读取，三元监督为3240条（真实648、生成2592）。不重选生成谱、不改标签、不重划分、不重拟合归一化，不做测试集预测。

如果基线或文件哈希检查失败，先回传错误，不要改数据或删掉检查。输出目录要使用新目录；check失败后的目录保留，下一次改新名称。

## 3. 小规模运行检查

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  /home/wqzheng/SERS_T330_Feature_Finetune/run_feature.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026 \
  --phase smoke --arm all --device cuda \
  --batch-size 16 --num-workers 0 --threads 2 \
  2>&1 | tee /home/wqzheng/t3_30_feature_smoke.log
```

smoke各运行2轮、每轮最多4个训练批次，验证仍运行全部504条。smoke结果不参与接受，不作为正式训练初始权重。

## 4. 两组正式训练

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  /home/wqzheng/SERS_T330_Feature_Finetune/run_feature.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026 \
  --phase train --arm all --device cuda \
  --epochs 50 --batch-size 16 \
  --warmup-epochs 3 --num-workers 0 --threads 2 \
  2>&1 | tee /home/wqzheng/t3_30_feature_train.log
```

两组顺序运行，各自从T3.28基线开始，各50轮；不是第一组训练完后继续第二组。每组473步/轮、23,650步。默认随机种子2026；FP32；梯度裁剪1。

默认学习率：浓度特征5e-6、DEL/TEB等级头1e-5、第6层Encoder1e-6。按更新步数先预热3轮，再余弦退火至各自最大值的1%。本轮实验重点是更新哪些特征以及如何监督，不是做一批学习率搜索。

这轮重新对光谱进行前向与反向计算，不能使用T3.29冻结特征缓存代替，所以速度会比T3.29慢。物理GPU0用于识别训练；GPU1不使用。

## 5. 断点恢复

如果正式训练中断，使用相同参数加`--resume`：

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  /home/wqzheng/SERS_T330_Feature_Finetune/run_feature.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026 \
  --phase train --arm all --device cuda \
  --epochs 50 --batch-size 16 \
  --warmup-epochs 3 --num-workers 0 --threads 2 \
  --resume \
  2>&1 | tee /home/wqzheng/t3_30_feature_resume.log
```

恢复最近完成轮次的模型、优化器、学习率进度和随机数状态。已完成组会跳过；尚未开始的组仍从原基线开始。恢复不改变训练设置。若中断在某轮中途，该轮从上一轮保存处重新执行。

## 6. 比较与打包

```bash
python /home/wqzheng/SERS_T330_Feature_Finetune/run_feature.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026 \
  --phase compare --device cpu

python /home/wqzheng/SERS_T330_Feature_Finetune/run_feature.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026 \
  --phase pack --device cpu
```

回传：

```text
/home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026/t3_30_feature_reports.zip
```

报告包括每轮训练/验证指标、逐条候选预测、保护规则、第50轮分真实/生成训练表现、只读审计。训练来源审计仅分析，不参与模型选择。打包只收集CSV/JSON/LOG，不含原光谱或权重。

## 7. 保存的权重与推理

每组、每种模式会保存`best_candidate.pt`；只有全部保护规则通过才保存`best_accepted.pt`。`last.pt`用于恢复，不自动部署。

新权重包含完整冻结基线和训练副本，可独立加载，不需要原始Adapter或T3.28权重文件再次拼接。使用本包的加载器；旧`Inference.py`不支持该版本。

```python
import torch
from feature_model import load_checkpoint

model, checkpoint = load_checkpoint("实际被接受的best_accepted.pt路径", "cuda")
with torch.no_grad():
    output = model(spectrum_inputs)
    presence_probability, grade, cumulative_probability = model.predict(output)
```

`spectrum_inputs`沿用原数据集的6个光谱张量/掩码字段，不包含真实标签。部署模式已经写入被接受的checkpoint，加载后不需要猜测。

验证集已经用于历史选择，这次仍是探索性验证改进，不代表独立测试提升。原模型与光谱文件通过哈希保护；所有新权重只写入新实验输出目录。
