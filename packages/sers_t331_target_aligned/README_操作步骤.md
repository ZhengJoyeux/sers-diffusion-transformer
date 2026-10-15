# T3.31：从已复核的T3.30继续，分别对齐TEB和DEL目标

本轮基线是T3.30 `concentration/TEB_only` 第17轮，已经在服务器独立重载复核：类别503、完整识别380、浓度等级985、三元125，三元DEL/CHL/TEB分别162/191/163。

默认父权重：

```text
/home/wqzheng/project_transformer/outputs/t3_30_feature_finetune_combined_seed2026/train/concentration/checkpoints/TEB_only/best_accepted.pt
```

两个实验都从这个检查点独立开始。不会沿用未被接受的DEL改变，也不从T3.28整体重新开始。

## 改动与实验

T3.30的联合27状态损失下降，但只有TEB-only部署模式获得净提升。这里把更新目标分开，检查针对实际部署等级监督是否更有效。这是实验假设，不保证提升。

| 实验 | 监督目标 | 原有模块更新 | 其他最终输出 |
|---|---|---|---|
| `teb_corn` | 仅TEB真实等级的条件CORN | 浓度融合、浓度投影、TEB等级头 | DEL、CHL、类别沿用当前T3.30 |
| `del_map` | 仅DEL真实等级的MAP坐标分类 | 浓度融合、浓度投影、DEL等级头 | TEB、CHL、类别沿用当前T3.30 |

TEB从T3.30已经接受的student浓度路径初始化。DEL从T3.30实际提供DEL预测的teacher联合路径初始化。不能把T3.30 student里未接受的DEL变化当作新基线。

主干仍为6层、8头、128维注意力、1901点输入、原210 token路径。CNN、全部Encoder、峰引导Query、多峰证据及归一化均冻结。这次不更新末端Encoder，因为上轮额外更新它没有通过保护规则。

每组参与训练的参数为144,517。新副本只复制浓度融合、投影、目标等级头；其他头复用冻结父模型的模块。**每组完整模型总参数3,244,720**，父模型3,100,203个参数全部冻结。

### 损失与解码

```text
teb_corn：TEB条件CORN + 0.01 × L2-SP
del_map ：DEL的MAP坐标交叉熵 + 0.01 × L2-SP
```

TEB仍按原CORN规则：所有S/M/H监督是否≥M，只有M/H监督是否H；按有效任务数量归一化，部署也使用原有累积概率median解码。

DEL的既有部署等级来自联合MAP坐标。程序把27个联合分数按DEL的S/M/H分成3组，每组取最大分数，再用这3个分数监督真实DEL等级。3组最大分数的argmax恰好等于原27状态全局MAP的DEL坐标；这点已用测试验证。损失不需要真实CHL/TEB浓度目标。组最大分数的softmax不是DEL边缘概率，不应解释为校准置信度。

L2-SP把训练参数约束在本组实际初始化路径附近。没有联合27状态真实组合损失，没有旧等级70/15/15参考，没有预测概率蒸馏，没有标签平滑。真实标签仅用于损失及事后指标，不进入模型、缓存预测接口或路由。

### 缓存不会冻结需要学习的浓度特征

缓存位置在`mixture_aware_query_fusion`入口：每条谱为3个128维Query。每轮仍重新计算可训练浓度融合、投影和目标等级头，梯度可以进入它们。

这与T3.29缓存已经形成的末端浓度特征再训练小纠正头不同。冻结CNN/Encoder等前缀只计算一次，因此正式训练会比T3.30快，但不是跳过训练数据。

缓存包括同一1512真实训练谱、同一6048生成训练谱以及504真实验证谱。真实数据12/4/4划分、固定每条件48条生成谱、标签、原始文件均不变。每轮读取7560条，浓度监督仍只作用于3240条真实标签为三元的样本（真实648、生成2592）。没有测试谱提取或预测。

路由仍只看冻结基线是否预测三药均存在。路由外沿用父模型；路由内仅改本组目标药物。CHL和类别保持父模型输出。验证集上的一元/二元与各药物、两种基质仍由接受规则保护。

## 1. 解压与测试

把ZIP放到`/home/wqzheng`：

```bash
conda activate sers_ddpm

unzip -n /home/wqzheng/SERS_T331_Target_Aligned_20261012.zip \
  -d /home/wqzheng

python /home/wqzheng/SERS_T331_Target_Aligned/test_aligned.py
```

应出现9项测试通过和`OK`。服务器测试默认读取上述真实T3.30第17轮父权重。

本地开发验证没有拿到该T3.30权重文件，使用了已上传的真实T3.28权重构造模拟T3.30父检查点，并使用合成1901点光谱及真实验证预测表检查代码。它不代表已验证新模型的实际性能。服务器的check必须复现真正基线的全部504条身份、等级与概率。

如果父权重路径不同，可给测试加`--checkpoint 实际父权重路径`，后续run命令也加相同`--checkpoint`。必须是已接受的T3.30 concentration/TEB_only第17轮；不能替换成last、last_encoder或其他候选。

## 2. 实际数据检查和特征缓存

```bash
conda activate sers_ddpm
cd /home/wqzheng/project_transformer
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python -u \
  /home/wqzheng/SERS_T331_Target_Aligned/run_aligned.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_31_target_aligned_t330epoch17_seed2026 \
  --phase check --device cuda \
  --batch-size 16 --num-workers 0 --threads 2 \
  2>&1 | tee /home/wqzheng/t3_31_target_check.log
```

必须出现：

```text
BASELINE PASS: class=503 grades=985 full=380 ternary=125; extracting pre-fusion train queries
CHECK PASS: train=7560 (1512 real + 6048 generated); validation=504; test predictions=0
```

check核对父权重版本/第17轮/TEB-only模式/指标/原始teacher权重，逐条复现504条真实验证预测，检查两个目标路径更新前完全保留基线，还要确认二者入口Query相同。

第一次检查成功后记录父权重SHA256，后续阶段严格锁定该文件、缓存、原数据和代码。由于没有收到实际第17轮权重文件，本包不会捏造其SHA256。原始teacher的持久化权重/缓冲也有独立固定校验；重建的位置编码和全部本地缓冲在训练期间保护。

如果检查失败，保留日志并回传错误，不要修改光谱、标签或删掉保护代码。改用新的输出目录重试。

## 3. 小规模运行检查

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  /home/wqzheng/SERS_T331_Target_Aligned/run_aligned.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_31_target_aligned_t330epoch17_seed2026 \
  --phase smoke --arm all --device cuda \
  --batch-size 16 --num-workers 0 --threads 2 \
  2>&1 | tee /home/wqzheng/t3_31_target_smoke.log
```

每组2轮、每轮最多4批次，验证完整504条。smoke不接受模型，也不作为正式训练的起点。

## 4. 正式训练

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  /home/wqzheng/SERS_T331_Target_Aligned/run_aligned.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_31_target_aligned_t330epoch17_seed2026 \
  --phase train --arm all --device cuda \
  --epochs 50 --batch-size 16 --warmup-epochs 3 \
  --num-workers 0 --threads 2 \
  2>&1 | tee /home/wqzheng/t3_31_target_train.log
```

两组独立从当前父模型开始，顺序各训练50轮。每组473更新/轮，总23,650步。FP32；种子2026；梯度裁剪1。

AdamW，weight decay为0，使用L2-SP锚定已接受参数。浓度特征学习率5e-6，目标等级头1e-5；预热3轮后按更新步数余弦退火到最大值的1%。冻结网络与可训练浓度模块的dropout都保持eval行为，参数梯度正常计算。

只使用物理GPU0，不占GPU1。

## 5. 中断后恢复

使用相同正式训练命令，追加`--resume`；可另换日志名：

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  /home/wqzheng/SERS_T331_Target_Aligned/run_aligned.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_31_target_aligned_t330epoch17_seed2026 \
  --phase train --arm all --device cuda \
  --epochs 50 --batch-size 16 --warmup-epochs 3 \
  --num-workers 0 --threads 2 --resume \
  2>&1 | tee /home/wqzheng/t3_31_target_resume.log
```

恢复模型、优化器、学习率、随机数和已选候选；已完成组会跳过，未开始组从同一个父模型开始。从最近完成的轮次恢复，中断轮次重新执行。

## 6. 比较、打包、回传

```bash
python /home/wqzheng/SERS_T331_Target_Aligned/run_aligned.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_31_target_aligned_t330epoch17_seed2026 \
  --phase compare --device cpu

python /home/wqzheng/SERS_T331_Target_Aligned/run_aligned.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_31_target_aligned_t330epoch17_seed2026 \
  --phase pack --device cpu
```

回传：

```text
/home/wqzheng/project_transformer/outputs/t3_31_target_aligned_t330epoch17_seed2026/t3_31_target_aligned_reports.zip
```

报告包含每轮目标损失、验证指标、保护规则、逐条候选预测和第50轮分来源训练表现。末轮来源审计只用于分析，不参与模型选择；打包不含原始光谱、缓存或权重。

接受规则基于当前T3.30：三元正确数严格超过125（至少126），同时总体完整识别≥380、浓度正确≥985、类别及各药物/组合/soil-water保护规则全部通过。不通过就保留当前父模型。新增等级错误仍可能在净计数改善时出现，报告保留修正/破坏数，不能把通过规则解释为逐样本都没有退步。

## 7. 独立重载新权重

新权重自带完整冻结父模型与目标浓度模块，用本包的`load_checkpoint`即可独立推理，目标药物已经写入checkpoint，不需要猜测部署模式。

拿到被接受的模型后：

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  /home/wqzheng/SERS_T331_Target_Aligned/run_aligned.py \
  --suite /home/wqzheng/project_transformer/outputs/t3_31_target_aligned_t330epoch17_seed2026 \
  --phase evaluate --device cuda \
  --batch-size 16 --num-workers 0 --threads 2 \
  --optimized-checkpoint 实际best_accepted.pt路径
```

程序会要求重载结果与保存时的全部关键指标一致。

```python
from aligned_model import load_checkpoint
import torch
model, state = load_checkpoint("实际best_accepted.pt路径", "cuda")
with torch.no_grad():
    presence_probability, grade, cumulative_probability = model.predict(model(spectrum_inputs))
```

输入沿用原6个光谱张量/掩码字段。DEL最终等级仍来自联合MAP坐标，不能把边缘概率再做median解码。缓存接口不接受真实标签或身份字段。

这是探索性验证优化，验证集已参与历史模型选择。没有独立测试提升结论；测试集仍不提取、不预测。
