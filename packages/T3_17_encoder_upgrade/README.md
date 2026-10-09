# T3.17：扩维注意力模型

基于本次上传的 SERSFormer_current_state.zip 制作。实验属于结构容量调整，不是已验证的性能提升或新的论文核心创新。

## 模型与维度

| 阶段 | 张量形状 |
|---|---|
| 现有三分支 CNN 及融合 | B × 210 × 64 |
| 学习型线性扩维 | B × 210 × 128 |
| 128 维位置编码、6 层 Encoder（8 头）、FFN（内部256维） | B × 210 × 128 |
| 峰引导 Query Attention（8 头） | B × 3 × 128 |
| 分类分支：降维，再接原有分类头 | B × 3 × 64 → B × 3 |
| 浓度分支：128 维混合 Query 融合，再降维，再接原有 CORN 头 | B × 3 × 128 → B × 3 × 64 → B × 3 × 2 |

所有三个注意力模块都使用128维、8头，每头16维。CNN输出和预测头输入维度仍为64。1901光谱点、210个token、有效区掩码和补尾策略保持当前 Dataset.py 的实现。

扩维不创造新的测量信息，只改变可学习表示及网络容量。新增层与宽度会增加计算量和过拟合风险。结构内部维度必须一致，不能把注意力头数理解成不同农药或不同特征峰的固定数量。

训练从随机初始化开始，更新完整模型，不使用T3.10作冻结锚点。50 epoch，batch-size 16，学习率1e-4，3 epoch线性warmup，随后逐optimizer更新进行余弦退火至1e-6；早停patience=50。两任务保留现有完整训练入口的DWA策略。保持CORN、median解码、ordinal checkpoint选择、训练集拟合峰先验。局部定量、anchored calibration和边界额外损失均不新增。

完整模式加载每条件全部200条最终D4.26生成谱；预期train=26712（real=1512，generated=25200），validation=504，test=504。仅计数test，不进行test推理。smoke模式仍构建全部数据，但每epoch只运行8个训练batch和4个验证batch。

## 文件

- Model_T3_17.py：在上传Model_v2.py基础上增加可选attention_dim和扩维/两分支降维，注意力层使用扩维特征。
- T3_17_Training.py：在上传完整训练入口基础上增加扩维配置、逐更新warmup/cosine和实际模型参数检查。
- Inference_T3_17.py：从model_config还原新模型，保持当前评价逻辑。
- T3_17_LR.py：无PyTorch依赖的学习率计算。
- verify_t317.py：源文件版本校验及人工数据前向、掩码、完整训练更新、旧模型关闭扩维兼容、新checkpoint推理还原测试。
- test_t317_static.py：6项无PyTorch依赖的学习率和CLI/编译检查。
- run_t317.sh：check/smoke/train/baseline四种模式。

升级包直接在自身目录运行，不覆盖服务器的Model_v2.py、Dataset.py、SERSFormer_Training.py或Inference.py，不修改已有checkpoint与输出目录。将包解压到/home/wqzheng后，从当前项目调用脚本。脚本比较本次上传源码的归一化文本SHA256；如源码改变会停止，不自动覆盖。

## 先执行检查与迷你训练

```bash
conda activate sers_ddpm
cd /home/wqzheng/project_transformer
unzip -n /home/wqzheng/T3_17_encoder_upgrade.zip -d /home/wqzheng
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_17_encoder_upgrade/run_t317.sh check
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_17_encoder_upgrade/run_t317.sh smoke
```

检查应显示SOURCE SNAPSHOT: PASS、10项检查OK和T3.17 ARTIFICIAL MODEL CHECK: PASS。smoke应打印heads=8、encoder_layers=6、encoder_dim=128、head_dim=16、tokens=210，warmup_updates=8、planned_updates=16，并训练2个epoch保存独立checkpoint。这只能验证工程流程，不能证明性能提升。请先反馈完整输出，再继续正式实验。

## 正式训练与对照（检查通过后执行）

```bash
conda activate sers_ddpm
cd /home/wqzheng/project_transformer
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_17_encoder_upgrade/run_t317.sh train
```

建议另做同一新数据、同一训练日程的当前结构基线：

```bash
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_17_encoder_upgrade/run_t317.sh baseline
```

baseline使用64维、4头、4层、FFN内部64维；新模型使用128维、8头、6层、FFN内部256维。两者都完整训练，不能把历史T3.10旧生成数据结果当成严格的新数据对照。本次是多个容量参数的联合比较，结果不能单独归因于扩维、头数或层数；若改善，再分别做对应消融。

评估真实validation上的分类F1、浓度等级准确率、三组分组合全对率、DEL/TEB的M等级表现、water/soil及严重等级错误。不得把S/M/H编码的RMSE/R²解释为实际物理浓度的定量结果。

## checkpoint与推理

新扩维checkpoint新增投影权重及attention_dim配置，注意力层尺寸也改变，不能直接用旧四层checkpoint续训。不能用服务器原Inference.py加载新扩维模型。

用包内专用入口（将CHECKPOINT和OUTPUT换成实际路径）：

```bash
conda activate sers_ddpm
cd /home/wqzheng/project_transformer
CUDA_VISIBLE_DEVICES=0 python /home/wqzheng/T3_17_encoder_upgrade/Inference_T3_17.py \
  --checkpoint CHECKPOINT --output-directory OUTPUT \
  --split validation --device cuda --batch-size 32 --num-workers 4 --ordinal-decoding auto
```

关闭扩维时新增模块为Identity，保持旧state_dict键结构，旧checkpoint可由新推理入口还原。包中包含严格加载及输出相等的服务器测试。

## 本地验证范围

生成环境未安装PyTorch，也没有用户的真实谱或checkpoint。已经完成6项静态检查、全部Python源码编译、Bash语法检查和源码变更核对。4项PyTorch模型测试和真实数据smoke须在sers_ddpm服务器环境执行；此处不声称已通过。
