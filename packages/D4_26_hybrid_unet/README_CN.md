# D4.26：卷积 U-Net + 实际底谱条件 Transformer 候选补丁

这是可应用、可回滚的新架构候选。代码验证通过不代表生成质量已经改善。服务器真实数据训练和 CUDA 检查需要按下面步骤执行。

## 1. 这次配对结果决定了什么

`evidence/` 内保留你上传的原始 summary、paired comparison 和 report，未改写其数值。

| 具体条件 | 独立→联合：MSE | 独立→联合：QQ standardized RMSE | 独立→联合：一阶导数 PCC | 独立→联合：pairwise MSE median ratio |
|---|---:|---:|---:|---:|
| DEL-H_water | 1581.4260→1567.8192 | 0.186039→0.184628 | 0.950356→0.949941 | 0.592071→0.589301 |
| CHL-M_TEB-M_water | 189.5166→192.7014 | 0.150931→0.158796 | 0.894274→0.892396 | 0.679996→0.700237 |
| DEL-S_TEB-S_CHL-S_soil | 239.2298→236.5913 | 0.243729→0.242478 | 0.698077→0.698570 | 0.857625→0.853386 |

三个条件的初始采样随机状态、outer 抽样和训练参考配对检查均通过；六次终点 CPU 回放通过，最大绝对误差约 3.05e-5。

联合采样确实改善了 outer/broad score 相关性的拟合，但没有一致改善生成质量。双组分 QQ RMSE 增加约 5.21%；三组分导数 PCC 增量约 0.00049。这里没有依据继续增加联合采样强度，也没有依据宣布 Transformer 一定会修好峰形。

因此新实验冻结 **A_independent** 的全部生成配置，joint_prior_sampling 保持 false。只改变去噪器中的新结构。DEL 和双组分在 A/B 中都出现 QQ diversity_floor_satisfied=false，新报告继续保存这个标志，以及全部后处理阶段的平均 pairwise MSE。

注意：上述表的 diversity 是随机配对距离的中位数比值；QQ guard 的日志使用平均 pairwise MSE 比值。两者不是同一个指标，不能直接用一个指标的 0.86 门槛评判另一个。日志说明已有 QQ 步骤存在多样性折损，但不能仅凭这三个条件推断全部 126 条件。

## 2. 实现细节

原 U-Net 已有各尺度 LinearAttention 和瓶颈 Attention。本补丁保留这些模块，在原瓶颈 attention 后、mid_block2 前加入一个完整条件 Transformer adapter。

| 项目 | 首版设置 |
|---|---|
| 卷积与 skip connections | 保留，原参数名不变 |
| 原来的 14 维条件编码、全层 FiLM | 保留 |
| 新 Transformer | depth=1，dim=128，heads=4，FF multiplier=2，dropout=0.05 |
| 自注意力 | residual bottleneck token 的全局关系 |
| 交叉注意力 Q | 残差去噪瓶颈 token |
| 交叉注意力 K/V | 本条样本实际使用的 reconstruction base，经有效区池化和投影 |
| 条件调制 | diffusion timestep embedding + 原 chemical condition embedding |
| 位置编码 | 固定 Raman 物理轴 Fourier 编码；不对短轴重新缩放 |
| 无效区 | 新 adapter 的无效 key 排除、query update 归零；部分边界 token 按覆盖率衰减 |
| 输出投影 | 初始 weight/bias 为零，逐步学习修正 |
| 参数 | 原 denoiser 1,346,881；cross 候选 1,688,385；新增 341,504，约 +25.35% |

模型物理轴仍是 1901 点，U-Net 输入补到 1904。四倍下采样后有 476 个位置：1401 条件有效 351 个 token，1901 条件有效 476 个 token；边界只有一个有效点时 coverage=0.25。短轴末端中心为 2000 cm^-1，长轴末端中心为 2500 cm^-1。新 adapter 的 mask 不会把缺失尾部当成实测 token。旧 backbone attention 的实现保持原样，本补丁没有将其全部改写为 masked attention。

新结构不拟合 PCA，不重新抽底谱，也不使用 validation/test 拟合先验。训练和生成沿用原 pipeline 传入的同一 `prior_conditioning` 张量。默认关闭时不增加参数键、不消耗额外初始化随机数，旧 checkpoint 可严格加载。开启时需要新训练；不能用旧 best.pt 的权重冒充新结构已训练。resume 会拒绝 baseline/self/cross 跨结构续训。

零输出投影使初始化时 eval 输出与旧网络一致。第一步新 attention 内部梯度为零是这个初始化的预期行为；投影开始更新后梯度打开。合成检查执行三次内存优化，再验证 attention 和所有 FiLM 梯度，避免错误要求第一步新 attention 就必须非零。

这是一项架构实验：保持数据划分、seed、训练光谱数、优化步数、损失、先验和后处理相同，但参数量与计算量增加，不能称为 FLOPs 匹配的对比。

## 3. 包内文件

- `payload/`：四个已有文件的更新，以及两个新模块、一个 13 项结构测试文件。
- `hybrid_changes.diff`：完整逐行改动。
- `apply_fix.py` / `rollback_fix.py`：SHA256 预检、备份、原子写入、幂等应用及回滚。
- `prerequisites.json`：要求你当前已应用的终点保护、stage trace、joint sampler 生产代码版本。
- `prepare_run.py`：读取旧训练配置和已完成的 A/B review，建立独立实验 suite。
- `verify_hybrid.py`：合成优化/掩码/EMA检查，或 checkpoint EMA 严格加载检查。
- `run_experiment.py`：check、train、review 分阶段运行、日志、续训和配对检查。
- `test_package_workflow.py` / `test_fixtures/`：可复现的包流程测试，使用临时项目与合成缓存。
- `local_verification.txt`：本地验证范围、版本与结果。

应用只写入七个 src/scripts/tests 文件，不修改原训练配置、checkpoint 或实验结果。每次运行 suite 后会检查冻结的源代码与配置 hash，防止中途变更后继续输出貌似公平的比较。版本不符时会在修改前停止并指出文件，不提供强制覆盖开关。

## 4. 上传、应用与测试

先下载 **D4_26_hybrid_unet.zip** 并上传到服务器 `/home/wqzheng/`。附件不会自动出现在服务器目录。

```bash
conda activate sers_ddpm
cd /home/wqzheng/project

ls -lh /home/wqzheng/D4_26_hybrid_unet.zip &&
unzip -o /home/wqzheng/D4_26_hybrid_unet.zip -d /home/wqzheng &&
python /home/wqzheng/D4_26_hybrid_unet/apply_fix.py \
  --project /home/wqzheng/project &&
python -m unittest discover -s tests \
  -p test_d4_26_hybrid_unet.py -v &&
python -m pytest -q
```

本地完整源代码测试结果为 251 passed；其中新结构测试 13 项。包流程测试另有 7 项全部通过。你的服务器若增加其他测试，总数可能不同。以全部通过为准。留存 `APPLY PASS` 输出的 backup 路径。

可选包流程测试：

```bash
D4_26_TEST_PROJECT=/home/wqzheng/project \
python -m unittest discover -s /home/wqzheng/D4_26_hybrid_unet \
  -p test_package_workflow.py -v
```

## 5. 只准备一次 suite，并执行 CUDA 与真实管线检查

```bash
conda activate sers_ddpm
cd /home/wqzheng/project

EXP='/home/wqzheng/project/outputs/experiments/d4_25_full_seed2026_20261003_155014_938728'
PAIR="$EXP/joint_prior_review_20261005_133522_721916"

python /home/wqzheng/D4_26_hybrid_unet/prepare_run.py \
  --project /home/wqzheng/project \
  --baseline-experiment "$EXP" \
  --paired-review "$PAIR" \
  --epochs 25 --seed 2026 &&
SUITE="$(cat /home/wqzheng/D4_26_hybrid_unet/last_suite.txt)" &&
python -u /home/wqzheng/D4_26_hybrid_unet/run_experiment.py \
  --suite "$SUITE" --phase check --variant cross --physical-gpu 1
```

PAIR 必须指向你本次已完成的 joint review 目录；该目录包含 `A_independent.yaml`、`joint_review_index.json`、`joint_prior_report.json`。脚本核对原报告配对通过、原 checkpoint/config SHA256、seed=2026、25 epoch、12/4/4、dim32/[1,2,4]、T200/S100。

SUITE 为新建的 `.../outputs/experiments/d4_26_hybrid_seed2026_时间戳`，内有 baseline/self/cross 配置。baseline 复用原 best.pt；self 配置预备给后续消融；首轮只训练 cross。新模型仍由全部 126 条件的 1512 条 train 光谱共享训练。三条件仅用于第一轮生成评价。

check 先做 CUDA 上三次内存优化，不保存训练 checkpoint；再调用现有 `--pipeline-check-only` 做真实管线检查。必须看到：

```text
HYBRID SYNTHETIC CHECK PASS
Transformer depth=1; heads=4; dim=128; tokens=476
actual-base cross attention=True
条件先验数量：126
共享DDPM训练光谱：1512
每条件训练光谱数：[12]
14维条件：已激活
全层FiLM梯度：非零
```

第一份日志检查新 attention 梯度，第二份日志检查现有真实数据管线。不能将二者合称新 attention 已在真实 126 条件上完成效果验证。

## 6. 正式训练 cross

无需再次执行 prepare_run.py。新终端从 last_suite.txt 读取同一个路径。

```bash
conda activate sers_ddpm
cd /home/wqzheng/project

SUITE="$(cat /home/wqzheng/D4_26_hybrid_unet/last_suite.txt)"
python -u /home/wqzheng/D4_26_hybrid_unet/run_experiment.py \
  --suite "$SUITE" --phase train --variant cross --physical-gpu 1
```

采用原 seed2026、25 epoch、batch8、2条件×4、warmup+cosine、原全部 D4.25 full 损失和先验配置。预计仍为 189 step/epoch，共 4725 step。候选 checkpoint 保存到 `$SUITE/cross/checkpoints/`；训练完成后自动严格加载候选 best EMA 并检查 bank 的126条件/1512训练谱。

需要中断续训时：

```bash
SUITE="$(cat /home/wqzheng/D4_26_hybrid_unet/last_suite.txt)"
python -u /home/wqzheng/D4_26_hybrid_unet/run_experiment.py \
  --suite "$SUITE" --phase train --variant cross --physical-gpu 1 --resume
```

已有 checkpoint 而未给 --resume 时脚本拒绝覆盖。baseline 不会重新训练。另一终端可观察：

```bash
watch -n 1 'nvidia-smi -i 1'
```

环境由 runner 设置 `CUDA_VISIBLE_DEVICES=1`，进程内显示 `cuda:0` 对应物理 GPU1。

## 7. 三条件配对生成、分阶段评价

训练完成后：

```bash
conda activate sers_ddpm
cd /home/wqzheng/project

SUITE="$(cat /home/wqzheng/D4_26_hybrid_unet/last_suite.txt)"
python -u /home/wqzheng/D4_26_hybrid_unet/run_experiment.py \
  --suite "$SUITE" --phase review \
  --variants baseline cross --physical-gpu 1
```

脚本要求每个训练分支的 latest.pt 已到 step4725，再使用各自 best.pt/EMA。原 baseline 和 cross 各生成三条件×200条，首轮共1200条诊断谱，不是全126条件25200条。

三个条件固定为 DEL-H_water、CHL-M_TEB-M_water、DEL-S_TEB-S_CHL-S_soil。每个条件单独重设同一 seed2026，生成 batch size、T200、实际S100和全部后处理参数相同。

两种结构均保存完整 stage cache：00底谱、01DDPM重建、02PCA spread、03QQ、04mean、05旧envelope、06最终support、07terminal。01 已包含 reverse diffusion 内部 guard；它不是完全无约束的 U-Net 输出。

评价分别读取 **01_ddpm_reconstruction** 和 **07_terminal**，参考使用对应条件的四条真实 validation。训练参考仍是对应12条 train。validation/test 不参与先验拟合和生成后处理；test 暂不用于这轮选候选。

配对报告写出前，脚本逐项检查：

- 20个batch、共200条的 CPU/CUDA sampling RNG fingerprint 完全相同；
- 实验条件、物理轴、train参考、outer/broad/base抽样完全相同；
- normalization、training indices、support、delivery和生成配置完全相同；
- 缓存有限、维度正确、EMA/T200/S100与checkpoint step相符；
- CPU final replay 通过。

评价包含 MSE、Cosine、PCC、Wasserstein、QQ、导数PCC、内部距离与Pearson多样性、峰位/峰高/峰宽变化范围、训练包络越界比例、最近训练样本距离及相对heldout距离。完整原评估器输出还包含各峰和分布明细、叠图、QQ图、peak diversity/violin图。

主要文件：

```text
$SUITE/pilot_review/hybrid_paired_comparison.csv
$SUITE/pilot_review/hybrid_stage_summary.csv
$SUITE/pilot_review/hybrid_report.json
$SUITE/pilot_review/evaluations/<variant>/<condition>/<stage>/
$SUITE/cross/training_console.log
$SUITE/cross/logs/training_log.csv
```

再次执行同一 review 命令可复用完整、签名一致的生成缓存和评价结果。若只有部分生成目录，脚本保留它并停止；把明确报出的部分目录改名留存后重试，不要覆盖完整缓存。只有缓存已完整而需要离线重做评价时才使用 --evaluate-only。

## 8. 怎么决定下一步

先把上述三个 CSV/JSON 和 cross 的 training_log.csv 发回。至少查看两阶段、三条件是否一致改善，不把约1%的单条件MSE变化当成功。

- 01已改善、07保留改善：支持继续做self消融及126条件覆盖验证。
- 01已改善、07变差或QQ继续压缩多样性：有证据指向后处理，需要单独处理QQ，不能把架构收益与后处理损失混在一起。
- 01没有改善：本候选没有证明去噪器有效，不继续堆层数或宣布峰形修复。
- 近训练距离明显缩小而validation相似性/多样性不改善：检查记忆风险，不能靠更接近train宣布模型更好。

三条件只是固定pilot，不能代表全部126。没有预设“必须改善5%”之类缺少不确定性依据的成功阈值；四条validation和单一seed的证据有限。候选如果有一致收益，再做126条件、短/长轴、水/土、1/2/3组分覆盖、最差条件、以及多seed验证。

self消融仍接受原底谱输入通道和FiLM，只关闭新增adapter的base cross attention，用于区分新增自注意力/FFN与实际底谱交叉注意力的贡献。可在cross初步有效后执行：

```bash
SUITE="$(cat /home/wqzheng/D4_26_hybrid_unet/last_suite.txt)"
python -u /home/wqzheng/D4_26_hybrid_unet/run_experiment.py \
  --suite "$SUITE" --phase check --variant self --physical-gpu 1 &&
python -u /home/wqzheng/D4_26_hybrid_unet/run_experiment.py \
  --suite "$SUITE" --phase train --variant self --physical-gpu 1 &&
python -u /home/wqzheng/D4_26_hybrid_unet/run_experiment.py \
  --suite "$SUITE" --phase review --variants baseline self cross --physical-gpu 1
```

生成谱仍只用于诊断，质量验证后再决定是否加入下游Transformer训练集；下游validation/test继续只使用真实数据。

## 9. 回滚

若需要恢复原架构代码，使用apply输出的具体备份目录：

```bash
python /home/wqzheng/D4_26_hybrid_unet/rollback_fix.py \
  --project /home/wqzheng/project \
  --backup /home/wqzheng/project/outputs/upgrades/d4_26_hybrid_实际时间戳
```

回滚前检查新代码未被后续修改；不删除旧或新实验产物。新cross/self checkpoint需要本补丁架构才能加载，恢复补丁后仍可使用。
