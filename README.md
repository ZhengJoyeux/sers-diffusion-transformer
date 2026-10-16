# D26+T24 分支：二维曲线图 DiT 与 SERSFormer

本项目用于 water / soil 中 DEL、CHL、TEB 单组分、双组分和三组分 SERS 光谱的条件生成、识别与浓度等级预测。

本分支的扩散端现已更新为 **D4.28 固定坐标曲线图 DiT**：将完整归一化光谱转换为二维浮点曲线图，在像素空间进行扩散，再解码为数值光谱。识别端保留本分支原有 **T3.24** 代码。分支名继续使用 `D26+T24`，不表示扩散端仍是 D4.26。

## 模型与数据

| 项目 | 当前实现 |
|---|---|
| 扩散骨干 | 纯 Transformer DiT，adaLN-Zero 条件注入；不使用 U-Net 作为当前骨干 |
| 二维输入 | 宽 1920、高 768；1901 个 Raman 采样位置直接对应原有列，剩余列屏蔽 |
| patch | 全高度长方形 768×16，共 120 tokens |
| 主干规模 | hidden 512、12 层、8 头；83,083,776 个参数 |
| 化学条件 | 14 维条件向量与逐样本配对的先验底谱 |
| 当前实验残差模式 | `full_residual`；历史 `broad_local` 模式仍兼容 |
| 数据划分 | 126 文件×20 条，每文件固定 12/4/4；1512/504/504 |
| Raman 有效范围 | 600–2500 cm⁻¹ 的 1901 点，或 600–2000 cm⁻¹ 的 1401 点；使用 valid_mask |
| 当前训练 | 50 epochs、batch size 16 |
| 当前生成 | 每条件 50 条，126 条件共 6300 条；默认物理 GPU 0 |
| 识别端 | 本分支现有 T3.24 SERSFormer，类别识别与 CORN 浓度等级预测 |

PCA、leave-one-out、先验残差、Raman 方差均衡、多样性、尾部约束、生成后校准、EMA、学习率调度及原最终交付保护继续保留。数据集和数值约束仍在光谱空间工作，加噪和 DiT 去噪在二维像素空间工作。

## 二维曲线图的含义

曲线图是没有标题、坐标文字和留白的确定性浮点栅格，不是 matplotlib 截图。所有样本共用固定尺寸、Raman 对应关系和由训练集拟合的纵轴。当前已完成实验的纵轴为全局归一化强度 `[-1.5, 1.5]`；新实验重新执行训练集坐标拟合，不能直接套用旧 checkpoint 的坐标。

每个全高度条带通过一次 Linear 投影提取信息，随后由 Transformer 混合各条带信息。输出图通过可微 soft-argmax 读取每列曲线位置，再进入原有数值光谱流程。栅格和读取存在有限误差，预检查保存往返还原审计；不能声称任意生成图都能无损还原。

## 目录

```text
diffusion/                         # 已整合 D4.27 / D4.28 的扩散源码
  src/dit_1d.py                    # DiT blocks 与历史一维模型
  src/dit_curve2d.py               # 全高度长方形 patch 的二维 DiT
  src/curve_image_codec.py         # 固定坐标栅格与曲线读取
  src/curve_image_diffusion.py     # 二维加噪、训练与采样
  tests/test_d4_28_curve_image.py  # 二维模型与数值接口测试
packages/D4_27_DiT_ablation/        # 从真实 D4.26 配置准备一维对照配置
packages/D4_28_CurveImage_DiT/       # 独立实验准备、自检、训练、生成入口
packages/D4_28_PriorBounds_Fix/     # 固定坐标内的整条先验重抽与审计
transformer/                       # 本分支原有识别代码
current_configs/                   # 历史配置快照；D4.26 配置不等于 D4.28 配置
CURRENT_MODEL.json                 # 当前扩散版本、实际服务器路径、权重交付状态
SOURCE_SHA256.json                 # 源文件 SHA256
docs/archive/D26_T24_README.md      # 更新前 D4.26 / T3.24 详细说明
```

仓库源码已整合，不需要再次运行旧补丁的 `apply_patch.py`。不包含真实数据、服务器 checkpoint、生成光谱或训练输出。`CURRENT_MODEL.json` 中服务器路径仅用于定位文件，不能据此从 GitHub 下载权重。旧一维 checkpoint 不能直接用于二维模型续训。

## 运行环境

服务器使用 Conda `sers_ddpm`、Python 3.11、PyTorch 2.11.0+cu128 和 RTX 3090。依赖见 [diffusion/requirements.txt](diffusion/requirements.txt)。已有可用 PyTorch 时不要为安装其他依赖而替换 CUDA 版本。

以下命令假设仓库位于 `/home/wqzheng/sers-diffusion-transformer`；扩散项目为其 `diffusion` 子目录。克隆仓库不会把服务器现有 `/home/wqzheng/project` 自动替换或迁移。

## 准备与训练

基于服务器实际使用的一维 `full_residual/runtime_config.yaml` 准备独立二维实验：

```bash
conda activate sers_ddpm
cd /home/wqzheng/sers-diffusion-transformer/diffusion

python ../packages/D4_28_CurveImage_DiT/prepare_run.py \
  --project "$PWD" \
  --base-config /home/wqzheng/project/outputs/experiments/d4_27_dit_e50_b16_seed2026_20261013_094005_591560/full_residual/runtime_config.yaml \
  --epochs 50 --batch-size 16 --height 768 --patch-width 16

python ../packages/D4_28_CurveImage_DiT/run_experiment.py \
  --phase check --physical-gpu 0

python ../packages/D4_28_CurveImage_DiT/run_experiment.py \
  --phase train --physical-gpu 0
```

已有实际一维配置的输入数据路径必须能访问。准备程序继承其数据划分、种子、优化器和功能设置，并输出新的 `SUITE`；后续入口默认读取本包 `last_suite.txt`，也可显式传入 `--suite`。`check` 不更新正式模型权重，成功后才允许训练。续训使用 `--phase train --resume`，仅适用于同一二维实验。

如果没有服务器的一维配置，可以从仓库的历史真实 D4.26 快照准备一维对照配置，再取其 full_residual 配置准备二维实验；先确认数据路径适合本机：

```bash
python ../packages/D4_27_DiT_ablation/prepare_run.py \
  --project "$PWD" \
  --base-config ../current_configs/D4.26/cross/runtime_config.yaml \
  --epochs 50 --batch-size 16 --seed 2026

python ../packages/D4_28_CurveImage_DiT/prepare_run.py \
  --project "$PWD" \
  --base-config "$(cat ../packages/D4_27_DiT_ablation/last_suite.txt)/full_residual/runtime_config.yaml" \
  --epochs 50 --batch-size 16 --height 768 --patch-width 16
```

这是从历史真实配置准备新实验，不是本次已完成 D4.28 运行配置的逐字备份。本次服务器 D4.28 runtime_config 与 checkpoint 未上传。

## 固定纵轴先验检查、生成与评价

PCA 随机先验可能偶尔超出训练时固定纵轴。生成入口保留坐标：只对越界的整条先验进行同条件重抽，不逐点裁切、不移动纵轴、不放宽最终交付阈值。它将原先验分布条件化到固定纵轴支持域，拒绝和重抽均保存审计，不能视作完全相同的抽样分布。

训练完成后执行：

```bash
python ../packages/D4_28_PriorBounds_Fix/run_generation.py \
  --suite "$(cat ../packages/D4_28_CurveImage_DiT/last_suite.txt)" \
  --phase check --physical-gpu 0 --number 50

python ../packages/D4_28_CurveImage_DiT/run_experiment.py \
  --phase generate --physical-gpu 0 --number 50

python ../packages/D4_28_CurveImage_DiT/run_experiment.py \
  --phase evaluate --physical-gpu 0 --number 50

python ../packages/D4_28_PriorBounds_Fix/run_generation.py \
  --suite "$(cat ../packages/D4_28_CurveImage_DiT/last_suite.txt)" \
  --phase pack --number 50
```

统一入口的 generate/evaluate 会调用相邻的 `D4_28_PriorBounds_Fix`，使用独立的 `generation_result_n50_prior_bounds.json` 与 `evaluation_n50_prior_bounds/`。直接使用修复包入口也可，其生成、评价策略一致。重抽达到上限或原最终交付预算拒绝时停止，不写虚假完成指针。

源码和配置哈希检查保留。已有服务器实验应继续使用与其 run_index 匹配的原项目和配置；不要用新克隆源码替换旧项目后修改 run_index 来跳过检查。

每条件保存少量实际采样浮点图及预览、列读取诊断和阶段追踪；原始图在数值后处理前保存。最终评价沿用数值光谱指标，不能把归一化光谱 MSE 标成图像像素 MSE。

## 验证与当前质量边界

```bash
cd /home/wqzheng/sers-diffusion-transformer/diffusion
python -m pytest -q \
  tests/test_d4_28_curve_image.py tests/test_d4_27_dit.py \
  tests/test_conditional_prior_residual.py tests/test_d4_25_stage_trace.py \
  tests/test_d4_25_joint_prior_sampling.py tests/test_d4_3_masked_diffusion.py

python ../packages/D4_28_PriorBounds_Fix/test_bounded_prior.py --project "$PWD"
```

本次服务器已完成 50 epoch 训练、126×50 生成和 validation 评价。报告中的最近参考光谱 Pearson 均值为 0.9677、模型归一化光谱 MSE 为 0.001116、成对 MSE 多样性比为 0.9432。当前程序将 122/126 条件标记为至少一侧 QQ 尾部不足。

与一维 full_residual 的 116 个共同合格条件初步比较，二维的一阶导数相关性均值由 0.8755 降至 0.7950，局部峰形细节仍需诊断。一维每条件 100 条，二维 50 条，当前不能声称二维全面优于一维，也未验证浓度预测收益。以上来自用户提供的评价报告；本仓库没有上传训练权重或完整报告。

## 历史模型与许可

历史 U-Net、混合结构和一维 DiT 路径保留供兼容和对照，配置的 `model.architecture` 决定实际构建方式。默认旧主 YAML 不自动改写为二维，二维实验请使用上面的准备入口。

更新前识别模型说明见 [D4.26 / T3.24 历史说明](docs/archive/D26_T24_README.md)，当前 Transformer 目录与其原有实验包保持本分支版本。

DiT 参考与改动说明见 [THIRD_PARTY_NOTICES](diffusion/THIRD_PARTY_NOTICES.md)，参考实现许可证见 [DiT_LICENSE](diffusion/DiT_LICENSE.txt)。
