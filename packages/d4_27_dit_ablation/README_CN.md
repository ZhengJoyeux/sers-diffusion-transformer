# D4.27：纯一维 DiT + broad/local 与完整先验残差对照

本补丁是模型骨架替换和残差学习域对照，不是仅调整网络参数。基础为 ZhengJoyeux/sers-diffusion-transformer 的 D26+T24 分支，commit `5a4e2159d18d56e60ef0e7411b30e60416172f8f`。不改识别端 Transformer。

## 参考实现与一维适配

核对了官方 [facebookresearch/DiT/models.py](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py)、论文 [Scalable Diffusion Models with Transformers](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)，以及 [fast-DiT](https://github.com/chuanyangjin/fast-DiT) 的梯度检查点实现。

新的 `src/dit_1d.py` 按官方 DiT 的 patch embedding、时间嵌入、Transformer block、adaLN-Zero、零初始化输出层结构实现一维光谱适配。原图像实现使用 VAE 潜空间；本项目直接建模 SERS 先验残差，不额外引入 VAE，保留原来 pred_x0、噪声调度和 DDIM/DDPM 采样过程。这是光谱空间 DiT，不能表述成“原官方 ImageNet 模型原封不动移植”。

去噪网络中没有 Conv1d、下采样、上采样、U-Net 跳跃连接或瓶颈 U-Net。patch 用 Linear 投影；每一层都是全局自注意力和 MLP。14 维化学条件经过 MLP 后，与时间嵌入相加，通过 adaLN-Zero 调制每个块和输出层。旧配置中 `conditioning.injection` 的 FiLM 字段保留作为兼容字段，在 DiT 架构下实际使用 adaLN-Zero。先验底谱继续作为输入信息，与带噪残差、有效区掩码一起分块，不丢弃这条条件路径。

## 两组实验

| 项目 | broad_local | full_residual |
|---|---|---|
| 外层条件 PCA 先验 | 保留 | 保留 |
| 训练外层先验 | 原留一交叉拟合 | 同样留一交叉拟合 |
| 原始残差 | 真实谱 − 外层先验谱 | 真实谱 − 外层先验谱 |
| broad/local 高斯拆分 | 原方式 | 不执行 |
| broad PCA 拟合/采样 | 保留 | 不执行 |
| DiT 输入的重建底谱 | 外层先验 + broad | 外层先验 |
| DiT 学习目标 | 缩放 local 残差 | 缩放完整先验残差 |
| 最终重建 | 外层先验 + broad + DiT local | 外层先验 + DiT 完整残差 |

完整残差组继续使用原 robust-asinh 变换和 Raman 方差均衡，但状态在完整残差上仅用训练集拟合；生成时从 checkpoint 读取同一状态进行逆变换。不能把拆分组的 local 缩放状态直接用于完整残差组。

为连接原 bank、full-spectrum support、导数斜率、生成追踪和 checkpoint 读取，保留 `broad_local_residual.enabled: true` 的兼容接口，同时设置 `residual_mode: full_residual`。这个 enabled 表示残差预处理接口启用，并不表示 full_residual 仍在拆分。full_residual 的 broad 输出恒为 0，不运行高斯滤波、broad SVD 或 broad 随机采样。checkpoint 中 broad PCA 的零占位数组只为兼容旧接口，不能统计成有效 broad 主成分。新的 summary 会给出 residual_mode、broad_prior_active 和实际有效组件数。

## 固定条件与默认参数

- 两组均：12 层，hidden_size=512，8 头，patch_size=8，mlp_ratio=4；实测每组去噪网络参数为 **57,922,056**。
- 原有效轴仍是 1901/1401 点；沿用原适配器将 1901 补到 1904 点。DiT 内部如需额外 patch 补齐，会遮蔽补齐点并裁回调用方长度。1904 点对应 238 个 token。
- 生成导出仍遵守基础配置和各条件原始轴，本包合成测试对应长轴 1901 点、短轴 1401 点；没有擅自新增尾部补全。
- 长轴有效 token 238 个；1401 有效点跨越 176 个 patch，最后一块仅含部分有效点。完全无效 patch 不能作为注意力 key，有效区外输出归零。
- 默认 50 epoch、batch size 16，seed 2026；正式运行 GPU=1。保持原学习率、warmup+cosine、EMA、T200/S100、pred_x0、dropout 和 mixed precision 设置。
- 开启梯度检查点，但不自动改变混合精度或学习率。
- 沿用 D26+T24 现有的每源文件 12/4/4 划分；不修改真实数据、标签、归一化方法或划分。它是同源 mapping 光谱拆分，对论文独立样品泛化的证明范围与原实验相同；本补丁没有把它变成 source_file 独立划分。
- 每条件生成 **100 条**。若仍为 126 条件，则每组 12,600 条，两组合计 25,200 条。
- 原峰形/导数、方差、多样性、尾部、负谷、最终保护、校准功能及其开关从实际基础 runtime 配置完整继承；原来关闭的模块保持关闭。
- 不直接改写 `config/ddpm_training.yaml`：该文件在此仓库不是当前 D4.26 正式配置。实验配置必须从当前正式运行的 runtime_config.yaml 复制，仅对核心架构、残差模式、共同训练预算、生成数量和输出位置做明确变更。

两组的预处理学习域不同，因此不能直接凭 scaled DDPM loss 的绝对值大小判断哪组更好。比较必须回到最终完整光谱上的峰形、QQ、Wasserstein、相似性、多样性、近邻复制检查以及下游识别/定量。

## 修改位置

| 文件 | 修改 |
|---|---|
| src/dit_1d.py | 新的纯 DiT 去噪网络 |
| src/dit_configuration.py | DiT 参数和残差模式校验 |
| src/dit_pipeline_check.py | 官方零初始化的延迟梯度验证 |
| src/model_builder.py / build_diffusion_model | 配置选择 DiT；继续调用原 MaskedGaussianDiffusion1D |
| src/broad_local_residual.py | 完整残差模式；状态持久化；旧状态默认原 broad_local 模式 |
| src/conditional_prior_residual.py | 阻止完整残差误用 joint broad sampler；增加真实学习域摘要 |
| src/configuration_loader.py | 接入新配置验证 |
| scripts/start_ddpm_training.py | DiT 续训兼容性检查和管线自检适配 |
| tests/test_d4_27_dit.py | 新结构、掩码、完整残差、train-only 状态、EMA 采样测试 |

旧 U-Net checkpoint 不能加载到 DiT，两组都要从头训练。旧 U-Net 配置和原 broad/local 状态仍可用，未指定 architecture/residual_mode 时保持旧路径。新的 DiT checkpoint 必须配合补丁读取；两组 checkpoint 的残差模式不可交换。训练入口会拒绝跨架构/跨模式续训。

## 第一步：安装、测试、真实数据管线检查

将压缩包上传到 `/home/wqzheng/D4_27_DiT_ablation.zip`，在服务器终端执行：

```bash
conda activate sers_ddpm
cd /home/wqzheng/project

unzip /home/wqzheng/D4_27_DiT_ablation.zip -d /home/wqzheng
python /home/wqzheng/D4_27_DiT_ablation/apply_patch.py --project /home/wqzheng/project
python -m pytest -q tests/test_d4_27_dit.py tests/test_broad_local_residual.py tests/test_conditional_prior_residual.py tests/test_d4_26_hybrid_unet.py
```

安装器先校验全部待修改文件的 SHA256；遇到服务器未提交修改会停止，且不修改任何文件。不要加“强制覆盖”。安装前自动备份到 `project/patch_backups/d4_27_dit_时间/`；不触及 data、outputs、旧 checkpoint 或 Git 历史。再次安装相同补丁是幂等的。

使用仓库记录的当前 D4.26 正式配置创建两组实验：

```bash
python /home/wqzheng/D4_27_DiT_ablation/prepare_run.py \
  --project /home/wqzheng/project \
  --base-config /home/wqzheng/project/outputs/experiments/d4_26_hybrid_e50_b16_seed2026_20261005_153523_720760/cross/runtime_config.yaml \
  --epochs 50 --batch-size 16 --seed 2026

python -u /home/wqzheng/D4_27_DiT_ablation/run_experiment.py \
  --phase check --physical-gpu 1
```

如果该路径不存在，先确认你实际 D4.26 正式运行配置的位置，不要拿历史 D4.2 配置替代。包内 `reference_D4_26_runtime.yaml` 是该分支公开快照，供核对和合成测试使用。

预期：新测试和回归共 39 passed；两组均打印 Pure DiT，正式参数 57922056；两组均打印管线自检通过；完整残差摘要显示 full_residual。DiT 梯度检查使用一次性副本做 3 次小更新，验证零初始化之后梯度能进入化学条件、时间编码、自注意力和 adaLN。正式模型不更新、不保存 checkpoint、不启动正式训练。

第一次安装或 check 报错时停止，把完整输出反馈后再修正。这里不假设你的服务器检查已经通过。

## 通过后：两组训练结束，自动生成两组全部条件

```bash
conda activate sers_ddpm
cd /home/wqzheng/project

python -u /home/wqzheng/D4_27_DiT_ablation/run_experiment.py \
  --phase train_generate --physical-gpu 1
```

顺序是 broad_local 训练 → full_residual 训练 → broad_local 全部条件生成 → full_residual 全部条件生成。两组都从头训练，不复用原 U-Net checkpoint。生成前核对最新 checkpoint 已完成训练预算，再使用各组自己的 best.pt 和 EMA。默认不执行评价。

想指定旧实验而不是最近一次创建的实验，给 runner 加 `--suite /实际实验根目录`。最近创建的路径保存在补丁目录 `last_suite.txt`；不要在运行期间创建别的 suite 后省略 --suite。

中断后恢复同一个 suite：

```bash
python -u /home/wqzheng/D4_27_DiT_ablation/run_experiment.py \
  --suite /实际实验根目录 \
  --phase train_generate --physical-gpu 1 --resume
```

单独运行阶段可用 `--phase train` 或 `--phase generate`。生成恢复时会保留已完成且完整的结果；若已有中断或损坏的生成目录，将写入新的 generated_attempt_时间 目录，不删除旧结果。实际结果路径记录在各组 generation_result.json，评价会跟随该路径。日志保存在各组 check_console.log、training_console.log、generation_console.log；同组重复运行会追加日志。

## 双 GPU：两组独立实验同时运行

使用 `run_parallel.py`，GPU 0 跑 broad_local，GPU 1 跑 full_residual。每张卡独立执行一组完整实验，模型参数不在两组之间共享。每组 batch size 仍为 16、epoch 仍为 50，数据划分、模型参数和生成数量不变。

先按第一步完成安装、测试和 prepare_run；如果已准备好正式 suite，不需要重新创建。更新此包时可用 `unzip -o` 覆盖补丁目录中的运行脚本；包不包含 last_suite.txt，不覆盖你已记录的实验路径。

先同时执行两组真实数据管线检查：

```bash
conda activate sers_ddpm
cd /home/wqzheng/project
python -u /home/wqzheng/D4_27_DiT_ablation/run_parallel.py \
  --phase check --broad-local-gpu 0 --full-residual-gpu 1
```

两组检查通过后，同时训练，训练全部完成后同时生成：

```bash
python -u /home/wqzheng/D4_27_DiT_ablation/run_parallel.py \
  --phase train_generate --broad-local-gpu 0 --full-residual-gpu 1
```

顺序为：两组同时训练 → 等待两组训练全部成功 → 两组各自在对应 GPU 同时生成每条件 100 条。任一训练失败，不进入生成；另一组可以完成正在进行的训练。仅训练而暂不生成时用 `--phase train`。

恢复同一实验：

```bash
python -u /home/wqzheng/D4_27_DiT_ablation/run_parallel.py \
  --suite /实际实验根目录 \
  --phase train_generate --broad-local-gpu 0 --full-residual-gpu 1 --resume
```

两张卡此时都用于扩散实验。程序不停止你已有的其他 GPU 进程。启动前可运行 `nvidia-smi` 查看占用，尤其注意之前用于识别模型的 GPU 0。

各组有独立的 `parallel_train_时间.log`、`parallel_generate_时间.log`，同时保留原 training_console.log 和 generation_console.log。控制台会打印两组 PID、物理 GPU 和日志完整路径；实验根目录有 `parallel_阶段_时间.json` 记录命令、GPU、PID 和退出码。

查看训练输出：

```bash
tail -f /实际实验根目录/broad_local/training_console.log
```

另一终端可查看 full_residual/training_console.log。`--dry-run` 只显示两个进程的命令，不启动训练。`--cpu-threads` 默认为每进程 8；仅影响 CPU 线程数，不改变训练设置。并行入口会锁定同一 suite，防止误启动两次。

双卡验证：3 项并行调度测试通过，实际 CPU 子进程同时执行两组原 CLI 自检通过；GPU 0/1 分配由独立 CUDA_VISIBLE_DEVICES 完成。尚未在你的双 RTX 3090 上执行正式训练。

## 后续评价

```bash
python -u /home/wqzheng/D4_27_DiT_ablation/run_experiment.py \
  --suite /实际实验根目录 \
  --phase evaluate --physical-gpu 1
```

复用原 evaluate_d4_2_generation，按 validation 参考评价全部条件，每组生成 condition_metrics_all.csv；原 QQ、特征峰、分布、多样性等完整报告在各组 evaluation 下。测试集留待方法和参数冻结之后进行最终评价。此 runner 不自动运行识别端或下游对照；它们还需要你真实训练后的谱质量筛选和现有 SERSFormer 实验。

## 本地验证范围

本地为 Python 3.12 / PyTorch 2.14.1 CPU、扩散后端固定 2.2.6。服务器要求仍是你的 Python 3.11 / PyTorch 2.11+cu128 环境，未给你升级依赖。需要在服务器执行上面的测试和真实管线检查，GPU 显存和实际速度尚未测量。

- 39 项新测试及相关旧测试。
- 两个合成条件，包含 1901 和 1401 点，40 条真实样式合成输入，训练/验证/测试 24/8/8。
- 两组实际 CLI 管线检查、1 epoch 小模型训练、checkpoint 保存/重载、EMA 条件采样。
- 每组每条件 100 条、共 400 条合成生成谱的 xlsx 导出；检查各条件原始交付轴（1901/1401 点）及状态中残差模式。
- 正式 57922056 参数 DiT 前向输出检查和无卷积检查。
- 原评价 CLI 在两个合成条件上按 validation 跑通，两组各汇总 2 行，每行 generated_count=100。
- 完成生成后的重复运行跳过完整结果，保留旧文件。
- 安装的校验、备份、重复安装和拒绝覆盖修改测试。

这些是功能测试，合成结果不能作为 SERS 实验性能证据，未在你的真实数据上正式训练。

可在服务器额外重复合成端到端测试（只写新的合成测试目录，不读真实数据）：

```bash
python /home/wqzheng/D4_27_DiT_ablation/smoke_test.py \
  --project /home/wqzheng/project --device cuda --physical-gpu 1
```

执行合成测试会更新 last_suite.txt；正式实验需重新 prepare，或始终明确指定 --suite。

DiT 官方参考许可和署名见 DiT_LICENSE.txt / THIRD_PARTY_NOTICES.md。
