# 生成端：条件先验残差 DiT

本目录负责光谱读取、条件 PCA 先验、残差处理、DiT 扩散训练、生成校准、保护和评价。当前正式版本为 D4.27，保留 broad/local 与完整残差两种对照。

| 文件夹 | 简要功能 |
|---|---|
| `src/` | 生成模型与光谱处理模块 |
| `scripts/` | 训练、条件生成、评价与诊断入口 |
| `tests/` | 网络、数据、先验与保护回归测试 |
| `config/` | 基础及历史对照配置 |

- 当前核心：`src/dit_1d.py`，由 `src/model_builder.py` 构建并连接扩散过程。
- 训练入口：`scripts/start_ddpm_training.py`；条件生成：`scripts/generate_conditional_spectra.py`。
- 正式两组配置：[D4.27](../current_configs/D4.27)。旧默认配置不会自动变成 DiT。
- 实验准备与调度：[D4.27 实验包](../packages/d4_27_dit_ablation)。
- [逐文件功能索引](../FILE_GUIDE.md) · [模型组成、流程图与结果](../README.md)

归档不包含数据与训练权重。旧 U-Net 代码供兼容与对照，不是当前 DiT 核心。
