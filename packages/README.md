# 实验调度与完整模型扩展

| 文件夹 | 简要功能 | 使用状态 |
|---|---|---|
| [d4_27_dit_ablation](d4_27_dit_ablation) | 准备、检查、训练和生成两种残差 DiT；保留安装与测试资源 | 当前生成实验 |
| [sers_t330_feature_finetune](sers_t330_feature_finetune) | 完整识别模型加载、浓度特征微调与部署解码 | 当前推荐识别模型 |
| [sers_t331_target_aligned](sers_t331_target_aligned) | 目标农药对齐的续训对照与候选比较 | 最新对照，未替代 T3.30 |

各包原操作说明、MANIFEST 和 vendor 依赖保持原样。`vendor/archived` 供模型构建、严格权重恢复及校验使用；D4.27 的 `payload` 与 `test_fixtures` 供安装器和测试读取。

[逐文件功能索引](../FILE_GUIDE.md) · [模型组成与正式结果](../README.md)
