# 文件与文件夹功能索引

本索引覆盖当前分支的全部文件与目录。功能说明用于快速定位；模块是否启用以正式配置和完整加载器为准。
同名 vendor、payload、fixtures 文件有固定依赖用途，不按同名或版本号判定为冗余。

## 文件夹

| 文件夹 | 简要功能 |
|---|---|
| [current_configs](current_configs) | 正式生成配置与当前推荐识别结构、结果快照。 |
| [current_configs/D4.26](current_configs/D4.26) | 准备 D4.27 所用的旧基础配置快照。 |
| [current_configs/D4.26/cross](current_configs/D4.26/cross) | D4.26 正式基础 runtime 配置。 |
| [current_configs/D4.27](current_configs/D4.27) | broad/local 与完整残差两组正式 DiT 配置。 |
| [current_configs/T3.30](current_configs/T3.30) | 当前已接受识别模型的结构与验证摘要。 |
| [diffusion](diffusion) | 条件先验残差 DiT 的生成、训练、评价和兼容工具。 |
| [diffusion/config](diffusion/config) | 基础与历史对照配置；当前正式配置另见 current_configs/D4.27。 |
| [diffusion/scripts](diffusion/scripts) | 生成端训练、条件生成、评价与诊断入口。 |
| [diffusion/src](diffusion/src) | 生成网络、先验残差、掩码扩散与光谱保护模块。 |
| [diffusion/tests](diffusion/tests) | 生成端网络、数据、先验及保护行为回归测试。 |
| [packages](packages) | 配对实验调度、完整识别扩展与必需 vendor 依赖。 |
| [packages/d4_27_dit_ablation](packages/d4_27_dit_ablation) | D4.27 两种残差的实验准备、调度、安装与检查。 |
| [packages/d4_27_dit_ablation/evidence](packages/d4_27_dit_ablation/evidence) | 保存补丁包的正式结构和本地验证摘要。 |
| [packages/d4_27_dit_ablation/payload](packages/d4_27_dit_ablation/payload) | 哈希安装器使用的 D4.27 更新内容，保留固定路径。 |
| [packages/d4_27_dit_ablation/payload/scripts](packages/d4_27_dit_ablation/payload/scripts) | 哈希安装器使用的 D4.27 更新内容，保留固定路径。 |
| [packages/d4_27_dit_ablation/payload/src](packages/d4_27_dit_ablation/payload/src) | 哈希安装器使用的 D4.27 更新内容，保留固定路径。 |
| [packages/d4_27_dit_ablation/payload/tests](packages/d4_27_dit_ablation/payload/tests) | 哈希安装器使用的 D4.27 更新内容，保留固定路径。 |
| [packages/d4_27_dit_ablation/test_fixtures](packages/d4_27_dit_ablation/test_fixtures) | 安装器与回归检查使用的固定夹具。 |
| [packages/d4_27_dit_ablation/test_fixtures/originals](packages/d4_27_dit_ablation/test_fixtures/originals) | 安装测试必需的更新前源码夹具，不是待删除备份。 |
| [packages/d4_27_dit_ablation/test_fixtures/originals/scripts](packages/d4_27_dit_ablation/test_fixtures/originals/scripts) | 安装测试必需的更新前源码夹具，不是待删除备份。 |
| [packages/d4_27_dit_ablation/test_fixtures/originals/src](packages/d4_27_dit_ablation/test_fixtures/originals/src) | 安装测试必需的更新前源码夹具，不是待删除备份。 |
| [packages/sers_t330_feature_finetune](packages/sers_t330_feature_finetune) | 当前推荐识别模型的完整加载器与微调入口。 |
| [packages/sers_t330_feature_finetune/reference](packages/sers_t330_feature_finetune/reference) | 保存实验复核所需的验证预测参考表。 |
| [packages/sers_t330_feature_finetune/vendor](packages/sers_t330_feature_finetune/vendor) | 按原实验结构保存的运行依赖，避免加载错误版本。 |
| [packages/sers_t330_feature_finetune/vendor/base](packages/sers_t330_feature_finetune/vendor/base) | 冻结 DEL-only 组合基线及其依赖。 |
| [packages/sers_t330_feature_finetune/vendor/base/vendor](packages/sers_t330_feature_finetune/vendor/base/vendor) | 按原实验结构保存的运行依赖，避免加载错误版本。 |
| [packages/sers_t330_feature_finetune/vendor/base/vendor/t328](packages/sers_t330_feature_finetune/vendor/base/vendor/t328) | 三元联合头、冻结基线桥接和实验保护依赖。 |
| [packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor) | 按原实验结构保存的运行依赖，避免加载错误版本。 |
| [packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327) | 已接受 Adapter 结构、优化器和学习率调度依赖。 |
| [packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived) | 保留当前模型及清单校验所需的兼容骨架、多峰和关系工具。 |
| [packages/sers_t331_target_aligned](packages/sers_t331_target_aligned) | 最新目标对齐对照；当前仍推荐其 T3.30 父模型。 |
| [packages/sers_t331_target_aligned/reference](packages/sers_t331_target_aligned/reference) | 保存实验复核所需的验证预测参考表。 |
| [packages/sers_t331_target_aligned/vendor](packages/sers_t331_target_aligned/vendor) | 按原实验结构保存的运行依赖，避免加载错误版本。 |
| [packages/sers_t331_target_aligned/vendor/t330](packages/sers_t331_target_aligned/vendor/t330) | T3.31 内嵌的完整 T3.30 父模型和校验资源。 |
| [packages/sers_t331_target_aligned/vendor/t330/reference](packages/sers_t331_target_aligned/vendor/t330/reference) | 保存实验复核所需的验证预测参考表。 |
| [packages/sers_t331_target_aligned/vendor/t330/vendor](packages/sers_t331_target_aligned/vendor/t330/vendor) | 按原实验结构保存的运行依赖，避免加载错误版本。 |
| [packages/sers_t331_target_aligned/vendor/t330/vendor/base](packages/sers_t331_target_aligned/vendor/t330/vendor/base) | 冻结 DEL-only 组合基线及其依赖。 |
| [packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor) | 按原实验结构保存的运行依赖，避免加载错误版本。 |
| [packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328) | 三元联合头、冻结基线桥接和实验保护依赖。 |
| [packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor) | 按原实验结构保存的运行依赖，避免加载错误版本。 |
| [packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327) | 已接受 Adapter 结构、优化器和学习率调度依赖。 |
| [packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived) | 保留当前模型及清单校验所需的兼容骨架、多峰和关系工具。 |
| [transformer](transformer) | 基础 SERSFormer、数据适配、训练和分层评价。 |
| [transformer/tests](transformer/tests) | 基础识别模型、峰引导、等级预测和数据保护测试。 |

## 文件

<details>
<summary>.：7 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [.gitignore](.gitignore) | 忽略训练数据、权重、运行输出与缓存，避免误提交。 |
| [ARCHIVE_VERIFICATION.json](ARCHIVE_VERIFICATION.json) | 记录归档时的源码、实验包哈希和语法核验结果。 |
| [CLEANUP_REPORT.json](CLEANUP_REPORT.json) | 记录已删除的旧文件、保留依赖和清理验证结果。 |
| [CURRENT_MODEL.json](CURRENT_MODEL.json) | 记录当前模型版本、推荐识别权重位置、参数和评价口径。 |
| [FILE_GUIDE.md](FILE_GUIDE.md) | 按目录逐项说明所有文件与文件夹的简要功能。 |
| [README.md](README.md) | 介绍系统、两种模型、机理流程、正式参数与使用入口。 |
| [SOURCE_SHA256.json](SOURCE_SHA256.json) | 保存当前仓库文件的 SHA256，核对归档内容是否一致。 |

</details>

<details>
<summary>current_configs：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [README.md](current_configs/README.md) | 说明本目录作用、关键入口与完整文件功能索引。 |

</details>

<details>
<summary>current_configs/D4.26/cross：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [runtime_config.yaml](current_configs/D4.26/cross/runtime_config.yaml) | D4.27 实验准备所用的 D4.26 正式基础配置快照。 |

</details>

<details>
<summary>current_configs/D4.27：2 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [broad_local_runtime_config.yaml](current_configs/D4.27/broad_local_runtime_config.yaml) | 正式 broad/local DiT 配置：学习 local 残差并采样 broad 先验。 |
| [full_residual_runtime_config.yaml](current_configs/D4.27/full_residual_runtime_config.yaml) | 正式完整残差 DiT 配置：不做 broad/local 分解。 |

</details>

<details>
<summary>current_configs/T3.30：2 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [parent_model_config.json](current_configs/T3.30/parent_model_config.json) | 记录当前已接受识别模型的主干、Adapter 与联合头参数。 |
| [validation_summary.json](current_configs/T3.30/validation_summary.json) | 记录 T3.30 推荐模型在真实验证集上的复核结果。 |

</details>

<details>
<summary>diffusion：9 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [.gitignore](diffusion/.gitignore) | 忽略训练数据、权重、运行输出与缓存，避免误提交。 |
| [Dataset.py](diffusion/Dataset.py) | 保留光谱读取、公共轴与识别预处理的兼容适配器。 |
| [README.md](diffusion/README.md) | 说明本目录作用、关键入口与完整文件功能索引。 |
| [analyze_peak_free_background.py](diffusion/analyze_peak_free_background.py) | 仅用训练谱识别无峰背景区及其噪声、负值支持。 |
| [apply_support8_negative_guard_hybrid_paired.py](diffusion/apply_support8_negative_guard_hybrid_paired.py) | 运行历史 support8 混合负值保护的成对诊断对照。 |
| [apply_support8_negative_guard_paired.py](diffusion/apply_support8_negative_guard_paired.py) | 运行历史 support8 负值保护的成对诊断对照。 |
| [pyproject.toml](diffusion/pyproject.toml) | 声明生成项目的构建、依赖和 pytest 配置。 |
| [requirements.txt](diffusion/requirements.txt) | 列出生成项目安装依赖。 |
| [run.py](diffusion/run.py) | 保留生成项目的通用命令调度入口；当前正式实验使用专用脚本。 |

</details>

<details>
<summary>diffusion/config：54 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [ddpm_training.yaml](diffusion/config/ddpm_training.yaml) | 兼容基础扩散训练配置；不会自动切换为当前 DiT。 |
| [ddpm_training_d3_new_single.yaml](diffusion/config/ddpm_training_d3_new_single.yaml) | 历史对照配置：d3_new_single；实际参数以文件内容为准。 |
| [ddpm_training_d4_0_b_mask_smoke.yaml](diffusion/config/ddpm_training_d4_0_b_mask_smoke.yaml) | 历史对照配置：管线检查；实际参数以文件内容为准。 |
| [ddpm_training_d4_1_formal.yaml](diffusion/config/ddpm_training_d4_1_formal.yaml) | 历史对照配置：d4_1_formal；实际参数以文件内容为准。 |
| [ddpm_training_d4_1_smoke.yaml](diffusion/config/ddpm_training_d4_1_smoke.yaml) | 历史对照配置：管线检查；实际参数以文件内容为准。 |
| [ddpm_training_d4_24_soft_variance_power05_25ep.yaml](diffusion/config/ddpm_training_d4_24_soft_variance_power05_25ep.yaml) | 历史对照配置：方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_24_soft_variance_power05_smoke2ep.yaml](diffusion/config/ddpm_training_d4_24_soft_variance_power05_smoke2ep.yaml) | 历史对照配置：管线检查、方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_2_formal.yaml](diffusion/config/ddpm_training_d4_2_formal.yaml) | 历史对照配置：d4_2_formal；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_10_30min.yaml](diffusion/config/ddpm_training_d4_3_2_10_30min.yaml) | 历史对照配置：d4_3_2_10_30min；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_10_30min_tailoff.yaml](diffusion/config/ddpm_training_d4_3_2_10_30min_tailoff.yaml) | 历史对照配置：尾部关闭对照；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_11_30min.yaml](diffusion/config/ddpm_training_d4_3_2_11_30min.yaml) | 历史对照配置：d4_3_2_11_30min；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_12_generation.yaml](diffusion/config/ddpm_training_d4_3_2_12_generation.yaml) | 历史对照配置：生成校准；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_13_independent_full_spectrum.yaml](diffusion/config/ddpm_training_d4_3_2_13_independent_full_spectrum.yaml) | 历史对照配置：d4_3_2_13_independent_full_spectrum；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_13_medium.yaml](diffusion/config/ddpm_training_d4_3_2_13_medium.yaml) | 历史对照配置：d4_3_2_13_medium；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_14_cross_fitted_pca_medium.yaml](diffusion/config/ddpm_training_d4_3_2_14_cross_fitted_pca_medium.yaml) | 历史对照配置：留一交叉先验；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_14_negative_guard_generation.yaml](diffusion/config/ddpm_training_d4_3_2_14_negative_guard_generation.yaml) | 历史对照配置：生成校准、负值保护；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_14_raw_generation.yaml](diffusion/config/ddpm_training_d4_3_2_14_raw_generation.yaml) | 历史对照配置：生成校准；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_14b_robust_negative_guard_generation.yaml](diffusion/config/ddpm_training_d4_3_2_14b_robust_negative_guard_generation.yaml) | 历史对照配置：生成校准、负值保护；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_15_raw_generation.yaml](diffusion/config/ddpm_training_d4_3_2_15_raw_generation.yaml) | 历史对照配置：生成校准；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_15_warmup_cosine.yaml](diffusion/config/ddpm_training_d4_3_2_15_warmup_cosine.yaml) | 历史对照配置：d4_3_2_15_warmup_cosine；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_15_warmup_cosine_2h_formal.yaml](diffusion/config/ddpm_training_d4_3_2_15_warmup_cosine_2h_formal.yaml) | 历史对照配置：d4_3_2_15_warmup_cosine_2h_formal；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_15a_shape_preserving_generation.yaml](diffusion/config/ddpm_training_d4_3_2_15a_shape_preserving_generation.yaml) | 历史对照配置：生成校准；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_16_variance_profile_1h.yaml](diffusion/config/ddpm_training_d4_3_2_16_variance_profile_1h.yaml) | 历史对照配置：方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_16_variance_profile_formal.yaml](diffusion/config/ddpm_training_d4_3_2_16_variance_profile_formal.yaml) | 历史对照配置：方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_16_variance_profile_raw.yaml](diffusion/config/ddpm_training_d4_3_2_16_variance_profile_raw.yaml) | 历史对照配置：方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_16_variance_profile_smoke.yaml](diffusion/config/ddpm_training_d4_3_2_16_variance_profile_smoke.yaml) | 历史对照配置：管线检查、方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_17_group_variance_formal.yaml](diffusion/config/ddpm_training_d4_3_2_17_group_variance_formal.yaml) | 历史对照配置：方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_18_1_budget010_formal.yaml](diffusion/config/ddpm_training_d4_3_2_18_1_budget010_formal.yaml) | 历史对照配置：辅助损失预算；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_18_train_reference_tail_formal.yaml](diffusion/config/ddpm_training_d4_3_2_18_train_reference_tail_formal.yaml) | 历史对照配置：d4_3_2_18_train_reference_tail_formal；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_19_variance_stratified_tail_formal.yaml](diffusion/config/ddpm_training_d4_3_2_19_variance_stratified_tail_formal.yaml) | 历史对照配置：方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_20_high_noise_recovery_formal.yaml](diffusion/config/ddpm_training_d4_3_2_20_high_noise_recovery_formal.yaml) | 历史对照配置：d4_3_2_20_high_noise_recovery_formal；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_21_raman_variance_equalization_50ep.yaml](diffusion/config/ddpm_training_d4_3_2_21_raman_variance_equalization_50ep.yaml) | 历史对照配置：方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_21_sampling200_eval.yaml](diffusion/config/ddpm_training_d4_3_2_21_sampling200_eval.yaml) | 历史对照配置：200 步采样、评价；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_22_min_snr5_raman_variance_equalization_25ep.yaml](diffusion/config/ddpm_training_d4_3_2_22_min_snr5_raman_variance_equalization_25ep.yaml) | 历史对照配置：min-SNR 权重、方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_22_min_snr5_raman_variance_equalization_50ep.yaml](diffusion/config/ddpm_training_d4_3_2_22_min_snr5_raman_variance_equalization_50ep.yaml) | 历史对照配置：min-SNR 权重、方差约束；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_1_gated_residual_tail_min_snr5_25ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_1_gated_residual_tail_min_snr5_25ep.yaml) | 历史对照配置：min-SNR 权重、残差尾部；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_1_gated_residual_tail_min_snr5_smoke2ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_1_gated_residual_tail_min_snr5_smoke2ep.yaml) | 历史对照配置：管线检查、min-SNR 权重、残差尾部；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_2_batch12_group4_severe_tail_only_min_snr5_25ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_2_batch12_group4_severe_tail_only_min_snr5_25ep.yaml) | 历史对照配置：min-SNR 权重；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_2_batch12_group4_severe_tail_only_min_snr5_smoke2ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_2_batch12_group4_severe_tail_only_min_snr5_smoke2ep.yaml) | 历史对照配置：管线检查、min-SNR 权重；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_2_batch12_group6_severe_tail_only_min_snr5_25ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_2_batch12_group6_severe_tail_only_min_snr5_25ep.yaml) | 历史对照配置：min-SNR 权重；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_2_batch12_group6_severe_tail_only_min_snr5_smoke2ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_2_batch12_group6_severe_tail_only_min_snr5_smoke2ep.yaml) | 历史对照配置：管线检查、min-SNR 权重；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_2_dim64_severe_tail_only_min_snr5_25ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_2_dim64_severe_tail_only_min_snr5_25ep.yaml) | 历史对照配置：min-SNR 权重；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_2_dim64_severe_tail_only_min_snr5_smoke2ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_2_dim64_severe_tail_only_min_snr5_smoke2ep.yaml) | 历史对照配置：管线检查、min-SNR 权重；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_2_severe_tail_only_min_snr5_25ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_2_severe_tail_only_min_snr5_25ep.yaml) | 历史对照配置：min-SNR 权重；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_2_severe_tail_only_min_snr5_smoke2ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_2_severe_tail_only_min_snr5_smoke2ep.yaml) | 历史对照配置：管线检查、min-SNR 权重；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_residual_tail_min_snr5_25ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_residual_tail_min_snr5_25ep.yaml) | 历史对照配置：min-SNR 权重、残差尾部；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_23_residual_tail_min_snr5_smoke2ep.yaml](diffusion/config/ddpm_training_d4_3_2_23_residual_tail_min_snr5_smoke2ep.yaml) | 历史对照配置：管线检查、min-SNR 权重、残差尾部；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_2_smoke.yaml](diffusion/config/ddpm_training_d4_3_2_2_smoke.yaml) | 历史对照配置：管线检查；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_4_smoke.yaml](diffusion/config/ddpm_training_d4_3_2_4_smoke.yaml) | 历史对照配置：管线检查；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_9_formal_tailoff_eval.yaml](diffusion/config/ddpm_training_d4_3_2_9_formal_tailoff_eval.yaml) | 历史对照配置：评价、尾部关闭对照；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_9_smoke.yaml](diffusion/config/ddpm_training_d4_3_2_9_smoke.yaml) | 历史对照配置：管线检查；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_2_9_tailoff_diagnostic.yaml](diffusion/config/ddpm_training_d4_3_2_9_tailoff_diagnostic.yaml) | 历史对照配置：尾部关闭对照；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_formal.yaml](diffusion/config/ddpm_training_d4_3_formal.yaml) | 历史对照配置：d4_3_formal；实际参数以文件内容为准。 |
| [ddpm_training_d4_3_smoke.yaml](diffusion/config/ddpm_training_d4_3_smoke.yaml) | 历史对照配置：管线检查；实际参数以文件内容为准。 |

</details>

<details>
<summary>diffusion/scripts：27 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [analyze_peak_free_background.py](diffusion/scripts/analyze_peak_free_background.py) | 仅用训练谱识别无峰背景区及其噪声、负值支持。 |
| [calibrate_broad_shape_tail.py](diffusion/scripts/calibrate_broad_shape_tail.py) | 按训练参考对宽背景形态与尾部进行软校准。 |
| [calibrate_pointwise_broad_tail.py](diffusion/scripts/calibrate_pointwise_broad_tail.py) | 对宽背景逐点极端尾部进行稳健校准。 |
| [diagnose_d2_6_timestep_recovery.py](diffusion/scripts/diagnose_d2_6_timestep_recovery.py) | 诊断 broad/local 残差在不同噪声时间步的恢复表现。 |
| [diagnose_ddpm_residual_distribution.py](diffusion/scripts/diagnose_ddpm_residual_distribution.py) | 检查生成残差与真实训练残差的分布差异。 |
| [diagnose_pca_prior_distribution.py](diffusion/scripts/diagnose_pca_prior_distribution.py) | 检查 PCA 先验的分布、变动范围和多样性。 |
| [diagnose_real_vs_generated_sers_distribution.py](diffusion/scripts/diagnose_real_vs_generated_sers_distribution.py) | 比较真实训练谱和生成谱的强度、峰形与分布。 |
| [diagnose_sampling_diversity_collapse.py](diffusion/scripts/diagnose_sampling_diversity_collapse.py) | 定位反向采样中的多样性收缩阶段。 |
| [diagnose_timestep_recovery.py](diffusion/scripts/diagnose_timestep_recovery.py) | 诊断不同扩散时间步的重建与去噪能力。 |
| [evaluate_d4_1_checkpoint.py](diffusion/scripts/evaluate_d4_1_checkpoint.py) | 评价 D4.1 条件生成 checkpoint 的兼容入口。 |
| [evaluate_d4_2_checkpoint.py](diffusion/scripts/evaluate_d4_2_checkpoint.py) | 评价条件先验残差 checkpoint 的兼容入口。 |
| [evaluate_d4_2_generation.py](diffusion/scripts/evaluate_d4_2_generation.py) | 按指定真实 split 评价条件生成谱及分层质量指标。 |
| [evaluate_generated_spectra.py](diffusion/scripts/evaluate_generated_spectra.py) | 提供通用真实/生成光谱比较入口。 |
| [filter_generated_spectra_by_training_guard.py](diffusion/scripts/filter_generated_spectra_by_training_guard.py) | 按训练支持检查生成光谱并输出筛选与诊断结果。 |
| [generate_conditional_spectra.py](diffusion/scripts/generate_conditional_spectra.py) | 逐条件生成光谱，依次校准、保护并导出获准结果。 |
| [generate_spectra.py](diffusion/scripts/generate_spectra.py) | 提供单组光谱采样、重建与导出的基础生成入口。 |
| [inspect_spectrum_data.py](diffusion/scripts/inspect_spectrum_data.py) | 查看真实光谱的数据规模、格式、轴和强度统计。 |
| [prepare_d4_0_b_config.py](diffusion/scripts/prepare_d4_0_b_config.py) | 准备历史掩码扩散对照配置。 |
| [prepare_d4_1_config.py](diffusion/scripts/prepare_d4_1_config.py) | 准备历史化学条件扩散配置。 |
| [prepare_d4_2_config.py](diffusion/scripts/prepare_d4_2_config.py) | 准备条件 PCA 先验残差扩散配置。 |
| [prepare_d4_3_configs.py](diffusion/scripts/prepare_d4_3_configs.py) | 准备条件掩码与多样性约束对照配置。 |
| [start_ddpm_training.py](diffusion/scripts/start_ddpm_training.py) | 启动配置指定的扩散训练或管线检查。 |
| [verify_d4_2_checkpoint.py](diffusion/scripts/verify_d4_2_checkpoint.py) | 检查条件先验 checkpoint 状态、分解和保存信息。 |
| [verify_d4_2_generated_names.py](diffusion/scripts/verify_d4_2_generated_names.py) | 检查条件生成文件命名、源文件对应和完整性。 |
| [verify_fixed_split.py](diffusion/scripts/verify_fixed_split.py) | 核对固定 12/4/4 划分和训练/验证/测试索引。 |
| [verify_mixed_raman_masks.py](diffusion/scripts/verify_mixed_raman_masks.py) | 核对长短 Raman 轴的 union 网格、有效区和归一化掩码。 |
| [verify_spectrum_conditions.py](diffusion/scripts/verify_spectrum_conditions.py) | 核对源文件条件解析、标签组合和条件覆盖。 |

</details>

<details>
<summary>diffusion/src：47 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [__init__.py](diffusion/src/__init__.py) | 声明 Python 模块包，支持模块导入。 |
| [broad_local_residual.py](diffusion/src/broad_local_residual.py) | 执行 broad/local 分解或完整残差处理，保存残差缩放与 broad PCA 状态。 |
| [checkpoint_manager.py](diffusion/src/checkpoint_manager.py) | 保存、读取 checkpoint 和 Raman 轴等恢复信息。 |
| [condition_grouped_batch_sampler.py](diffusion/src/condition_grouped_batch_sampler.py) | 把同条件的多条光谱编入一个 batch，支持成对多样性监督。 |
| [conditional_diversity_constraints.py](diffusion/src/conditional_diversity_constraints.py) | 计算条件与有效区感知的残差几何、多样性和质量辅助损失。 |
| [conditional_prior_residual.py](diffusion/src/conditional_prior_residual.py) | 管理各化学条件的训练先验、交叉拟合残差及生成采样状态。 |
| [conditional_spectrum_constraints.py](diffusion/src/conditional_spectrum_constraints.py) | 拟合训练集完整光谱支持范围，并执行重建和软包络约束。 |
| [configuration_loader.py](diffusion/src/configuration_loader.py) | 读取 YAML 配置并校验数据、网络、残差和生成设置。 |
| [dataset_splitter.py](diffusion/src/dataset_splitter.py) | 实现可复现数据划分及固定源文件内 12/4/4 划分。 |
| [ddpm_trainer.py](diffusion/src/ddpm_trainer.py) | 组织扩散训练、验证、EMA、学习率调度与 checkpoint 保存。 |
| [dit_1d.py](diffusion/src/dit_1d.py) | 实现一维 Linear 分块、全局注意力和 adaLN-Zero DiT 去噪网络。 |
| [dit_configuration.py](diffusion/src/dit_configuration.py) | 校验纯 DiT 的结构参数与残差模式兼容性。 |
| [dit_pipeline_check.py](diffusion/src/dit_pipeline_check.py) | 检查 DiT 零初始化后时间、条件与注意力路径的梯度传递。 |
| [feature_peak_residual_limiter.py](diffusion/src/feature_peak_residual_limiter.py) | 按训练支持对先验特征峰附近的残差进行软包络限制。 |
| [final_spectrum_delivery.py](diffusion/src/final_spectrum_delivery.py) | 在最终导出前检查强度支持和最多 5% 修改点预算。 |
| [generation_stage_trace.py](diffusion/src/generation_stage_trace.py) | 记录生成各阶段的只读快照与分解、多样性诊断。 |
| [hybrid_unet_configuration.py](diffusion/src/hybrid_unet_configuration.py) | 保留 D4.26 混合 U-Net 配置校验与兼容路径。 |
| [intensity_normalizer.py](diffusion/src/intensity_normalizer.py) | 拟合训练有效点的全局强度归一化，并支持逆变换。 |
| [joint_prior_sampling.py](diffusion/src/joint_prior_sampling.py) | 提供 PCA 边际与跨块 Gaussian copula 联合先验采样选项。 |
| [masked_diffusion.py](diffusion/src/masked_diffusion.py) | 实现带 Raman 有效区掩码的扩散损失、预测与反向采样。 |
| [masked_unet.py](diffusion/src/masked_unet.py) | 保留掩码条件 U-Net 兼容骨架与条件注入。 |
| [model_builder.py](diffusion/src/model_builder.py) | 按配置构建 DiT 或兼容 U-Net，并连接先验、损失和生成保护。 |
| [one_dimensional_ddpm.py](diffusion/src/one_dimensional_ddpm.py) | 封装外部一维 DDPM 后端及其版本检查。 |
| [package_version.py](diffusion/src/package_version.py) | 声明生成项目的包版本。 |
| [physics_guided_diffusion.py](diffusion/src/physics_guided_diffusion.py) | 保留第三方扩散后端与物理、分布、多样性约束的兼容连接。 |
| [prior_bottleneck_transformer.py](diffusion/src/prior_bottleneck_transformer.py) | 提供旧混合 U-Net 的先验条件瓶颈 Transformer。 |
| [prior_residual.py](diffusion/src/prior_residual.py) | 拟合训练 PCA 先验，执行残差变换、方差均衡和逆变换。 |
| [random_seed_manager.py](diffusion/src/random_seed_manager.py) | 统一随机种子与 DataLoader 随机状态。 |
| [residual_aware_loss.py](diffusion/src/residual_aware_loss.py) | 校验并构建残差感知的逐点主损失权重。 |
| [sers_distribution_constraints.py](diffusion/src/sers_distribution_constraints.py) | 拟合训练峰分布并提供批量分布与多样性约束。 |
| [sers_diversity_constraints.py](diffusion/src/sers_diversity_constraints.py) | 提供高频残差和峰形层面的多样性约束选项。 |
| [sers_generation_evaluator.py](diffusion/src/sers_generation_evaluator.py) | 计算完整光谱相似性、分布与多样性评价指标。 |
| [sers_local_peak_distribution_constraints.py](diffusion/src/sers_local_peak_distribution_constraints.py) | 提供局部稳定峰的翼部、位移、宽度与负尾约束。 |
| [sers_peak_derivative_constraints.py](diffusion/src/sers_peak_derivative_constraints.py) | 提供训练自动峰区的一阶导数软约束。 |
| [sers_peak_parameter_constraints.py](diffusion/src/sers_peak_parameter_constraints.py) | 提供完整光谱峰位、峰宽等参数的训练支持软约束。 |
| [sers_physics_constraints.py](diffusion/src/sers_physics_constraints.py) | 提供 SERS 物理范围与病态光谱保护选项。 |
| [sers_relative_peak_intensity_constraints.py](diffusion/src/sers_relative_peak_intensity_constraints.py) | 提供训练峰间相对强度的软约束。 |
| [sers_sampling_calibrator.py](diffusion/src/sers_sampling_calibrator.py) | 根据训练参考执行反向采样阶段的光谱校准。 |
| [sers_training_guard.py](diffusion/src/sers_training_guard.py) | 基于训练峰与负值支持检查生成谱质量。 |
| [spectrum_conditioning.py](diffusion/src/spectrum_conditioning.py) | 解析农药、等级、组合与基质，生成 14 维化学条件。 |
| [spectrum_dataset.py](diffusion/src/spectrum_dataset.py) | 返回扩散训练样本、条件向量、先验底谱与有效区掩码。 |
| [spectrum_exporter.py](diffusion/src/spectrum_exporter.py) | 按物理轴导出生成光谱和预览文件。 |
| [spectrum_file_reader.py](diffusion/src/spectrum_file_reader.py) | 读取不同格式和不同长度的 Raman 光谱表格。 |
| [spectrum_generator.py](diffusion/src/spectrum_generator.py) | 连接残差采样、先验重建、校准和完整光谱生成。 |
| [spectrum_length_adapter.py](diffusion/src/spectrum_length_adapter.py) | 按 Raman 物理轴插值、补齐、裁切并维护有效区。 |
| [stage_trace_replay.py](diffusion/src/stage_trace_replay.py) | 在 CPU 上重放保存的后处理阶段，核对生成追踪结果。 |
| [training_logger.py](diffusion/src/training_logger.py) | 记录训练、验证与各约束分项损失。 |

</details>

<details>
<summary>diffusion/tests：36 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [test_broad_local_residual.py](diffusion/tests/test_broad_local_residual.py) | 回归检查：执行 broad/local 分解或完整残差处理，保存残差缩放与 broad PCA 状态。 |
| [test_condition_aware_diversity.py](diffusion/tests/test_condition_aware_diversity.py) | 回归检查：同条件有效区的多样性与质量目标。 |
| [test_conditional_prior_residual.py](diffusion/tests/test_conditional_prior_residual.py) | 回归检查：管理各化学条件的训练先验、交叉拟合残差及生成采样状态。 |
| [test_d4_25_final_delivery.py](diffusion/tests/test_d4_25_final_delivery.py) | 回归检查：最终交付边界与修改点预算。 |
| [test_d4_25_joint_prior_sampling.py](diffusion/tests/test_d4_25_joint_prior_sampling.py) | 回归检查：联合先验采样的边际与支持保护。 |
| [test_d4_25_spectrum_support.py](diffusion/tests/test_d4_25_spectrum_support.py) | 回归检查：完整光谱支持约束。 |
| [test_d4_25_stage_trace.py](diffusion/tests/test_d4_25_stage_trace.py) | 回归检查：生成阶段快照与重放。 |
| [test_d4_26_hybrid_unet.py](diffusion/tests/test_d4_26_hybrid_unet.py) | 回归检查：旧混合 U-Net 兼容和先验条件连接。 |
| [test_d4_27_dit.py](diffusion/tests/test_d4_27_dit.py) | 回归检查：纯 DiT、掩码、残差模式与条件连接。 |
| [test_d4_3_2_15a_shape_preserving_guard.py](diffusion/tests/test_d4_3_2_15a_shape_preserving_guard.py) | 回归检查：负谷保护与峰形、多样性保持。 |
| [test_d4_3_2_16_variance_profile.py](diffusion/tests/test_d4_3_2_16_variance_profile.py) | 回归检查：训练 Raman 方差剖面与逐点监督。 |
| [test_d4_3_2_17_group_variance.py](diffusion/tests/test_d4_3_2_17_group_variance.py) | 回归检查：条件分组方差参考。 |
| [test_d4_3_masked_diffusion.py](diffusion/tests/test_d4_3_masked_diffusion.py) | 回归检查：条件、掩码、底谱与扩散过程连接。 |
| [test_dataset_splitter.py](diffusion/tests/test_dataset_splitter.py) | 回归检查：可复现数据划分。 |
| [test_ddpm_forward.py](diffusion/tests/test_ddpm_forward.py) | 回归检查：扩散前向与训练损失。 |
| [test_feature_peak_residual_limiter.py](diffusion/tests/test_feature_peak_residual_limiter.py) | 回归检查：按训练支持对先验特征峰附近的残差进行软包络限制。 |
| [test_final_qq_diversity_retention.py](diffusion/tests/test_final_qq_diversity_retention.py) | 回归检查：QQ 校准后的多样性保留。 |
| [test_mini_training.py](diffusion/tests/test_mini_training.py) | 回归检查：最小扩散训练管线。 |
| [test_mixed_raman_masks.py](diffusion/tests/test_mixed_raman_masks.py) | 回归检查：长短 Raman 轴与有效区掩码。 |
| [test_prior_residual.py](diffusion/tests/test_prior_residual.py) | 回归检查：拟合训练 PCA 先验，执行残差变换、方差均衡和逆变换。 |
| [test_residual_aware_loss.py](diffusion/tests/test_residual_aware_loss.py) | 回归检查：校验并构建残差感知的逐点主损失权重。 |
| [test_sers_distribution_constraints.py](diffusion/tests/test_sers_distribution_constraints.py) | 回归检查：拟合训练峰分布并提供批量分布与多样性约束。 |
| [test_sers_diversity_constraints.py](diffusion/tests/test_sers_diversity_constraints.py) | 回归检查：提供高频残差和峰形层面的多样性约束选项。 |
| [test_sers_generation_evaluator.py](diffusion/tests/test_sers_generation_evaluator.py) | 回归检查：计算完整光谱相似性、分布与多样性评价指标。 |
| [test_sers_local_peak_distribution_constraints.py](diffusion/tests/test_sers_local_peak_distribution_constraints.py) | 回归检查：提供局部稳定峰的翼部、位移、宽度与负尾约束。 |
| [test_sers_peak_derivative_constraints.py](diffusion/tests/test_sers_peak_derivative_constraints.py) | 回归检查：提供训练自动峰区的一阶导数软约束。 |
| [test_sers_peak_parameter_constraints.py](diffusion/tests/test_sers_peak_parameter_constraints.py) | 回归检查：提供完整光谱峰位、峰宽等参数的训练支持软约束。 |
| [test_sers_physics_constraints.py](diffusion/tests/test_sers_physics_constraints.py) | 回归检查：提供 SERS 物理范围与病态光谱保护选项。 |
| [test_sers_relative_peak_intensity_constraints.py](diffusion/tests/test_sers_relative_peak_intensity_constraints.py) | 回归检查：提供训练峰间相对强度的软约束。 |
| [test_sers_training_guard.py](diffusion/tests/test_sers_training_guard.py) | 回归检查：基于训练峰与负值支持检查生成谱质量。 |
| [test_source_named_condition_generation.py](diffusion/tests/test_source_named_condition_generation.py) | 回归检查：生成文件源名称与条件名称对应。 |
| [test_spectrum_conditioning.py](diffusion/tests/test_spectrum_conditioning.py) | 回归检查：14 维化学条件编码。 |
| [test_spectrum_dataset.py](diffusion/tests/test_spectrum_dataset.py) | 回归检查：扩散数据集样本与形状。 |
| [test_spectrum_file_reader.py](diffusion/tests/test_spectrum_file_reader.py) | 回归检查：光谱表格读取。 |
| [test_spectrum_generation.py](diffusion/tests/test_spectrum_generation.py) | 回归检查：完整光谱生成与后处理保护。 |
| [test_warmup_cosine_scheduler.py](diffusion/tests/test_warmup_cosine_scheduler.py) | 回归检查：预热、余弦学习率与调度恢复。 |

</details>

<details>
<summary>packages：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [README.md](packages/README.md) | 说明本目录作用、关键入口与完整文件功能索引。 |

</details>

<details>
<summary>packages/d4_27_dit_ablation：12 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [DiT_LICENSE.txt](packages/d4_27_dit_ablation/DiT_LICENSE.txt) | 保存参考 DiT 实现的许可条款。 |
| [README_CN.md](packages/d4_27_dit_ablation/README_CN.md) | 提供 D4.27 配对实验、安装、运行和恢复操作说明。 |
| [THIRD_PARTY_NOTICES.md](packages/d4_27_dit_ablation/THIRD_PARTY_NOTICES.md) | 记录 DiT 适配的第三方来源和许可说明。 |
| [apply_patch.py](packages/d4_27_dit_ablation/apply_patch.py) | 按文件哈希安装 D4.27 更新，支持预检、备份和重复安装检查。 |
| [file_hashes.json](packages/d4_27_dit_ablation/file_hashes.json) | 记录 D4.27 安装器所需的更新前后文件哈希。 |
| [prepare_run.py](packages/d4_27_dit_ablation/prepare_run.py) | 从正式基础 runtime 配置准备两种残差的配对实验。 |
| [reference_D4_26_runtime.yaml](packages/d4_27_dit_ablation/reference_D4_26_runtime.yaml) | 提供 D4.27 准备和合成检查所需的基础 runtime 配置。 |
| [run_experiment.py](packages/d4_27_dit_ablation/run_experiment.py) | 调度两组检查、训练、生成和评价阶段。 |
| [run_parallel.py](packages/d4_27_dit_ablation/run_parallel.py) | 为两种残差分配独立 GPU，并在训练完成后同步进入生成。 |
| [smoke_test.py](packages/d4_27_dit_ablation/smoke_test.py) | 使用合成光谱验证完整训练和生成管线。 |
| [test_package.py](packages/d4_27_dit_ablation/test_package.py) | 检查安装预检、备份、幂等和源码漂移拒绝行为。 |
| [test_parallel.py](packages/d4_27_dit_ablation/test_parallel.py) | 检查双组进程、GPU 映射、阶段屏障及失败处理。 |

</details>

<details>
<summary>packages/d4_27_dit_ablation/evidence：2 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [formal_model.json](packages/d4_27_dit_ablation/evidence/formal_model.json) | 记录正式 DiT 结构与参数核验结果。 |
| [verification_summary.json](packages/d4_27_dit_ablation/evidence/verification_summary.json) | 记录 D4.27 补丁包的本地验证摘要。 |

</details>

<details>
<summary>packages/d4_27_dit_ablation/payload/scripts：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [start_ddpm_training.py](packages/d4_27_dit_ablation/payload/scripts/start_ddpm_training.py) | 安装载荷：启动配置指定的扩散训练或管线检查。 |

</details>

<details>
<summary>packages/d4_27_dit_ablation/payload/src：7 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [broad_local_residual.py](packages/d4_27_dit_ablation/payload/src/broad_local_residual.py) | 安装载荷：执行 broad/local 分解或完整残差处理，保存残差缩放与 broad PCA 状态。 |
| [conditional_prior_residual.py](packages/d4_27_dit_ablation/payload/src/conditional_prior_residual.py) | 安装载荷：管理各化学条件的训练先验、交叉拟合残差及生成采样状态。 |
| [configuration_loader.py](packages/d4_27_dit_ablation/payload/src/configuration_loader.py) | 安装载荷：读取 YAML 配置并校验数据、网络、残差和生成设置。 |
| [dit_1d.py](packages/d4_27_dit_ablation/payload/src/dit_1d.py) | 安装载荷：实现一维 Linear 分块、全局注意力和 adaLN-Zero DiT 去噪网络。 |
| [dit_configuration.py](packages/d4_27_dit_ablation/payload/src/dit_configuration.py) | 安装载荷：校验纯 DiT 的结构参数与残差模式兼容性。 |
| [dit_pipeline_check.py](packages/d4_27_dit_ablation/payload/src/dit_pipeline_check.py) | 安装载荷：检查 DiT 零初始化后时间、条件与注意力路径的梯度传递。 |
| [model_builder.py](packages/d4_27_dit_ablation/payload/src/model_builder.py) | 安装载荷：按配置构建 DiT 或兼容 U-Net，并连接先验、损失和生成保护。 |

</details>

<details>
<summary>packages/d4_27_dit_ablation/payload/tests：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [test_d4_27_dit.py](packages/d4_27_dit_ablation/payload/tests/test_d4_27_dit.py) | 回归检查：纯 DiT、掩码、残差模式与条件连接。 |

</details>

<details>
<summary>packages/d4_27_dit_ablation/test_fixtures/originals/scripts：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [start_ddpm_training.py](packages/d4_27_dit_ablation/test_fixtures/originals/scripts/start_ddpm_training.py) | 更新前夹具：启动配置指定的扩散训练或管线检查。 |

</details>

<details>
<summary>packages/d4_27_dit_ablation/test_fixtures/originals/src：4 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [broad_local_residual.py](packages/d4_27_dit_ablation/test_fixtures/originals/src/broad_local_residual.py) | 更新前夹具：执行 broad/local 分解或完整残差处理，保存残差缩放与 broad PCA 状态。 |
| [conditional_prior_residual.py](packages/d4_27_dit_ablation/test_fixtures/originals/src/conditional_prior_residual.py) | 更新前夹具：管理各化学条件的训练先验、交叉拟合残差及生成采样状态。 |
| [configuration_loader.py](packages/d4_27_dit_ablation/test_fixtures/originals/src/configuration_loader.py) | 更新前夹具：读取 YAML 配置并校验数据、网络、残差和生成设置。 |
| [model_builder.py](packages/d4_27_dit_ablation/test_fixtures/originals/src/model_builder.py) | 更新前夹具：按配置构建 DiT 或兼容 U-Net，并连接先验、损失和生成保护。 |

</details>

<details>
<summary>packages/sers_t330_feature_finetune：7 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [LOCAL_VALIDATION.json](packages/sers_t330_feature_finetune/LOCAL_VALIDATION.json) | 记录实验包发布前的本地验证结果和范围。 |
| [MANIFEST.json](packages/sers_t330_feature_finetune/MANIFEST.json) | 列出实验包强制校验的文件哈希；加载实验入口时使用。 |
| [README_操作步骤.md](packages/sers_t330_feature_finetune/README_操作步骤.md) | 提供对应识别实验的复核、微调、评价与恢复步骤。 |
| [feature_model.py](packages/sers_t330_feature_finetune/feature_model.py) | 实现 T3.30 冻结基线、浓度微调副本及最终部署解码。 |
| [local_tests.log](packages/sers_t330_feature_finetune/local_tests.log) | 保存实验包原始校验日志；该日志也是 MANIFEST 的必需资源。 |
| [run_feature.py](packages/sers_t330_feature_finetune/run_feature.py) | 运行 T3.30 复核、微调、候选比较和部署模式选择。 |
| [test_feature.py](packages/sers_t330_feature_finetune/test_feature.py) | 检查 T3.30 模型梯度、冻结约束、路由及混合解码。 |

</details>

<details>
<summary>packages/sers_t330_feature_finetune/reference：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [combined_validation.csv](packages/sers_t330_feature_finetune/reference/combined_validation.csv) | 保存冻结组合基线的验证预测参考，用于逐条复核。 |

</details>

<details>
<summary>packages/sers_t330_feature_finetune/vendor/base：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [combined_model.py](packages/sers_t330_feature_finetune/vendor/base/combined_model.py) | 封装冻结 DEL-only 基线，保留三元 DEL 联合 MAP 解码。 |

</details>

<details>
<summary>packages/sers_t330_feature_finetune/vendor/base/vendor/t328：6 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [base_bridge.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/base_bridge.py) | 加载已接受 Adapter 基线并连接当前联合模型。 |
| [pack_reports.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/pack_reports.py) | 整理三元实验报告和评价输出。 |
| [sam_update.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/sam_update.py) | 提供历史三元实验 SAM 参数更新对照工具。 |
| [selection_policy.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/selection_policy.py) | 定义候选采用与分组保护规则，防止已有表现退步。 |
| [ternary_model.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/ternary_model.py) | 实现三元 27 状态联合评分与按预测存在概率的路由。 |
| [train_ternary.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/train_ternary.py) | 提供三元联合头实验的训练、检查和 checkpoint 加载工具。 |

</details>

<details>
<summary>packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327：3 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [peft.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/peft.py) | 实现 Adapter、LoRA 及分组学习率配置；当前继承 Adapter。 |
| [run_optimization.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/run_optimization.py) | 提供 Adapter/LoRA/优化器对照及基线恢复工具。 |
| [schedule.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/schedule.py) | 实现按更新步数的预热与余弦学习率调度。 |

</details>

<details>
<summary>packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived：9 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [Model_T3_17.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived/Model_T3_17.py) | 保留注意力扩维的历史识别骨架兼容依赖。 |
| [Model_T3_24.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived/Model_T3_24.py) | 保留浓度更新对照的历史骨架与校验依赖。 |
| [Model_T3_25.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived/Model_T3_25.py) | 保留多峰证据对照的历史识别骨架依赖。 |
| [Model_T3_26.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived/Model_T3_26.py) | 完整识别主干：128 维注意力、六层 Encoder 与多峰证据。 |
| [MultiPeak_T3_25.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived/MultiPeak_T3_25.py) | 实现实测峰窗的曲线、统计量和 Query 交叉注意力证据。 |
| [Relations_T3_26.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived/Relations_T3_26.py) | 保留跨组合浓度关系监督工具；当前续训不启用该损失。 |
| [data_audit.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived/data_audit.py) | 检查固定数据选择、样本来源与实测峰窗掩码。 |
| [expected_project_sources.json](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived/expected_project_sources.json) | 记录识别项目关键源码的预期哈希，防止加载错误版本。 |
| [runtime_t326.py](packages/sers_t330_feature_finetune/vendor/base/vendor/t328/vendor/t327/archived/runtime_t326.py) | 提供受源码哈希保护的数据加载、完整前向与训练评价工具。 |

</details>

<details>
<summary>packages/sers_t331_target_aligned：7 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [LOCAL_VALIDATION.json](packages/sers_t331_target_aligned/LOCAL_VALIDATION.json) | 记录实验包发布前的本地验证结果和范围。 |
| [MANIFEST.json](packages/sers_t331_target_aligned/MANIFEST.json) | 列出实验包强制校验的文件哈希；加载实验入口时使用。 |
| [README_操作步骤.md](packages/sers_t331_target_aligned/README_操作步骤.md) | 提供对应识别实验的复核、微调、评价与恢复步骤。 |
| [aligned_model.py](packages/sers_t331_target_aligned/aligned_model.py) | 实现 T3.31 目标农药对齐的特征或等级头续训对照。 |
| [local_tests.log](packages/sers_t331_target_aligned/local_tests.log) | 保存实验包原始校验日志；该日志也是 MANIFEST 的必需资源。 |
| [run_aligned.py](packages/sers_t331_target_aligned/run_aligned.py) | 运行 T3.31 目标对齐检查、缓存、训练和候选比较。 |
| [test_aligned.py](packages/sers_t331_target_aligned/test_aligned.py) | 检查 T3.31 目标损失、缓存一致性与未更新农药保护。 |

</details>

<details>
<summary>packages/sers_t331_target_aligned/reference：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [baseline_validation.csv](packages/sers_t331_target_aligned/reference/baseline_validation.csv) | 保存 T3.31 父模型验证预测参考，用于复核父模型恢复。 |

</details>

<details>
<summary>packages/sers_t331_target_aligned/vendor/t330：7 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [LOCAL_VALIDATION.json](packages/sers_t331_target_aligned/vendor/t330/LOCAL_VALIDATION.json) | 记录实验包发布前的本地验证结果和范围。 |
| [MANIFEST.json](packages/sers_t331_target_aligned/vendor/t330/MANIFEST.json) | 列出实验包强制校验的文件哈希；加载实验入口时使用。 |
| [README_操作步骤.md](packages/sers_t331_target_aligned/vendor/t330/README_操作步骤.md) | 提供对应识别实验的复核、微调、评价与恢复步骤。 |
| [feature_model.py](packages/sers_t331_target_aligned/vendor/t330/feature_model.py) | 实现 T3.30 冻结基线、浓度微调副本及最终部署解码。 |
| [local_tests.log](packages/sers_t331_target_aligned/vendor/t330/local_tests.log) | 保存实验包原始校验日志；该日志也是 MANIFEST 的必需资源。 |
| [run_feature.py](packages/sers_t331_target_aligned/vendor/t330/run_feature.py) | 运行 T3.30 复核、微调、候选比较和部署模式选择。 |
| [test_feature.py](packages/sers_t331_target_aligned/vendor/t330/test_feature.py) | 检查 T3.30 模型梯度、冻结约束、路由及混合解码。 |

</details>

<details>
<summary>packages/sers_t331_target_aligned/vendor/t330/reference：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [combined_validation.csv](packages/sers_t331_target_aligned/vendor/t330/reference/combined_validation.csv) | 保存冻结组合基线的验证预测参考，用于逐条复核。 |

</details>

<details>
<summary>packages/sers_t331_target_aligned/vendor/t330/vendor/base：1 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [combined_model.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/combined_model.py) | 封装冻结 DEL-only 基线，保留三元 DEL 联合 MAP 解码。 |

</details>

<details>
<summary>packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328：6 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [base_bridge.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/base_bridge.py) | 加载已接受 Adapter 基线并连接当前联合模型。 |
| [pack_reports.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/pack_reports.py) | 整理三元实验报告和评价输出。 |
| [sam_update.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/sam_update.py) | 提供历史三元实验 SAM 参数更新对照工具。 |
| [selection_policy.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/selection_policy.py) | 定义候选采用与分组保护规则，防止已有表现退步。 |
| [ternary_model.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/ternary_model.py) | 实现三元 27 状态联合评分与按预测存在概率的路由。 |
| [train_ternary.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/train_ternary.py) | 提供三元联合头实验的训练、检查和 checkpoint 加载工具。 |

</details>

<details>
<summary>packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327：3 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [peft.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/peft.py) | 实现 Adapter、LoRA 及分组学习率配置；当前继承 Adapter。 |
| [run_optimization.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/run_optimization.py) | 提供 Adapter/LoRA/优化器对照及基线恢复工具。 |
| [schedule.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/schedule.py) | 实现按更新步数的预热与余弦学习率调度。 |

</details>

<details>
<summary>packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived：9 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [Model_T3_17.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived/Model_T3_17.py) | 保留注意力扩维的历史识别骨架兼容依赖。 |
| [Model_T3_24.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived/Model_T3_24.py) | 保留浓度更新对照的历史骨架与校验依赖。 |
| [Model_T3_25.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived/Model_T3_25.py) | 保留多峰证据对照的历史识别骨架依赖。 |
| [Model_T3_26.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived/Model_T3_26.py) | 完整识别主干：128 维注意力、六层 Encoder 与多峰证据。 |
| [MultiPeak_T3_25.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived/MultiPeak_T3_25.py) | 实现实测峰窗的曲线、统计量和 Query 交叉注意力证据。 |
| [Relations_T3_26.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived/Relations_T3_26.py) | 保留跨组合浓度关系监督工具；当前续训不启用该损失。 |
| [data_audit.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived/data_audit.py) | 检查固定数据选择、样本来源与实测峰窗掩码。 |
| [expected_project_sources.json](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived/expected_project_sources.json) | 记录识别项目关键源码的预期哈希，防止加载错误版本。 |
| [runtime_t326.py](packages/sers_t331_target_aligned/vendor/t330/vendor/base/vendor/t328/vendor/t327/archived/runtime_t326.py) | 提供受源码哈希保护的数据加载、完整前向与训练评价工具。 |

</details>

<details>
<summary>transformer：13 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [Dataset.py](transformer/Dataset.py) | 读取光谱、固定划分、补齐短轴并构建识别输入与峰先验。 |
| [EvalutionPlots.py](transformer/EvalutionPlots.py) | 绘制混淆矩阵、等级预测与训练历史图。 |
| [Inference.py](transformer/Inference.py) | 加载基础模型并输出整体、农药、组合与基质分层评价。 |
| [LICENSE](transformer/LICENSE) | 保存基础识别实现的许可条款。 |
| [Metric.py](transformer/Metric.py) | 计算多标签类别及等级码回归评价指标。 |
| [Model_v2.py](transformer/Model_v2.py) | 基础三分支 CNN、峰引导 Query 和有序等级 SERSFormer。 |
| [README.md](transformer/README.md) | 说明本目录作用、关键入口与完整文件功能索引。 |
| [SERSFormer_Training.py](transformer/SERSFormer_Training.py) | 训练基础多任务 SERSFormer 并保存模型和验证结果。 |
| [T3_14_Calibration.py](transformer/T3_14_Calibration.py) | 提供冻结基线的有界局部有序等级校准和相关损失。 |
| [Train_T3_14.py](transformer/Train_T3_14.py) | 运行历史冻结基线锚定校准对照。 |
| [Train_T3_15.py](transformer/Train_T3_15.py) | 运行历史仅浓度融合及有序头的微调对照。 |
| [required_packges.txt](transformer/required_packges.txt) | 列出基础识别项目的安装依赖。 |
| [setup.sh](transformer/setup.sh) | 安装基础识别项目依赖的历史环境入口。 |

</details>

<details>
<summary>transformer/tests：18 个文件</summary>

| 文件 | 简要功能 |
|---|---|
| [test_adaptive_mixture_gate.py](transformer/tests/test_adaptive_mixture_gate.py) | 回归检查：自适应混合门控可选路径。 |
| [test_attention_specialization_diagnostics.py](transformer/tests/test_attention_specialization_diagnostics.py) | 回归检查：农药 Query 注意力的差异诊断。 |
| [test_boundary_aware_ordinal_loss.py](transformer/tests/test_boundary_aware_ordinal_loss.py) | 回归检查：浓度边界有序等级损失。 |
| [test_boundary_specific_mixture_gate.py](transformer/tests/test_boundary_specific_mixture_gate.py) | 回归检查：S/M、M/H 独立混合门控可选路径。 |
| [test_common_axis_and_evaluation.py](transformer/tests/test_common_axis_and_evaluation.py) | 回归检查：公共 Raman 轴与基础评价管线。 |
| [test_corn_ordinal_loss.py](transformer/tests/test_corn_ordinal_loss.py) | 回归检查：CORN 条件等级损失和单调概率。 |
| [test_full_axis_chemistry_prior.py](transformer/tests/test_full_axis_chemistry_prior.py) | 回归检查：完整 Raman 轴上的化学峰先验。 |
| [test_matched_condition_peak_prior.py](transformer/tests/test_matched_condition_peak_prior.py) | 回归检查：训练匹配条件的峰先验。 |
| [test_mixture_aware_query_fusion.py](transformer/tests/test_mixture_aware_query_fusion.py) | 回归检查：跨农药 Query 浓度融合。 |
| [test_ordinal_concentration_head.py](transformer/tests/test_ordinal_concentration_head.py) | 回归检查：有序浓度头与等级解码。 |
| [test_ordinal_head_capacity.py](transformer/tests/test_ordinal_head_capacity.py) | 回归检查：有序头隐藏层容量与结构兼容。 |
| [test_peak_guided_sersformer.py](transformer/tests/test_peak_guided_sersformer.py) | 回归检查：峰引导 Query 与基础多任务模型。 |
| [test_query_only_attention.py](transformer/tests/test_query_only_attention.py) | 回归检查：不使用峰引导的 Query 注意力对照。 |
| [test_runtime_tail_to_2500.py](transformer/tests/test_runtime_tail_to_2500.py) | 回归检查：短轴谱内存补齐及原文件不变。 |
| [test_t313_local_quantitative.py](transformer/tests/test_t313_local_quantitative.py) | 回归检查：旧局部定量分支的统计和梯度。 |
| [test_t314_anchored_calibration.py](transformer/tests/test_t314_anchored_calibration.py) | 回归检查：旧冻结基线锚定校准与保护。 |
| [test_t315_quantitative_finetuning.py](transformer/tests/test_t315_quantitative_finetuning.py) | 回归检查：旧浓度专用微调和主干冻结。 |
| [test_ternary_boundary_weighting.py](transformer/tests/test_ternary_boundary_weighting.py) | 回归检查：三元样本等级边界监督权重。 |

</details>

