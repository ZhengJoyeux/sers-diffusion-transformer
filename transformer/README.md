# 识别基础：SERSFormer

本目录保存 DEL/CHL/TEB 的基础数据适配、三分支 CNN + Encoder-Only Transformer、峰引导 Query、有序等级训练及分层评价。

| 文件 | 简要功能 |
|---|---|
| `Dataset.py` | 光谱读取、固定划分、短轴补齐、特征与峰先验 |
| `Model_v2.py` | 基础峰引导多任务识别模型 |
| `SERSFormer_Training.py` | 基础训练及验证入口 |
| `Inference.py` / `Metric.py` | 基础推理、整体与分层指标 |
| `EvalutionPlots.py` | 评价与训练可视化 |
| `T3_14_Calibration.py` / `Train_T3_14.py` / `Train_T3_15.py` | 历史校准与浓度微调对照 |
| `tests/` | 基础识别、数据保护和可选模块回归测试 |

**当前完整推荐模型不只由 `Model_v2.py` 构成。** 使用 [T3.30 完整扩展](../packages/sers_t330_feature_finetune) 中 `feature_model.load_checkpoint` 恢复 Adapter、多峰、联合头、冻结基线与微调副本，再调用模型 `predict()`。

[逐文件功能索引](../FILE_GUIDE.md) · [模型组成、流程图与结果](../README.md)

## 上游来源与许可

基础实现来源于 [SERSFormer2](https://github.com/jianlin-cheng/SERSFormer2)，本项目按 DEL/CHL/TEB、水/土壤数据和等级任务进行了适配。原 README 给出的引用为：Hegde, A., Hajikhani, M., Snyder, J., Cheng, J., & Lin, M. (2024). *Leveraging SERS and transformer models for simultaneous detection of multiple pesticides in fresh produce.* 许可保留于 [LICENSE](LICENSE)。
