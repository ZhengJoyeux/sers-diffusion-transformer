# T3.34 三元识别只读诊断

直接读取 T3.33 已保存的 train/validation 特征缓存和 best_accepted.pt；不运行 CNN/Transformer，不训练，不修改权重、数据、原报告，也不打开测试缓存。CPU 即可。

## 运行

```bash
conda activate sers_ddpm
python -u /home/wqzheng/T3_34_Frozen_Diagnosis/run_diagnosis.py
```

默认依赖 `/home/wqzheng/T3_33_Ternary_Correction`；运行目录为 `/home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026`。必须保留该目录下 `feature_cache/<fingerprint>/train.pt` 和 `validation.pt`。脚本不会自动重建缓存。

输出 `/home/wqzheng/project_transformer/outputs/t3_34_frozen_diagnosis_seed2026_reports.zip`。把此 ZIP 发回即可，无需上传模型和特征缓存。

已存在同名诊断目录时会停止，防止覆盖。重复运行可指定新的 `--output-directory /home/wqzheng/project_transformer/outputs/t3_34_frozen_diagnosis_seed2026_repeat`。

## 报告

- group_metrics.csv：真实训练、生成训练、验证三部分的整体/单药/双药/三药及基质表现；基线与修正使用同一统计口径。
- ternary_conditions.csv：逐条件三元表现，便于判断错误是否集中在少数组合。
- ternary_confusions.csv：分来源、农药、真实/预测等级的混淆计数。
- correction_bound_audit.csv：DEL/TEB 错误中，哪些即使允许逐样本独立选择 ±bound 内的最佳残差也无法修正；记录实际残差接近边界的样本数。
- paired_predictions.csv：基线/修正预测、阈值概率、残差以及界内可达性。
- ternary_errors.csv：剩余三元错误及另外两种农药的真实等级，供分析混峰干扰。
- diagnosis_summary.json：缓存/权重哈希、验证复现、零训练步数、零测试访问记录。

bound 分析沿用原模型的 CORN 概率公式及概率偏移保护。±bound 是理论闭区间，tanh 实际只能趋近边界，因此这里只给出乐观可达上限；不是承诺某个共享修正头能达到该成绩。CHL 和未路由样本仍固定。三元完整预测受 CHL 错误影响，不应把全部失败都归因于 DEL/TEB。

## 判断原则

1. 真实训练明显好于验证：优先检查泛化、生成样本权重和数据批次差异。
2. 两者都不理想且大量错误超出修正幅度：优先考虑输出修正范围的限制，不能直接归因于冻结特征。
3. 大量错误在幅度范围内仍未修正：需继续区分特征可分性、损失目标、共享头容量和优化问题。
4. 生成训练明显好于真实训练：提示来源差异，但不足以证明生成样本损害了识别。

真实训练成绩属于训练后回看，不能作为泛化成绩。这一步不会直接证明冻结特征丢失信息，也不会据此自动启动微调。先诊断，再选择一次有证据支持的最小改动。
