# 当前二维识别及优化模块

| 目录 | 简要功能 | 使用状态 |
|---|---|---|
| [T3_32_Image_Recognition](T3_32_Image_Recognition) | 二维折线图编码、CNN＋Transformer、多峰数值辅助 | 基础模型 |
| [T3_33_Ternary_Correction](T3_33_Ternary_Correction) | 冻结二维主干，预测三元时仅修正DEL/TEB | 当前保留模型，第4轮 |
| [T3_34_Frozen_Diagnosis](T3_34_Frozen_Diagnosis) | 真实/生成训练与验证错误及修正幅度审计 | 只读诊断 |
| [T3_35_Ternary_Head_Finetune](T3_35_Ternary_Head_Finetune) | 仅微调两个复制的等级头 | 未提升对照 |
| [T3_36_Ternary_Fusion_Finetune](T3_36_Ternary_Fusion_Finetune) | 复制融合/投影/等级头，含CUDA初始化修复 | 未提升对照 |

从[统一入口](../run_recognition.py)解析仓库路径；从[加载接口](../current_recognition.py)恢复当前模型。T3.36修复已整合，无需再次叠加补丁。仓库不包含数据、权重和缓存。
