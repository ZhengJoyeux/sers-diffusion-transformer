# T3.23补丁文件与范围

独立实验包，不覆盖任何原项目文件或已有权重。继续从T3.18冻结主干与零修正支路开始，不把T3.22候选作为训练初始化。

| 文件 | 变化 |
|---|---|
| Model_T3_17.py | 原版主干，源码保持逐字一致 |
| Model_T3_23.py | 沿用T3.22联合修正结构与1266维缓存格式，两组均no_reference |
| Loss_T3_23.py | 稳定累计log odds、混合M双边界分离、训练正确边界保护；plain保持旧目标 |
| engine_t323.py | 组件loss日志、各药物正确数、DEL-M/TEB分层、严格推荐检查、错误迁移报告 |
| T3_23_Training.py | protected/plain匹配实验，历史T3.22比较与连续Markdown表格 |
| Inference_T3_23.py | 独立恢复checkpoint与固定缓冲区，真实验证推理，不拟合 |
| data_t323.py / Paths_T3_23.py | 原划分、随机48身份和原数据路径，保持补尾种子行为 |
| prepare_t323.py / experiment_t323.py / run_t323.sh | 隔离准备、旧T3.22权重指纹保护、完整命令入口 |
| previous_validation_reference.json | 从上传报告逐谱重算的T3.22历史验证指标，只用于核对与比较 |
| test_t323.py | 损失数值/梯度、正确/错误掩码、历史退步拒绝、两组短训练与独立加载等验证 |
| expected_* / reference_random48.json | 已诊断源码、原模型与数据设置 |
| VALIDATION.json / package_manifest.json | 本地检查记录和包完整性 |

本轮没有改变生成数量、来源权重、主干结构、补尾策略、数据划分、分类头或物理浓度定义。浓度仍是S/M/H等级。保护条件只用于训练损失与验证选择，推理没有真实标签门控。

推荐规则更严格并非性能提升的证据。protected需要与plain及T3.22历史结果比较，并检查新错误明细；候选未通过时零修正fallback明确标记。验证集已用于多轮探索，最终结论需要独立测试。
