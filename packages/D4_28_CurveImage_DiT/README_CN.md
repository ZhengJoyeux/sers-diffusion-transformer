# D4.28 仓库运行入口

二维模型已整合至 `../../diffusion/`，不需要 apply_patch 或复制 payload。完整说明与运行命令见 [仓库 README](../../README.md)。

- `prepare_run.py`：继承指定真实一维 DiT 配置，准备独立二维实验，默认每条件生成 50 条。
- `run_experiment.py`：check / train / generate / evaluate / pack，默认物理 GPU 0。
- generate / evaluate 调用相邻 `../D4_28_PriorBounds_Fix/run_generation.py`，保持已验证的整条先验重抽策略。
- `last_suite.txt` 由 prepare 自动写入，属于本机运行状态，不提交仓库。

已有服务器实验可显式使用 `--suite`，但项目源码与 runtime_config 必须通过 run_index 原有哈希校验。不要为复用旧实验而修改哈希记录。
