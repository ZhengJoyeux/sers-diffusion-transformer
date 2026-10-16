# D4.27 一维配置准备工具

本目录保留 `prepare_run.py`，用于从已有真实 D4.26 runtime_config 准备一维 DiT 的 broad_local / full_residual 两组配置，默认 50 epoch、batch size 16、seed 2026。源码已经整合至 `../../diffusion`，无需额外补丁。

D4.28 使用输出的 full_residual 配置作为继承基础，不需要先运行一维训练。该历史一维工具保留每条件 100 条的配置；D4.28 准备工具将新二维实验的生成数量设为 50。

`last_suite.txt` 由本机 prepare 生成，不提交仓库。完整操作见 [仓库 README](../../README.md)。
