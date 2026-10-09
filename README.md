# D26+T24

当前服务器模型源码快照：D4.26 扩散生成 + T3.24 浓度优化。

- diffusion/：/home/wqzheng/project 当前源码。
- transformer/：/home/wqzheng/project_transformer 当前源码。
- packages/：D4.26、T3.17、T3.22、T3.23、T3.24 独立运行包，包含 T3.24 初始化复现修复。
- current_configs/：当前实验配置及摘要。
- CURRENT_MODEL.json：当前 checkpoint 路径、SHA256 与推荐选择。
- SOURCE_SHA256.json：本次上传源码校验清单。

T3.24 推荐为 residual_control 第 2 轮；验证集组分等级正确 978/1152，三元全部正确 119/216。测试集尚未评价。

本分支保存源码、配置和结果摘要。数据及 checkpoint 保留在服务器，未上传。
源码保留原服务器路径；重新部署需恢复数据、权重与历史实验，并将 packages 中的包放回 /home/wqzheng 下原同名目录。
