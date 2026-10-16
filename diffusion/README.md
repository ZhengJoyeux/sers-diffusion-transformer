# D4.28 固定坐标曲线图 DiT

当前扩散端已集成纯 Transformer DiT：完整光谱在训练集固定坐标下栅格化为二维浮点曲线图，进行二维像素扩散，再恢复为数值光谱。

- 正式实验图像 1920×768；全高度 patch 为 768×16，共 120 tokens。
- 当前实验采用 full_residual，保留 PCA、14维条件、valid_mask、原数值约束和后处理。
- 50 epoch、batch size 16；生成每条件 50 条；当前入口默认物理 GPU 0。
- 生成先验越界采用整条同条件重抽，记录审计并保留固定纵轴及原最终交付规则。
- 真实实验的局部峰形与 QQ 尾部仍有问题，不能声称优于一维 DiT 或已改善浓度预测。

完整模型说明、准备、自检、训练、生成和评价指令见 [仓库 README](../README.md)。仓库源码已集成，无需再次安装旧补丁。

主模型文件：`src/dit_curve2d.py`、`src/curve_image_codec.py`、`src/curve_image_diffusion.py`。运行入口位于相邻 `packages/D4_28_CurveImage_DiT` 和 `packages/D4_28_PriorBounds_Fix`。

旧无条件 README 保存在 [历史说明](docs/archive/unconditional_README.md)，其中基线范围不代表当前模型功能。
