# 固定条件均值残差扩散

当前入口为 `python diffusion/run_generation.py`。生成端使用固定条件均值、二维像素 DiT 和 ridge_0 采样输出。D4.31 历史入口为 `python diffusion/run_generation_d431.py`。

每个条件用 12 条真实训练谱拟合均值和残差尺度，将 `z = asinh((I - mean) / scale)` 渲染为二维残差图。DiT 在 1920×768 像素空间去噪，采用 120 个 768×16 条带 patch。采样加载 EMA 权重，在每次模型前向输出上把每列渲染为单中心高斯曲线。解码后以 `I = mean + scale * sinh(z)` 还原强度。有效区间由实测掩码确定。

## 生成与后处理

依赖见 [requirements.txt](fixed_mean_residual/requirements.txt)。使用已有、可信的 checkpoint 和与其配套的 prepared 目录。代码、权重和准备数据的指纹都必须匹配。权重与数据需从实验存储取得。

```bash
PREPARED=/absolute/path/to/fixed_mean_residual_data_v1
CHECKPOINT=/absolute/path/to/best.pt
BATCH=/absolute/path/to/new_batch
CONDITION=DEL-H_water

python diffusion/run_generation.py generate \
  --prepared "$PREPARED" --checkpoint "$CHECKPOINT" --condition "$CONDITION" \
  --number 50 --batch-size 1 --seed 2026 --device cuda \
  --output "$BATCH/generated/$CONDITION"

python diffusion/run_generation.py postprocess \
  --prepared "$PREPARED" --generated "$BATCH/generated/$CONDITION" \
  --output "$BATCH/pca/$CONDITION"
```

生成阶段保留原始 QC 和 PNG 往返检查。后处理对解码残差做三点中值滤波，再加入条件内训练谱拟合的至多四个主成分的联合扰动，强度为 0.5，批内扰动均值为零。连线 PNG 与内部 NPZ 一起保存。该步骤输出增强候选。

## 评价

使用上述目录结构收集全部 126 组，每组 50 条。若不同批次包含互补条件，可重复传入 `--batch`。组数不完整或条件重复时，评价入口会报错。

```bash
python diffusion/run_generation.py evaluate \
  --prepared "$PREPARED" --batch "$BATCH" \
  --output /absolute/path/to/new_evaluation
```

逐条件比较原版、中值滤波、中值滤波＋PCA。每组参考 4 条真实验证谱。指标包括归一化 MSE、Pearson、COS、MMD²、JSD、切片 Wasserstein、多样性、峰位与漏峰、背景和 PNG 连续性。中值滤波单独版本由数值复算，当前后处理流程只保存原版与 PCA 候选的 PNG。

126 组中位数显示，中值滤波之后加入 PCA 使多样性比从 0.128 提升至 0.314，MMD² 从 0.1793 降至 0.1576，漏峰率从 10.81% 升至 12.56%。这组比较用于衡量 PCA 后处理的增益和代价。当前结果是同源条件内验证评价，测试集封存。

## 本地补齐实验清单

把实验服务器原始的 provenance.json 放到 diffusion/fixed_mean_residual/。从实验存储取得原始字节，保持该文件与 PACKAGE_SHA256.json 的记录一致。
