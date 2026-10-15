# 二维折线谱图 CNN＋Transformer 识别

当前保留模型为 **T3.32第16轮二维主干＋T3.33第4轮三元DEL/TEB修正**。输入为固定尺度灰度折线谱图与数值峰证据，输出DEL/CHL/TEB存在概率及S/M/H等级。T3.35、T3.36均未超过当前模型，保留为对照。

## 入口与模块

| 路径 | 简要功能 | 状态 |
|---|---|---|
| run_recognition.py | 统一解析仓库依赖、数据和输出路径，转发原入口 | 当前运行入口 |
| current_recognition.py | 加载T3.33内嵌二维主干和保存的图像尺度 | 当前模型加载 |
| extensions/T3_32_Image_Recognition | 固定尺度渲染、二维CNN、Transformer、峰引导和数值辅助 | 二维基础模型 |
| extensions/T3_33_Ternary_Correction | 冻结主干，预测三元时仅修正DEL/TEB | 当前保留版本 |
| extensions/T3_34_Frozen_Diagnosis | 比较真实/生成训练与验证错误及修正幅度限制 | 只读诊断 |
| extensions/T3_35_Ternary_Head_Finetune | 复制DEL/TEB等级头，训练8,580参数 | 未提升对照 |
| extensions/T3_36_Ternary_Fusion_Finetune | 复制融合、投影和等级头，训练148,807参数；含CUDA初始化修复 | 未提升对照 |
| Dataset.py | 原始谱读取、12/4/4固定划分、短轴处理和峰先验 | 原共享数据适配 |
| Model_v2.py及旧训练脚本 | 历史一维基础模型 | 保留兼容和溯源 |

各扩展README保留最初独立补丁的绝对路径说明。克隆仓库后优先使用统一入口，它显式传入仓库中的依赖路径，不需要把代码再复制到`/home/wqzheng/T3_*`。不修改受SHA256保护的原源码。

## 图像表示

共享asinh映射仅由真实训练谱拟合并随权重保存。横轴600–2500 cm⁻¹，2048×512灰度图，白底黑线，不逐条归一化，不绘制未实测尾部，不包含文字或标签。二维CNN使用坐标、纵向位置带和曲线矩保留峰高信息。

主干为二维CNN→210位置token→6层8头128维Transformer→峰引导Query，结合17峰窗数值证据后分别接分类与CORN头。这是图像＋数值辅助模型，不是纯ViT或STFT/CWT。T3.33使用371维冻结特征和23,940参数修正器，路由只看分类预测，不输入真实标签、条件ID或文件名。

## 复用现有服务器项目

默认项目是本目录，数据为`transformer/data/real`与`transformer/data/generated`，输出为`transformer/outputs`。数据、权重、图片与特征缓存不在Git中。

复用现有服务器工件时，所有阶段使用同一`--project /home/wqzheng/project_transformer`。从仓库根执行：

```bash
conda activate sers_ddpm
python transformer/run_recognition.py --stage fusion \
  --project /home/wqzheng/project_transformer --phase check --device cuda --dry-run
```

dry-run只显示解析后的参数，不读取数据或权重。显式传入的baseline/checkpoint/teacher-run/output-directory等参数位于默认参数之后，优先级更高。原缓存指纹包含路径与来源信息，不要通过移动数据和改划分来绕过校验。

## 二维基础训练与当前修正

T3.32复现需要已有T3.30迁移权重。缺失时停止，不自动改为随机初始化。明确使用`--from-scratch`是独立实验，不能称为复现已报告成绩。

```bash
CUDA_VISIBLE_DEVICES=0 python transformer/run_recognition.py --stage image \
  --project /home/wqzheng/project_transformer --phase check \
  --include-generated --maximum-generated-per-condition 48 --device cuda

CUDA_VISIBLE_DEVICES=0 python transformer/run_recognition.py --stage image \
  --project /home/wqzheng/project_transformer --phase train \
  --include-generated --maximum-generated-per-condition 48 \
  --epochs 50 --batch-size 16 --warmup-epochs 3 --device cuda

CUDA_VISIBLE_DEVICES=0 python transformer/run_recognition.py --stage correction \
  --project /home/wqzheng/project_transformer --phase check --device cuda
```

T3.33原入口支持check/smoke/train/evaluate/pack；训练依赖T3.32第16轮报告权重。已有正式输出时拒绝覆盖；新实验应指定新的输出目录。

## 诊断与微调对照

```bash
python transformer/run_recognition.py --stage diagnosis \
  --project /home/wqzheng/project_transformer

python transformer/run_recognition.py --stage heads \
  --project /home/wqzheng/project_transformer --phase check --device cpu

CUDA_VISIBLE_DEVICES=0 python transformer/run_recognition.py --stage fusion \
  --project /home/wqzheng/project_transformer --phase check --device cuda
```

heads/fusion支持smoke/train/evaluate/pack。默认最多20轮，3轮预热＋余弦退火，真实/生成损失75%/25%。三元正确数必须增加，整体等级/完整谱/分类及DEL、TEB各自正确数不得下降。没有合格改善时best_accepted.pt保存epoch=0回退，不应使用latest.pt替代它。

T3.36先缓存融合前Query再训练小分支。CUDA修复在评估时对参数完全相同的新旧分支明确返回零更新，不放宽一致性检查，也不按容差抹掉实际学到的小更新。query_cache.py保持原字节，可复用原查询缓存。

## 当前模型加载

从仓库根导入。权重内嵌完整T3.32模型，部署无需旧T3.32权重或训练缓存。

```python
import torch
from transformer.current_recognition import load_checkpoint

model, plot_scale, metadata = load_checkpoint(
    '/home/wqzheng/project_transformer/outputs/t3_33_ternary_correction_seed2026/train/checkpoints/best_accepted.pt',
    device='cuda',
)
# 用T3.32 LineImageDataset与返回的plot_scale构造相同输入，不重新拟合尺度。
# image_inputs只有image、image_valid_columns、raw_intensity、measured_mask。
with torch.inference_mode():
    output = model(image_inputs)
    levels = 1 + (output['op'] >= 0.5).sum(-1)
    predicted = torch.where(output['cp'] >= 0.5, levels, 0)
```

顺序DEL/CHL/TEB；0表示未检测到，1/2/3对应S/M/H。当前不输出物理单位连续浓度。不要用旧Model_v2.py直接加载新权重。

## 结果与限制

当前验证集：成分组合504/504，存在成分等级1027/1152，完整谱408/504，三元完整133/216。T3.35/36未超过它。T3.36真实训练三元最高80.25%、验证最高61.57%，仍有泛化差距；该版本当前只保留用户终端摘要，未收到正式报告ZIP。

每条件12/4/4为同源文件内部划分，验证集已用于多轮选择，不等于独立测试或跨批次成绩。转图与迁移/训练配置同时改变，不能将全部增益归因于图像。

## 来源与许可

一维基础来源于[SERSFormer2](https://github.com/jianlin-cheng/SERSFormer2)。原引用为Hegde, A., Hajikhani, M., Snyder, J., Cheng, J., & Lin, M. (2024), *Leveraging SERS and transformer models for simultaneous detection of multiple pesticides in fresh produce.* 原许可保留于[LICENSE](LICENSE)。历史T3.30/T3.31扩展保留迁移和对照依赖。

[逐文件功能索引](../FILE_GUIDE.md) · [模型流程与结果](../README.md) · [当前模型状态](../CURRENT_MODEL.json)
