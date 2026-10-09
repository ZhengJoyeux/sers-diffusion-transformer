# T3.24 zero-reproduction修复

终端表明15项check已通过，但CUDA smoke在缓存完成后进行初始化复现检查时失败。没有进入正式训练，原项目源码和权重完整。原检查对全部概率使用逐位相等比较，没有记录触发失败的列及实际差值，因此仅凭原日志不能断定这次服务器差值的大小。

在CPU人工边界样本中，已复现权重相同而重算logit差异约6e-8、足以触发检查或阈值翻转的同类问题。这是支持数值误差解释的证据；没有在本地执行CUDA复现。

修复措施：

1. 对原融合与更新副本的完整状态做严格比较。融合以及某个药物头的权重/缓冲完全相同时，将该路径的数值变化规范为零；通过减去detach的数值误差保留训练导数。发生实际参数更新后仍使用真实预测变化，不能用“小于某个阈值”抹去学到的更新。
2. 总变化严格为零的药物直接复用缓存或原在线模型的概率，避免重新计算sigmoid时改变0.5附近的原等级。
3. 初始化审计仍要求样本身份、类别概率和全部分类/等级决定一致；概率只允许atol=1e-6、rtol=0，初始分支变化只允许1e-6。错误初始化权重、非零残差输出、等级改变、大误差、非有限值仍被拒绝。
4. 新增zero_reproduction_audit.json，记录逐列误差及失败原因；失败时日志也显示差值。
5. check入口运行原15项及新增8项回归，共23项。没有改变可训练范围、学习率、数据划分、生成数量、损失或正式采用规则。T3.24检查点格式和状态键兼容旧版本。

下载到/home/wqzheng/T3_24_zero_reproduction_fix.zip后执行：

```bash
conda activate sers_ddpm
cd /home/wqzheng/project_transformer

unzip -n /home/wqzheng/T3_24_zero_reproduction_fix.zip -d /home/wqzheng && \
python /home/wqzheng/T3_24_zero_reproduction_fix/apply_fix.py \
  --package-directory /home/wqzheng/T3_24_concentration_update && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_24_concentration_update/run_t324.sh check && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_24_concentration_update/run_t324.sh smoke && \
CUDA_VISIBLE_DEVICES=0 bash /home/wqzheng/T3_24_concentration_update/run_t324.sh train
```

安装器验证原包内容、保存被替换文件，并更新包内指纹；支持重复执行。无需删除失败的smoke目录，每次运行会创建新的时间戳目录。smoke成功后才会自动正式训练。

本修复仍是局部训练，两组各50轮、batch16，最高学习率1e-5。现有原模型和旧实验检查点均不覆盖。若误差或等级确实改变，新检查仍停止；请保留新的详细错误和zero_reproduction_audit.json。
