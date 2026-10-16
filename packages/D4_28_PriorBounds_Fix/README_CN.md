# D4.28 固定纵轴内的先验重抽

`run_generation.py` 提供 check / generate / evaluate / pack。默认每条件 50 条、物理 GPU 0、每条最多重抽 100 次。

先验底谱任一有效点越过 checkpoint 保存的固定纵轴时，重新抽取整条同条件先验。已合格底谱保持原值，不逐点裁切、不改变坐标、不筛选生成后的光谱。发生拒绝时后续随机数序列会变化，因此审计记录所有提案、拒绝和重新抽取。

这是将原先验分布条件化到固定坐标支持域的策略变更，不表示原先验分布完全不变。源码、配置和 checkpoint 哈希检查以及原最终交付保护继续执行。

先从 `../../diffusion` 执行测试：

```bash
python ../packages/D4_28_PriorBounds_Fix/test_bounded_prior.py --project "$PWD"
```

运行命令及检查输出位置见 [仓库 README](../../README.md)。两个 D4.28 入口必须保留为相邻目录，以便统一入口调用。
