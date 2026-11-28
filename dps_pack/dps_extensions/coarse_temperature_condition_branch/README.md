# 粗温度条件分支

把粗观测按历史范围 304.6579284667969–1617.9052734375 K 归一化，由双线性插值送到 48×64 潜空间网格。两层小卷积输出 128 通道特征，直接加到去噪网络第一段卷积的输出，再送入后续网络。末层零初始化。训练冻结 AE 和扩散先验，仅按噪声预测 MSE 更新分支。历史名 `A600` 表示分支训练 600 次更新，Adam 学习率 0.0001，种子 42。

**接口。** `load_condition(model, path, expected_base)` 接收已校验的历史模型和分支权重，返回可传给采样入口的条件函数。`train_branch(model, operator, frames, ...)` 接收冻结模型和外部 K 单位 `B×1×192×256` 批次，迭代次数由传入批次数决定。`adapter.py` 为分支结构。`model_config.json` 记录权重指纹，组合入口据此加载及检查配套关系。参考 `local_work/dps12_condition_lab` 与 `local_work/dps12_integrated_candidate_20261120/engine/core`。

**论文关联。** `文本版/25_贝叶斯条件扩散湍流时空生成.md` §2.3.2 给出通过训练把条件直接输入去噪网络的思路，采用拼接条件并训练条件扩散模型。本项目另外设计了冻结旧主干、训练小卷积分支并以特征加法注入的实现，600 次更新为本地设置。

Gao, H., Han, X., Fan, X., Sun, L., Liu, L.-P., Duan, L., & Wang, J.-X. (2024). Bayesian conditional diffusion models for versatile spatiotemporal turbulence generation. *Computer Methods in Applied Mechanics and Engineering*, 427, 117023. [DOI](https://doi.org/10.1016/j.cma.2024.117023)
