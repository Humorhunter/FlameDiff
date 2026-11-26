# 区块温度投影

对每个非重叠区块，把区块内所有像素统一加上 \(\bar T_{\mathrm{obs}}-\bar T_{\mathrm{pred}}\)。修正后的区块均温等于观测值，块内像素之间的温差保持不变。在线性块均观测下，它是把温度场按欧氏距离投影到满足 \(AT=y\) 的仿射集合。

**接口。** `project_means(prediction, observations, row_edges, col_edges)`，所用场和观测均为 K。输出二维高分辨率场。依赖 NumPy 和共享区块几何。历史名 `P`，`mean_P` 表示先做多种子温度场平均，再做本投影。参考源为 `local_work/dps12_integrated_candidate_20261120/engine/projection.py`。

**论文关联。** `文本版/05_DPS.md` §2.2、§3.2 和附录 C.1 讨论按观测约束作投影以及噪声放大的风险。原论文的 DPS 每步使用软引导。本项目另外设计了仅在采样结束后执行的块均投影，针对这里当作精确值使用的无噪块均温度。

Chung, H., Kim, J., McCann, M. T., Klasky, M. L., & Ye, J. C. (2023). Diffusion Posterior Sampling for General Noisy Inverse Problems. *International Conference on Learning Representations (ICLR)*. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)
