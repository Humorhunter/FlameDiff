# 粗温度插值参照

以观测块的实际几何中心为节点，先沿列、再沿行线性插值到高分辨率像素中心。超出最外侧块中心的查询使用边缘值。可由组合入口再做一次区块温度投影，得到块均温与观测一致的非神经网络对照。

**接口。** `interpolate(observations, (row_edges, col_edges))` 返回二维温度场。依赖 NumPy 和共享区块几何。参考源为 `local_work/dps12_integrated_candidate_20261120/engine/evaluate.py`。

**论文关联。** `文本版/02_课题组火焰LDM前作.md` 的图 11 用双三次插值作为火焰超分的频谱对照。这里沿用设置插值对照的思路。按实际块中心作分离线性插值和后续块均对齐是本项目的实现选择。

Wu, S., Liang, W., & Luo, K. H. (2025). Multi-task latent diffusion model for reconstructing high-fidelity turbulent non-premixed NH₃/H₂/N₂ flames from sparse observations. *Combustion and Flame*, 282, 114469. [DOI](https://doi.org/10.1016/j.combustflame.2025.114469)
