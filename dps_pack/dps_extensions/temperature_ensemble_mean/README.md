# 多种子温度场平均

同一观测运行多条独立随机种子的完整 DPS 轨迹，逐条解码成 K 单位的温度场，再逐像素计算算术平均。历史名 `K4`、`K2` 分别代表四种子和两种子。平均场近似条件均值，可能弱化各轨迹位置不一致的细结构。

**接口。** `ensemble_mean(members)` 接收 `(成员数, H, W)` 数组，输出 `(H, W)`。依赖 NumPy。参考源为 `local_work/dps12_integrated_candidate_20261120/engine/postprocess.py`。

**论文关联。** `文本版/22_CFM近壁湍流重建与不确定性.md` §3.1–3.2 和式 (3.1) 同样逐点平均由同一观测条件生成的多个场，比较集合均值与 DNS 的重建误差。该文用条件流匹配生成速度场。本项目把同一统计量用于 DPS 温度场，并根据本地采样预算采用四种子。

Parikh, M. H., Fan, X., & Wang, J.-X. (2026). Conditional flow matching for generative modelling of near-wall turbulence with quantified uncertainty. *Journal of Fluid Mechanics*, 1029, A32. [DOI](https://doi.org/10.1017/jfm.2026.11193)
