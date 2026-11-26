# 块内细节分尺度收缩

对已对齐的 DPS 平均场和插值底图取残差 \(R\)。用标准差 1 和 3 像素的高斯滤波拆成 \(G_3R,\ G_1R-G_3R,\ R-G_1R\)。滤波边界为 `reflect`，`truncate=4`。各频带去掉各自的块均值，再以 0.90、0.75、0.90 加回底图，最后投影。历史名 `gentle_low`。新基座探索中，点温度误差下降，功率谱误差上升，因此作为可选方法保留。

**接口。** `shrink_details(aligned, baseline, observations, edges, project, weights=...)`。依赖 NumPy 与 SciPy。调用方显式提供投影函数。参考源为 `local_work/dps12_integrated_candidate_20261120/engine/postprocess.py`。

**论文关联。** `文本版/27_谱损失湍流超分.md` 提供在波数维度检查尺度误差的依据。该文将谱损失用于训练。这里的高斯频带、逐块去均值和三组收缩权重都是本项目设计的推理后处理。

Cheng, R., Shamooni, A., Zirwes, T., & Kronenburg, A. (2025). Improved super-resolution reconstruction of turbulent flows with spectral loss function. *Physics of Fluids*, 37, 035208. [DOI](https://doi.org/10.1063/5.0258090)
