# 块内细节等比例收缩

令 \(T\) 为已对齐块均值的 DPS 平均场，\(B\) 为同样对齐的插值底图。输出 \(B+0.75(T-B)\)。因为每块内 \(T-B\) 的均值为零，缩放残差后仍满足块均观测。原名 `fusion075`。

**接口。** `shrink_details(aligned, baseline, factor=0.75)` 接收同尺寸的二维数组。依赖 NumPy。输入对齐和最终投影由调用方负责。参考源为 `local_work/dps12_integrated_candidate_20261120/engine/postprocess.py`。

**论文关联。** `文本版/20_湍流残差扩散.md` §2.2 也用粗、高分辨率间的残差组织超分问题，其残差进入扩散过程。这里的输出融合以及 0.75 系数是本项目另外设计的经验后处理。

Tabe Jamaat, G., Okatani, T., & Hattori, Y. (2026). High-fidelity turbulent flow field reconstruction using a residual-based diffusion model. *Journal of Fluid Mechanics*, 1035, A3. [DOI](https://doi.org/10.1017/jfm.2026.11511)
