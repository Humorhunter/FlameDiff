# 按当前去噪幅度缩放观测梯度

令 \(g_t=\nabla_{x_t}\|y-A(\hat z_0)\|_2\)，其中 \(A\) 包括 AE 解码和块均观测。每步从带噪 DDPM 提议值减去
\[
\beta\Big\|\mu_{s|t}-\frac{\alpha_s}{\alpha_t}x_t\Big\|_2\,\frac{g_t}{\|g_t\|_2}.
\]
梯度给出方向，无噪反向均值相对纯缩放项的位移给出幅度。梯度为零时修正为零。

**接口。** `relative_sampler(DPSSampler, beta)` 返回可显式传入的采样器类。`parameters.json` 收录 L5 标定值 0.23745060666879275、500/1000 步配方和固定强度对照。两批试验的最终判读均未达到采用条件。参考源为 `local_work/dps_relative_amplitude/guidance.py`。

**论文关联。** `文本版/25_贝叶斯条件扩散湍流时空生成.md` §2.3.1 式 (30)–(31) 是直接的方法依据。式 (30) 将条件梯度化为单位方向，式 (31) 用无条件分数对反向均值的贡献来定引导幅度，并设幅度下限。本项目沿用“方向和幅度分别确定”的策略，幅度取 Azula DDPM 的上述确定性去噪位移，全程用同一标定 β，不设幅度下限，也不做末步重复精修。

Gao, H., Han, X., Fan, X., Sun, L., Liu, L.-P., Duan, L., & Wang, J.-X. (2024). Bayesian conditional diffusion models for versatile spatiotemporal turbulence generation. *Computer Methods in Applied Mechanics and Engineering*, 427, 117023. [DOI](https://doi.org/10.1016/j.cma.2024.117023)
