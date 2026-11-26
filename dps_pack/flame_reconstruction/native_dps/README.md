# 原生固定强度 DPS 接口

`posterior.py` 负责原 DDPM 累计系数到 \([0,1]\) 采样时间的转换、epsilon → 干净潜变量估计、AE 解码与块均观测。HR 行列边界分别为 \(\lfloor i\,192/16\rfloor\) 和 \(\lfloor j\,256/21\rfloor\)。区块均值为区块内像素的算术平均，观测梯度经该算子和解码器反传。

每个反向步由外部 `azula.guidance.dps.DPSSampler` 实现，从 DDPM 提议值减去固定系数 ζ 乘以观测残差 L2 范数的梯度。`upstream.json` 记录版本和实际源码指纹。项目侧代码依据 `local_work/flamediff_native_dps12_downsample/posterior.py` 提取。Azula 指纹基于项目 `azula-master/azula` 对照，运行时通过已安装的软件包导入。

**论文依据。** `文本版/05_DPS.md` §3 给出基于预测干净状态近似似然梯度的 DPS。`文本版/04_CoNFiLD.md` §2.3 将观测映射接在潜变量解码之后，提供潜空间调节的同类思路。本项目使用卷积 AE 解码和非重叠块均观测。

- Chung, H., Kim, J., McCann, M. T., Klasky, M. L., & Ye, J. C. (2023). Diffusion Posterior Sampling for General Noisy Inverse Problems. *International Conference on Learning Representations (ICLR)*. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687)
- Du, P., Parikh, M. H., Fan, X., Liu, X.-Y., & Wang, J.-X. (2024). Conditional neural field latent diffusion model for generating spatiotemporal turbulence. *Nature Communications*, 15, 10416. [DOI](https://doi.org/10.1038/s41467-024-54712-1)
