# 傅里叶评价方法

- `paper27.py` 评价方向功率谱 NRMSE，按不同方向和频带归约。依据 `文本版/27_谱损失湍流超分.md` 中的谱误差定义。
- `parseval.py` 计算参考和预测的复谱差，输出空间 MSE 的频段分解，并按幅度与幅度加权相位分解。数学约定见 [MATH.md](../MATH.md)。

Cheng, R., Shamooni, A., Zirwes, T., & Kronenburg, A. (2025). Improved super-resolution reconstruction of turbulent flows with spectral loss function. *Physics of Fluids*, 37, 035208. [DOI](https://doi.org/10.1063/5.0258090)
